"""Metrics, index, search service and id -> file mapping."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from src.data.features import load_features
from src.evaluation.metrics import (evaluate_embeddings, evaluate_protocol, l2_normalize, ranks_image_to_text,
                                    ranks_text_to_image, recall_at_k, summarize_ranks)
from src.retrieval.index import ImageIndex, IndexError_
from src.retrieval.models import ModelArtifactError, load_retriever
from src.retrieval.search import (ImageStore, InvalidQueryError, ModelNotFoundError, ModelUnavailableError,
                                  SearchService, search_images)


# ------------------------------------------------------------------ metrics
def test_text_to_image_ranks_on_a_hand_built_case():
    img = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    txt = np.array([[0.8, 0.6], [0, 1], [0.8, 0.6]], dtype=np.float32)  # captions of images 0, 1, 2
    gt = np.array([0, 1, 2])
    ranks = ranks_text_to_image(txt, img, gt)
    # caption 0 = [0.8, 0.6]: scores vs images = 0.80, 0.60, 0.96 -> its own image 0 is second -> rank 2
    # caption 1 = [0, 1]:     scores = 0.00, 1.00, 0.80            -> image 1 is best        -> rank 1
    # caption 2 = [0.8, 0.6]: scores = 0.80, 0.60, 0.96            -> image 2 is best        -> rank 1
    assert ranks.tolist() == [2, 1, 1]
    r = recall_at_k(ranks, (1, 2))
    assert r[1] == pytest.approx(200 / 3) and r[2] == 100.0
    s = summarize_ranks(ranks)
    assert s["mean_rank"] == pytest.approx(4 / 3) and s["median_rank"] == 1


def test_ties_are_ranked_pessimistically():
    img = np.ones((4, 3), dtype=np.float32)
    txt = np.ones((4, 3), dtype=np.float32)
    assert ranks_text_to_image(txt, img, np.arange(4)).tolist() == [4, 4, 4, 4]  # constant scores give no credit
    assert ranks_image_to_text(img, txt, np.arange(4)).tolist() == [4, 4, 4, 4]


def test_image_to_text_uses_best_ranked_ground_truth_caption():
    img = np.array([[1, 0], [0, 1]], dtype=np.float32)
    # image 0 owns captions 0 (weak, 0.6) and 1 (perfect, 1.0); image 1 owns captions 2 (perfect) and 3 (weak)
    txt = np.array([[0.6, 0.8], [1, 0], [0, 1], [0.8, 0.6]], dtype=np.float32)
    ranks = ranks_image_to_text(img, txt, np.array([0, 0, 1, 1]))
    assert ranks.tolist() == [1, 1]  # the BEST ground-truth caption counts, the weak one is ignored
    # when even the best own caption is outranked by another image's caption, the rank is > 1
    txt2 = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
    assert ranks_image_to_text(img, txt2, np.array([0, 1])).tolist() == [2, 2]


def test_perfect_embeddings_give_perfect_scores_and_validation_errors():
    emb = l2_normalize(np.random.RandomState(0).randn(10, 16))
    cap_emb = np.repeat(emb, 5, axis=0)
    idx = np.repeat(np.arange(10), 5)
    m = evaluate_embeddings(emb, cap_emb, idx)
    assert m["t2i_R@1"] == 100.0 and m["i2t_R@1"] == 100.0 and m["t2i_mean_rank"] == 1.0 and m["t2i_rsum"] == 300.0
    with pytest.raises(ValueError):
        evaluate_embeddings(emb, cap_emb[:, :8], idx)
    with pytest.raises(ValueError):
        evaluate_embeddings(emb, cap_emb, idx + 100)
    with pytest.raises(ValueError):
        evaluate_embeddings(emb, cap_emb[:-1], idx)


def test_random_embeddings_are_near_chance():
    rng = np.random.RandomState(1)
    img, txt = l2_normalize(rng.randn(200, 32)), l2_normalize(rng.randn(1000, 32))
    m = evaluate_embeddings(img, txt, np.repeat(np.arange(200), 5))
    assert 0.0 <= m["t2i_R@10"] < 15  # chance is 5 %
    assert 80 < m["t2i_mean_rank"] < 120  # chance is ~100.5


def test_one_k_protocol_folds():
    rng = np.random.RandomState(2)
    emb = l2_normalize(rng.randn(2000, 8))
    cap = np.repeat(emb, 5, axis=0) + 0.01
    idx = np.repeat(np.arange(2000), 5)
    res = evaluate_protocol(emb, l2_normalize(cap), idx, "1k")
    assert res["num_folds"] == 2 and res["num_images"] == 2000 and res["t2i_R@1"] > 99
    full = evaluate_protocol(emb, l2_normalize(cap), idx, "full")
    assert full["protocol"] == "full" and full["num_captions"] == 10000
    with pytest.raises(ValueError):
        evaluate_protocol(emb[:100], l2_normalize(cap[:500]), idx[:500], "1k")
    with pytest.raises(ValueError):
        evaluate_protocol(emb, cap, idx, "bogus")


# ------------------------------------------------------------------ index
def _index(n=50, d=16, seed=0, **kw):
    emb = np.random.RandomState(seed).randn(n, d).astype(np.float32)
    return ImageIndex(emb, [f"img{i}" for i in range(n)], **kw), emb


def test_exact_search_matches_brute_force_and_is_sorted():
    idx, emb = _index()
    q = np.random.RandomState(5).randn(3, 16).astype(np.float32)
    scores, pos = idx.search(q, 7)
    brute = l2_normalize(q) @ l2_normalize(emb).T
    for r in range(3):
        assert pos[r].tolist() == np.argsort(-brute[r], kind="stable")[:7].tolist()
        assert np.all(np.diff(scores[r]) <= 1e-7)
    assert idx.search(q[0], 1000)[1].shape == (1, 50)  # k larger than the corpus is clipped
    assert idx.backend == "exact"


def test_index_validation_errors():
    idx, _ = _index()
    with pytest.raises(IndexError_):
        idx.search(np.zeros(8, dtype=np.float32), 3)  # wrong dimension
    with pytest.raises(IndexError_):
        idx.search(np.full(16, np.nan, dtype=np.float32), 3)
    with pytest.raises(IndexError_):
        ImageIndex(np.zeros((2, 4), dtype=np.float32), ["a", "a"])  # duplicate ids
    with pytest.raises(IndexError_):
        ImageIndex(np.zeros((2, 4), dtype=np.float32), ["a"])  # id count mismatch
    with pytest.raises(IndexError_):
        ImageIndex(np.full((1, 4), np.inf, dtype=np.float32), ["a"])
    with pytest.raises(IndexError_):
        ImageIndex(np.zeros((0, 4), dtype=np.float32), []).search(np.ones(4, dtype=np.float32))  # empty index


def test_index_add_and_persistence(tmp_path):
    idx, _ = _index(n=10)
    extra = np.random.RandomState(9).randn(2, 16).astype(np.float32)
    idx.add(extra, ["new1", "new2"])
    assert idx.size == 12 and idx.ids[-1] == "new2"
    with pytest.raises(IndexError_):
        idx.add(extra, ["new1", "new3"])  # id clash
    with pytest.raises(IndexError_):
        idx.add(np.zeros((1, 5), dtype=np.float32), ["x"])  # dimension mismatch
    q = extra[0]
    assert idx.ids[int(idx.search(q, 1)[1][0, 0])] == "new1"  # an added image is retrievable
    idx.save(tmp_path / "ix")
    again = ImageIndex.load(tmp_path / "ix")
    assert again.ids == idx.ids and np.allclose(again.embeddings, idx.embeddings)
    assert json.loads((tmp_path / "ix" / "index_meta.json").read_text())["size"] == 12
    with pytest.raises(FileNotFoundError):
        ImageIndex.load(tmp_path / "nothing")


def test_index_load_rejects_metadata_mismatch(tmp_path):
    idx, _ = _index(n=4, d=3)
    target = tmp_path / "ix"
    idx.save(target)
    meta = json.loads((target / "index_meta.json").read_text())
    meta["size"] = 3
    (target / "index_meta.json").write_text(json.dumps(meta))
    with pytest.raises(IndexError_, match="metadata"):
        ImageIndex.load(target)


def test_faiss_and_exact_backends_agree():
    pytest.importorskip("faiss")
    exact, emb = _index(n=300, d=32, prefer_faiss=False)
    fast = ImageIndex(emb, exact.ids, prefer_faiss=True, faiss_min_size=0)
    assert fast.backend == "faiss"
    q = np.random.RandomState(4).randn(5, 32).astype(np.float32)
    s1, p1 = exact.search(q, 10)
    s2, p2 = fast.search(q, 10)
    assert np.array_equal(p1, p2) and np.allclose(s1, s2, atol=1e-5)


# ------------------------------------------------------------------ search service on the smoke workspace
def test_index_maps_each_id_to_the_right_image_embedding(smoke_ws):
    """The row stored for an id must be the embedding of THAT image's features."""
    name = "smoke_baseline"
    run = smoke_ws["artifacts"] / name
    index = ImageIndex.load(run / "index")
    feats, ids = load_features(smoke_ws["processed"] / "features", "test")
    assert index.ids == ids
    retr = load_retriever(run)
    for i in (0, len(ids) // 2, len(ids) - 1):
        assert np.allclose(index.embeddings[index.ids.index(ids[i])], retr.encode_image_features(feats[i:i + 1])[0], atol=1e-6)


def test_search_returns_ranked_results_with_existing_files_and_captions(smoke_ws):
    svc = SearchService.from_artifacts(smoke_ws["artifacts"], smoke_ws["processed"] / "metadata.json",
                                       smoke_ws["images_root"])
    assert svc.startup_warnings == [] and svc.default_model == "smoke_baseline"
    res = svc.search("a red circle on a white background", top_k=5)
    assert res["num_results"] == 5 and res["model"] == "smoke_baseline" and res["score_type"] == "cosine_similarity"
    scores = [r["score"] for r in res["results"]]
    assert scores == sorted(scores, reverse=True) and [r["rank"] for r in res["results"]] == [1, 2, 3, 4, 5]
    for r in res["results"]:
        path = svc.image_store.path_for(r["image_id"])
        assert path is not None and path.is_file() and path.name == r["file_name"].split("/")[-1]
        assert r["image_available"] and len(r["captions"]) == 5 and r["image_url"] == f"/api/images/{r['image_id']}"
    assert {r["split"] for r in res["results"]} == {"test"}
    assert search_images("blue square", 3, service=svc)["num_results"] == 3  # module-level function


def test_a_query_word_that_helps_changes_the_ranking(smoke_ws):
    """The text really influences the ranking (it is not a constant list)."""
    svc = SearchService.from_artifacts(smoke_ws["artifacts"], smoke_ws["processed"] / "metadata.json", smoke_ws["images_root"])
    a = [r["image_id"] for r in svc.search("a large red circle on black background", 30)["results"]]
    b = [r["image_id"] for r in svc.search("a small yellow triangle on white background", 30)["results"]]
    assert sorted(a) == sorted(b) and a != b


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "!!! ???", None, 42, "x" * 400])
def test_invalid_queries_raise_clean_errors(smoke_ws, query):
    svc = SearchService.from_artifacts(smoke_ws["artifacts"], smoke_ws["processed"] / "metadata.json", smoke_ws["images_root"])
    with pytest.raises(InvalidQueryError):
        svc.search(query, 5)


@pytest.mark.parametrize("top_k", [0, -1, 101, "5", 2.5, None, True])
def test_invalid_top_k_raises(smoke_ws, top_k):
    svc = SearchService.from_artifacts(smoke_ws["artifacts"], smoke_ws["processed"] / "metadata.json", smoke_ws["images_root"])
    with pytest.raises(InvalidQueryError):
        svc.search("a red circle", top_k)


def test_top_k_larger_than_corpus_returns_whole_corpus_and_unknown_words_are_reported(smoke_ws):
    svc = SearchService.from_artifacts(smoke_ws["artifacts"], smoke_ws["processed"] / "metadata.json", smoke_ws["images_root"])
    res = svc.search("a red giraffe", 100)
    assert res["num_results"] == 30 and res["unknown_tokens"] == ["giraffe"]
    with pytest.raises(InvalidQueryError, match="vocabulary"):
        svc.search("giraffe elephant", 3)  # nothing the model knows


def test_unknown_and_unavailable_models(smoke_ws, tmp_path):
    svc = SearchService.from_artifacts(smoke_ws["artifacts"], smoke_ws["processed"] / "metadata.json", smoke_ws["images_root"])
    with pytest.raises(ModelNotFoundError):
        svc.search("a red circle", 3, model="does_not_exist")
    # a model folder whose files are missing must be reported as unavailable, not crash startup
    broken = tmp_path / "artifacts" / "gru_broken"
    broken.mkdir(parents=True)
    (broken / "model_metadata.json").write_text(json.dumps({"name": "gru_broken", "architecture": {"type": "gru"}}))
    svc2 = SearchService.from_artifacts(tmp_path / "artifacts", smoke_ws["processed"] / "metadata.json", smoke_ws["images_root"])
    info = svc2.list_models()[0]
    assert info["available"] is False and info["unavailable_reason"]
    with pytest.raises(ModelUnavailableError):
        svc2.search("a red circle", 3)


def test_search_service_rejects_an_index_built_from_another_checkpoint(smoke_ws, tmp_path):
    copied = tmp_path / "artifacts"
    shutil.copytree(smoke_ws["artifacts"], copied)
    index_meta = copied / "smoke_baseline" / "index" / "index_meta.json"
    data = json.loads(index_meta.read_text())
    data["checkpoint_sha256"] = "0" * 64
    index_meta.write_text(json.dumps(data))
    svc = SearchService.from_artifacts(copied, smoke_ws["processed"] / "metadata.json", smoke_ws["images_root"])
    model = svc.list_models()[0]
    assert not model["available"]
    assert "checkpoint hash does not match" in model["unavailable_reason"]


def test_retriever_rejects_a_modified_checkpoint(smoke_ws, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(smoke_ws["artifacts"] / "smoke_baseline", run)
    with (run / "baseline.npz").open("ab") as checkpoint:
        checkpoint.write(b"modified")
    with pytest.raises(ModelArtifactError, match="checkpoint hash"):
        load_retriever(run)


def test_missing_artifacts_and_metadata_do_not_crash_startup(tmp_path):
    svc = SearchService.from_artifacts(tmp_path / "none", tmp_path / "none.json")
    assert svc.models == {} and len(svc.startup_warnings) == 2
    assert all(str(tmp_path) not in warning for warning in svc.startup_warnings)
    with pytest.raises(ModelUnavailableError):
        svc.search("a red circle", 3)


def test_model_loading_errors_do_not_disclose_workspace_paths(tmp_path):
    artifacts = tmp_path / "artifacts"
    run = artifacts / "broken"
    run.mkdir(parents=True)
    (run / "model_metadata.json").write_text(json.dumps({"name": "broken", "architecture": {"type": "unknown"}}))
    svc = SearchService.from_artifacts(artifacts, tmp_path / "missing-metadata.json")
    model = svc.list_models()[0]
    assert model["available"] is False
    assert str(tmp_path) not in model["unavailable_reason"]


def test_missing_image_file_is_reported_per_result_without_failing(smoke_ws, tmp_path):
    root = tmp_path / "imgs"
    shutil.copytree(smoke_ws["images_root"], root)
    svc = SearchService.from_artifacts(smoke_ws["artifacts"], smoke_ws["processed"] / "metadata.json", root)
    first = svc.search("a red circle", 5)["results"][0]
    (root / first["file_name"]).unlink()
    res = svc.search("a red circle", 5)
    flags = {r["image_id"]: r["image_available"] for r in res["results"]}
    assert flags[first["image_id"]] is False and sum(flags.values()) == 4


def test_image_store_never_escapes_its_root(tmp_path):
    from src.data.records import ImageRecord

    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    (root / "ok.png").write_bytes(b"x")
    store = ImageStore(root, [ImageRecord("a", "ok.png", ["c"]), ImageRecord("evil", "../secret.txt", ["c"])])
    assert store.path_for("a") == (root / "ok.png").resolve()
    assert store.path_for("evil") is None and store.path_for("../secret.txt") is None
    assert store.path_for("unknown") is None
