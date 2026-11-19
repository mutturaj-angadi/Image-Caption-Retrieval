"""Dataset loading, validation, text processing and feature caching."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.data.features import (FeatureExtractionError, SimpleImageFeatures, check_features, extract_split_features,
                               load_features)
from src.data.pairs import build_split_data
from src.data.prepare import prepare_dataset
from src.data.records import (ImageRecord, load_coco_captions, load_karpathy, make_random_split,
                              standardize_captions)
from src.data.splits import subset_records
from src.data.synthetic import make_synthetic_dataset
from src.data.text import (UNK_ID, Vocabulary, clean_caption, load_glove_matrix, mean_word_vectors,
                           synthetic_word_vectors, tokenize)
from src.data.validate import DataLeakageError, check_no_leakage, validate_records


# ------------------------------------------------------------------ text
def test_tokenize_and_clean():
    assert clean_caption("  A Dog,  RUNNING!\n") == "a dog, running!"
    assert tokenize("A dog, running through snow.") == ["a", "dog", "running", "through", "snow"]
    assert tokenize("!!!") == []


def test_vocab_is_deterministic_and_train_frequency_ordered():
    toks = [["a", "dog"], ["a", "cat"], ["a", "dog"]]
    v1, v2 = Vocabulary.build(toks), Vocabulary.build(list(reversed(toks)))
    assert v1.itos == v2.itos == ["<pad>", "<unk>", "a", "dog", "cat"]
    assert Vocabulary.build(toks, min_freq=2).itos == ["<pad>", "<unk>", "a", "dog"]


def test_encode_batch_padding_lengths_truncation_and_unknowns():
    v = Vocabulary.build([["a", "dog", "runs"]])
    ids, lens = v.encode_batch(["a dog runs fast", "dog", "???"], max_len=3)
    assert ids.shape == (3, 3) and ids.dtype == np.int32
    assert lens.tolist() == [3, 1, 1]  # truncated; empty caption becomes a single <unk>
    assert ids[1].tolist() == [v.stoi["dog"], 0, 0]
    assert ids[2, 0] == UNK_ID
    assert v.encode_batch(["a dog"], max_len=8, pad_to=8)[0].shape == (1, 8)
    assert v.unknown_tokens(["a", "zebra"]) == ["zebra"]


def test_vocab_save_load_roundtrip(tmp_path):
    v = Vocabulary.build([["x", "y", "y"]])
    v.save(tmp_path / "v.json")
    assert Vocabulary.load(tmp_path / "v.json").itos == v.itos
    with pytest.raises(ValueError):
        Vocabulary(["a", "b"])  # must start with <pad>, <unk>


def test_glove_loader_coverage_and_errors(tmp_path):
    v = Vocabulary.build([["cat", "dog", "zebra"]])
    glove = tmp_path / "glove.txt"
    glove.write_text("cat 1 2 3\ndog 4 5 6\nother 7 8 9\n", encoding="utf-8")
    mat, stats = load_glove_matrix(glove, v, dim=3)
    assert mat.shape == (len(v), 3)
    assert mat[v.stoi["cat"]].tolist() == [1, 2, 3]
    assert mat[0].tolist() == [0, 0, 0]  # padding row is zero
    assert stats["num_found"] == 2 and stats["num_missing"] == 1 and abs(stats["coverage"] - 2 / 3) < 1e-9
    with pytest.raises(ValueError):
        load_glove_matrix(glove, v, dim=4)
    with pytest.raises(FileNotFoundError):
        load_glove_matrix(tmp_path / "nope.txt", v, dim=3)


def test_synthetic_vectors_are_deterministic_and_mean_vectors_handle_unknowns():
    v = Vocabulary.build([["a", "b"]])
    m1, _ = synthetic_word_vectors(v, 8, seed=1)
    m2, _ = synthetic_word_vectors(v, 8, seed=1)
    assert np.array_equal(m1, m2) and not np.array_equal(m1, synthetic_word_vectors(v, 8, seed=2)[0])
    feats, known = mean_word_vectors(["a b", "zzz", "a zzz"], v, m1)
    assert known.tolist() == [2, 0, 1]
    assert np.allclose(feats[0], (m1[v.stoi["a"]] + m1[v.stoi["b"]]) / 2)
    assert np.all(feats[1] == 0)


# ------------------------------------------------------------------ records
def test_karpathy_loader_merges_restval_into_train(tmp_path):
    data = {"images": [
        {"filename": "a.jpg", "filepath": "train2014", "cocoid": 1, "split": "train", "sentences": [{"raw": "x"}]},
        {"filename": "b.jpg", "filepath": "val2014", "cocoid": 2, "split": "restval", "sentences": [{"raw": "y"}]},
        {"filename": "c.jpg", "filepath": "val2014", "cocoid": 3, "split": "test", "sentences": [{"raw": "z"}]},
        {"filename": "d.jpg", "filepath": "val2014", "cocoid": 4, "split": "weird", "sentences": []},
    ]}
    p = tmp_path / "k.json"
    p.write_text(json.dumps(data))
    recs = load_karpathy(p)
    assert [(r.image_id, r.split, r.file_name) for r in recs] == [
        ("1", "train", "train2014/a.jpg"), ("2", "train", "val2014/b.jpg"), ("3", "test", "val2014/c.jpg")]
    with pytest.raises(ValueError):
        p.write_text("{}")
        load_karpathy(p)


def test_coco_captions_loader(tmp_path):
    data = {"images": [{"id": 7, "file_name": "x.jpg"}],
            "annotations": [{"image_id": 7, "id": 1, "caption": " A cat. "}, {"image_id": 7, "id": 2, "caption": "b"}]}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data))
    recs = load_coco_captions(p, "train2014")
    assert recs[0].file_name == "train2014/x.jpg" and recs[0].captions == ["A cat.", "b"]


def test_random_split_is_deterministic_and_sized():
    recs = [ImageRecord(str(i), f"{i}.jpg", ["c"] * 5) for i in range(100)]
    a, b = make_random_split(recs, 10, 10, seed=3), make_random_split(recs, 10, 10, seed=3)
    assert [r.split for r in a] == [r.split for r in b]
    counts = {s: sum(r.split == s for r in a) for s in ("train", "val", "test")}
    assert counts == {"train": 80, "val": 10, "test": 10}
    assert [r.split for r in a] != [r.split for r in make_random_split(recs, 10, 10, seed=4)]
    with pytest.raises(ValueError):
        make_random_split(recs, 60, 60)


def test_standardize_captions():
    recs = [ImageRecord("1", "1.jpg", [" a ", "", "b", "c", "d", "e", "f"]), ImageRecord("2", "2.jpg", ["only one"])]
    kept, rep = standardize_captions(recs, 5)
    assert len(kept) == 1 and kept[0].captions == ["a", "b", "c", "d", "e"]
    assert rep["images_dropped_too_few_captions"] == 1 and rep["empty_captions_removed"] == 1


# ------------------------------------------------------------------ validation
def test_validate_detects_missing_corrupted_duplicate_and_empty(tiny_images):
    root, recs = tiny_images
    (root / "bad.png").write_bytes(b"this is not an image")
    extra = [
        ImageRecord("missing", "nope.png", ["c"], "train"),
        ImageRecord("bad", "bad.png", ["c"], "train"),
        ImageRecord("id0", "im0.png", ["dup id"], "train"),          # duplicate id
        ImageRecord("nocap", "im1.png", [], "val"),                  # no captions
        ImageRecord("badsplit", "im1.png", ["c"], "holdout"),        # invalid split
    ]
    valid, rep = validate_records(recs + extra, root)
    assert [r.image_id for r in valid] == ["id0", "id1", "id2"]
    assert rep["num_missing"] == 1 and rep["num_corrupted"] == 1
    assert rep["duplicate_image_ids_sample"] == ["id0"] and rep["no_captions_sample"] == ["nocap"]
    assert rep["invalid_split"] == ["badsplit"]


def test_duplicate_content_across_splits_keeps_evaluation_copy(tmp_path):
    root = tmp_path / "i"
    root.mkdir()
    Image.new("RGB", (8, 8), (5, 5, 5)).save(root / "a.png")
    (root / "b.png").write_bytes((root / "a.png").read_bytes())  # byte-identical copy
    recs = [ImageRecord("tr", "a.png", ["c"], "train"), ImageRecord("te", "b.png", ["c"], "test")]
    valid, rep = validate_records(recs, root, hash_duplicates=True)
    assert [r.image_id for r in valid] == ["te"]  # the train copy is dropped, the test image is kept
    assert len(rep["duplicate_content_groups"]) == 1


def test_leakage_check():
    ok = [ImageRecord("1", "1", ["c"], "train"), ImageRecord("2", "2", ["c"], "val")]
    check_no_leakage(ok)
    with pytest.raises(DataLeakageError):
        check_no_leakage(ok + [ImageRecord("1", "1", ["c"], "test")])


def test_subset_is_deterministic_per_split_and_never_mixes_splits():
    recs = [ImageRecord(f"{i:03d}", "f", ["c"], "train" if i < 60 else "val" if i < 80 else "test") for i in range(100)]
    a = subset_records(recs, {"train": 10, "val": 5, "test": None}, seed=1)
    b = subset_records(recs, {"train": 10, "val": 5, "test": None}, seed=1)
    assert [r.image_id for r in a] == [r.image_id for r in b]
    assert {s: sum(r.split == s for r in a) for s in ("train", "val", "test")} == {"train": 10, "val": 5, "test": 20}
    assert all(r.split == ("train" if int(r.image_id) < 60 else "val" if int(r.image_id) < 80 else "test") for r in a)


# ------------------------------------------------------------------ preparation + features
def test_prepare_dataset_builds_vocab_from_train_only(tmp_path):
    records, root = make_synthetic_dataset(tmp_path / "syn", 12, 4, 4)
    records[-1].captions[0] = "zyxwv unseenword"  # a test-only word
    out = prepare_dataset(records, root, tmp_path / "proc", dataset_name="t", split_source="generated",
                          synthetic=True, min_freq=1)
    vocab = Vocabulary.load(tmp_path / "proc" / "vocab.json")
    assert "unseenword" not in vocab.stoi and "circle" in vocab.stoi
    assert out["dataset"]["num_images"] == {"train": 12, "val": 4, "test": 4}
    for name in ("metadata.json", "stats.json", "validation_report.json", "vocab.json"):
        assert (tmp_path / "proc" / name).is_file()
    sub = prepare_dataset(records, root, tmp_path / "proc2", dataset_name="t", split_source="generated",
                          min_freq=1, max_per_split={"train": 6, "val": None, "test": 2})
    assert sub["dataset"]["is_subset"] is True and sub["dataset"]["num_images"]["train"] == 6


def test_simple_features_shape_validity_and_distinguish_colours(tiny_images):
    root, recs = tiny_images
    feats = SimpleImageFeatures().extract([root / r.file_name for r in recs])
    assert feats.shape == (3, 256) and feats.dtype == np.float32 and np.isfinite(feats).all()
    assert not np.allclose(feats[0], feats[1])
    with pytest.raises(FeatureExtractionError):
        SimpleImageFeatures().extract([root / "missing.png"])
    with pytest.raises(FeatureExtractionError):
        check_features(np.full((2, 4), np.nan, dtype=np.float32), 2, 4)
    with pytest.raises(FeatureExtractionError):
        check_features(np.zeros((2, 4), dtype=np.float32), 3, 4)


def test_feature_cache_alignment_and_reuse(tmp_path):
    records, root = make_synthetic_dataset(tmp_path / "syn", 6, 3, 3)
    ex = SimpleImageFeatures()
    first = extract_split_features(records, root, tmp_path / "f", ex)
    assert first["splits"]["train"]["cached"] is False
    second = extract_split_features(records, root, tmp_path / "f", ex)
    assert second["splits"]["train"]["cached"] is True
    feats, ids = load_features(tmp_path / "f", "val")
    assert feats.shape == (3, 256) and ids == sorted(r.image_id for r in records if r.split == "val")
    row = ex.extract([root / next(r.file_name for r in records if r.image_id == ids[1])])[0]
    assert np.allclose(feats[1], row)  # row i really belongs to ids[i]
    with pytest.raises(FileNotFoundError):
        load_features(tmp_path / "empty", "train")


def test_build_split_data_aligns_captions_with_feature_rows(smoke_ws):
    from src.data.pairs import load_records

    recs, _ = load_records(smoke_ws["processed"] / "metadata.json")
    vocab = Vocabulary.load(smoke_ws["processed"] / "vocab.json")
    feats, ids = load_features(smoke_ws["processed"] / "features", "test")
    data = build_split_data("test", recs, ids, feats, vocab, 16)
    assert data.num_images == len(ids) and data.num_captions == 5 * len(ids)
    by_id = {r.image_id: r for r in recs}
    for j in (0, 7, data.num_captions - 1):
        assert data.captions[j] in by_id[ids[data.caption_img_idx[j]]].captions
    with pytest.raises(ValueError):
        build_split_data("train", recs, ids, feats, vocab, 16)  # test ids are not train records
