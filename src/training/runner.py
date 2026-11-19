"""Orchestration used by the command-line scripts, the smoke test and the tests.

* :func:`train_model`     - fit the baseline or train a GRU / LSTM on the TRAIN split, select on VALIDATION.
* :func:`evaluate_run`    - evaluate a saved model on a chosen split (test only when explicitly requested).
* :func:`build_run_index` - embed the images of a split and save the retrieval index.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.data.features import load_features
from src.data.pairs import SplitData, build_split_data, load_records
from src.data.text import (Vocabulary, load_glove_matrix, mean_word_vectors, random_word_vectors,
                           synthetic_word_vectors)
from src.evaluation.metrics import evaluate_embeddings, evaluate_protocol, format_metrics, higher_is_better
from src.models.baseline import LinearRegressionBaseline
from src.retrieval.index import ImageIndex
from src.retrieval.models import load_retriever, read_metadata
from src.utils.common import (environment_info, get_logger, load_json, resolve_path, save_json, set_seed,
                              sha256_file, torch_available)

log = get_logger("ncir.runner")
ARCH_VERSION = "ncir-model-metadata-v1"


def run_dir_for(cfg: Mapping[str, Any]) -> Path:
    return resolve_path(cfg.get("artifacts_dir", "artifacts")) / str(cfg["name"])


def load_split_data(cfg: Mapping[str, Any], split: str, vocab: Vocabulary) -> SplitData:
    d = cfg["data"]
    records, _ = load_records(resolve_path(d["processed_dir"]) / "metadata.json")
    feats, ids = load_features(resolve_path(d["features_dir"]), split)
    return build_split_data(split, records, ids, feats, vocab, int(d.get("max_caption_len", 32)))


def get_word_vectors(cfg: Mapping[str, Any], vocab: Vocabulary) -> tuple[np.ndarray, dict]:
    """Embedding matrix according to the config: GloVe, synthetic (smoke) or random (ablation)."""
    dim = int(cfg["model"].get("word_dim", 300))
    seed = int(cfg.get("seed", 42))
    if cfg["model"].get("pretrained_embeddings", True) is False:
        return random_word_vectors(vocab, dim, seed)
    glove = cfg["data"].get("glove_path")
    if glove is None:
        log.warning("data.glove_path is null -> using SYNTHETIC word vectors (smoke tests only, not GloVe)")
        return synthetic_word_vectors(vocab, dim, seed)
    return load_glove_matrix(resolve_path(glove), vocab, dim, seed)


def _write_metadata(run: Path, cfg: Mapping[str, Any], dataset_info: Mapping[str, Any], emb_stats: Mapping[str, Any],
                    training: Mapping[str, Any], ckpt_file: Path, vocab_size: int, extractor: str,
                    embedding_dim: Optional[int] = None) -> dict:
    m = cfg["model"]
    old = load_json(run / "model_metadata.json") if (run / "model_metadata.json").is_file() else {}
    meta = {
        "format": ARCH_VERSION, "name": cfg["name"],
        "architecture": {
            "type": m["type"], "embedding_dim": embedding_dim, "hidden_dim": m.get("hidden_dim"),
            "word_dim": m.get("word_dim"), "num_layers": m.get("num_layers"), "dropout": m.get("dropout"),
            "pretrained_embeddings": m.get("pretrained_embeddings", True),
            "freeze_embeddings": m.get("freeze_embeddings", False), "vocab_size": vocab_size,
            "image_feature_extractor": extractor,
        },
        "training_config": {"seed": cfg.get("seed"), "train": cfg.get("train"), "model": dict(m),
                            "max_caption_len": cfg["data"].get("max_caption_len")},
        "word_vectors": dict(emb_stats),
        "dataset": dict(dataset_info),
        "checkpoint": {"file": ckpt_file.name, "version": "ncir-ckpt-v1" if m["type"] != "baseline" else "ncir-baseline-v1",
                       "sha256": sha256_file(ckpt_file)},
        "training": dict(training),
        "environment": environment_info(),
        "evaluation": old.get("evaluation", {}),
        "status": "trained",
    }
    save_json(meta, run / "model_metadata.json")
    return meta


# ------------------------------------------------------------------------------------------ training
def train_model(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Train (or fit) the model described by ``cfg``; artifacts go to ``artifacts/<name>/``."""
    run = run_dir_for(cfg)
    run.mkdir(parents=True, exist_ok=True)
    log_path = run / "train.log.jsonl"
    get_logger("ncir.runner", log_file=log_path)
    set_seed(int(cfg.get("seed", 42)))

    processed = resolve_path(cfg["data"]["processed_dir"])
    vocab = Vocabulary.load(processed / "vocab.json")
    shutil.copyfile(processed / "vocab.json", run / "vocab.json")
    _, dataset_info = load_records(processed / "metadata.json")
    extractor = load_json(resolve_path(cfg["data"]["features_dir"]) / "extractor.json")["name"]
    train = load_split_data(cfg, "train", vocab)
    val = load_split_data(cfg, "val", vocab)
    word_vectors, emb_stats = get_word_vectors(cfg, vocab)
    log.info("word vectors: %s", emb_stats)

    kind = cfg["model"]["type"]
    t0 = time.time()
    if kind == "baseline":
        summary = _train_baseline(cfg, train, val, vocab, word_vectors, run)
        ckpt_file = run / "baseline.npz"
    elif kind in ("gru", "lstm"):
        if not torch_available():
            raise ImportError("PyTorch is not installed, but GRU/LSTM training needs it. "
                              "Install it with: pip install -r requirements.txt (see README for the CUDA build).")
        from src.training.trainer import train_neural
        # For the random-init ablation get_word_vectors() already returned a seeded random matrix.
        summary = train_neural(cfg, train, val, len(vocab), word_vectors, run)
        ckpt_file = run / "best.pt"
    else:
        raise ValueError(f"unknown model type '{kind}' (baseline | gru | lstm)")
    summary["wall_clock_seconds"] = round(time.time() - t0, 1)
    meta = _write_metadata(run, cfg, dataset_info, emb_stats, summary, ckpt_file, len(vocab), extractor,
                           embedding_dim=int(cfg["model"].get("embedding_dim") or train.features.shape[1]))
    log.info("saved model artifacts to %s", run)
    return meta


def _train_baseline(cfg: Mapping[str, Any], train: SplitData, val: SplitData, vocab: Vocabulary,
                    word_vectors: np.ndarray, run: Path) -> dict[str, Any]:
    t0 = time.time()
    x_train, _ = mean_word_vectors(train.captions, vocab, word_vectors)
    model = LinearRegressionBaseline().accumulate(x_train, train.features, train.caption_img_idx)
    grid = cfg["model"].get("l2", 1.0)
    grid = [float(v) for v in (grid if isinstance(grid, (list, tuple)) else [grid])]
    metric = str(cfg.get("train", {}).get("selection_metric", "t2i_rsum"))
    x_val, _ = mean_word_vectors(val.captions, vocab, word_vectors)
    img_val = _l2(val.features)
    results, best = {}, None
    for l2 in grid:  # ridge strength is chosen on VALIDATION only
        model.solve(l2)
        txt_val = _l2(model.predict(x_val))
        score = evaluate_embeddings(img_val, txt_val, val.caption_img_idx)
        results[str(l2)] = {k: float(v) for k, v in score.items()}
        if best is None or (score[metric] > best[0] if higher_is_better(metric) else score[metric] < best[0]):
            best = (score[metric], l2)
        log.info("baseline l2=%g | %s", l2, format_metrics(score))
    assert best is not None
    model.solve(best[1])
    model.save(run / "baseline.npz", word_vectors)
    save_json({"selection_metric": metric, "grid": results, "chosen_l2": best[1]}, run / "history.json")
    return {"selected_l2": best[1], "selection_metric": metric, "best_val_metric": {metric: float(best[0])},
            "l2_grid_validation": results, "train_seconds": round(time.time() - t0, 1), "device": "cpu (NumPy)"}


def _l2(x: np.ndarray) -> np.ndarray:
    from src.evaluation.metrics import l2_normalize

    return l2_normalize(x)


# ------------------------------------------------------------------------------------------ evaluation
def evaluate_run(cfg: Mapping[str, Any], split: str = "val", protocol: str = "full", out_dir: Optional[str | Path] = None,
                 device: str = "cpu") -> dict[str, Any]:
    """Evaluate the saved model of ``cfg`` on ``split`` and write ``reports/metrics/<name>_<split>_<protocol>.json``.

    Never used for model selection. Evaluate on ``test`` only once the configuration is final.
    """
    run = run_dir_for(cfg)
    retriever = load_retriever(run, device=device)
    vocab = Vocabulary.load(run / "vocab.json")
    data = load_split_data(cfg, split, vocab)
    t0 = time.perf_counter()
    img = retriever.encode_image_features(data.features)
    txt = retriever.encode_captions(data.captions)
    encode_seconds = time.perf_counter() - t0
    metrics = evaluate_protocol(img, txt, data.caption_img_idx, protocol)
    # single-query latency: encode one caption and score against all images (median of 50 runs)
    lat = []
    sample = data.captions[: min(50, len(data.captions))]
    for cap in sample:
        s = time.perf_counter()
        e = retriever.encode_captions([cap])
        _ = (img @ e[0]).argsort()[-10:]
        lat.append((time.perf_counter() - s) * 1000)
    meta = read_metadata(run)
    result = {
        "name": cfg["name"], "model_type": cfg["model"]["type"], "split": split, "protocol": protocol,
        "metrics": {k: v for k, v in metrics.items() if k != "protocol"},
        "num_images": data.num_images, "num_captions": data.num_captions,
        "embedding_dim": retriever.embedding_dim,
        "encode_seconds": round(encode_seconds, 3),
        "query_latency_ms_median": round(float(np.median(lat)), 3) if lat else None,
        "latency_setup": f"single caption query vs {data.num_images} images, brute-force NumPy, {meta.get('environment', {}).get('processor')}, "
                         f"device={device}, median of {len(lat)} queries",
        "dataset": meta.get("dataset", {}), "training_seconds": meta.get("training", {}).get("train_seconds"),
        "environment": environment_info(),
        "note": "t2i = caption -> image retrieval (what the web app does); i2t = image -> caption retrieval. "
                "Ties are ranked pessimistically.",
    }
    out = resolve_path(out_dir or "reports/metrics") / f"{cfg['name']}_{split}_{protocol}.json"
    save_json(result, out)
    meta["evaluation"][f"{split}_{protocol}"] = {"metrics": result["metrics"], "report_file": out.name,
                                                  "num_images": data.num_images}
    save_json(meta, run / "model_metadata.json")
    log.info("%s [%s/%s] %s", cfg["name"], split, protocol, format_metrics(metrics))
    return result


# ------------------------------------------------------------------------------------------ indexing
def build_run_index(cfg: Mapping[str, Any], split: Optional[str] = None, device: str = "cpu") -> ImageIndex:
    """Embed every image of ``split`` with the saved model and write ``artifacts/<name>/index``."""
    run = run_dir_for(cfg)
    split = split or cfg.get("index", {}).get("split", "test")
    retriever = load_retriever(run, device=device)
    feats, ids = load_features(resolve_path(cfg["data"]["features_dir"]), split)
    if feats.shape[1] != retriever.image_feature_dim:
        raise ValueError(f"features have {feats.shape[1]} dims but the model expects {retriever.image_feature_dim}")
    emb = retriever.encode_image_features(feats)
    index = ImageIndex(emb, ids, faiss_min_size=int(cfg.get("index", {}).get("faiss_min_size", 20000)), meta={
        "model": cfg["name"], "split": split, "embedding_dim": retriever.embedding_dim,
        "checkpoint_sha256": read_metadata(run).get("checkpoint", {}).get("sha256")})
    index.save(run / "index")
    log.info("indexed %d images (%s, dim=%d, backend=%s) -> %s", index.size, split, index.dim, index.backend, run / "index")
    return index
