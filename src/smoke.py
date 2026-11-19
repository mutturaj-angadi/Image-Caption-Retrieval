"""End-to-end smoke pipeline on SYNTHETIC data (no COCO, no GloVe download, no GPU needed).

It exercises the same code paths as the real pipeline: prepare -> features -> train -> evaluate -> index
-> search. The numbers it produces say nothing about real-world quality and must never be reported
as results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.data.features import extract_split_features, get_extractor
from src.data.prepare import prepare_dataset
from src.data.synthetic import make_synthetic_dataset
from src.retrieval.search import SearchService
from src.training.runner import build_run_index, evaluate_run, train_model
from src.utils.common import deep_update, get_logger, load_config, resolve_path, torch_available

log = get_logger("ncir.smoke")


def smoke_config(workdir: Path, base_config: str, name: str, overrides: Optional[dict] = None) -> dict[str, Any]:
    """Load a real config and redirect it to the smoke workspace with tiny dimensions."""
    cfg = load_config(base_config)
    small = {
        "name": name,
        "artifacts_dir": (workdir / "artifacts").as_posix(),
        "data": {"processed_dir": (workdir / "processed").as_posix(),
                 "features_dir": (workdir / "processed" / "features").as_posix(),
                 "glove_path": None, "max_caption_len": 16, "image_feature_dim": 256},
        "index": {"split": "test", "faiss_min_size": 10 ** 9},
    }
    if cfg["model"]["type"] == "baseline":
        small["model"] = {"l2": [0.01, 0.1, 1.0], "word_dim": 32}
    else:
        small["model"] = {"embedding_dim": 48, "hidden_dim": 48, "word_dim": 32}
        small["train"] = {"batch_size": 32, "epochs": 12, "lr": 0.003, "patience": 6, "val_subset_images": 0,
                          "device": "cpu"}
    return deep_update(deep_update(cfg, small), overrides or {})


def build_smoke_workspace(workdir: str | Path, with_neural: Optional[bool] = None, n_train: int = 150,
                          n_val: int = 30, n_test: int = 30) -> dict[str, Any]:
    """Generate synthetic data and run the whole pipeline. Returns paths and per-step results.

    The baseline always runs (pure NumPy). GRU / LSTM run only if PyTorch is installed (or
    ``with_neural=True``, which then raises if it is missing).
    """
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    records, images_root = make_synthetic_dataset(work / "synthetic", n_train, n_val, n_test)
    processed = work / "processed"
    prep = prepare_dataset(records, images_root, processed, dataset_name="synthetic-shapes (SMOKE TEST ONLY)",
                           split_source="generated", synthetic=True, min_freq=1, seed=7)
    extractor = get_extractor("simple_rgb256")
    extract_split_features(records, images_root, processed / "features", extractor)

    configs = ["configs/baseline.yaml"]
    run_neural = torch_available() if with_neural is None else with_neural
    if run_neural:
        configs += ["configs/gru.yaml", "configs/lstm.yaml"]
    out: dict[str, Any] = {"workdir": work, "processed": processed, "artifacts": work / "artifacts",
                           "images_root": images_root, "prepare": prep, "models": {}, "neural_skipped": not run_neural}
    for base in configs:
        name = "smoke_" + Path(base).stem
        cfg = smoke_config(work, base, name)
        log.info("== %s ==", name)
        meta = train_model(cfg)
        val_res = evaluate_run(cfg, "val", "full", out_dir=work / "reports" / "metrics")
        test_res = evaluate_run(cfg, "test", "full", out_dir=work / "reports" / "metrics")
        build_run_index(cfg, "test")
        out["models"][name] = {"config": cfg, "metadata": meta, "val": val_res, "test": test_res}
    return out


def smoke_service(info: dict[str, Any]) -> SearchService:
    """Build a SearchService over the smoke workspace artifacts."""
    return SearchService.from_artifacts(info["artifacts"], info["processed"] / "metadata.json",
                                        info["images_root"], faiss_min_size=10 ** 9)
