"""Read-only access to evaluation outputs and dataset statistics (plain Python, no web framework).

Results are loaded from files written by ``scripts/evaluate.py`` / the trainer. Nothing here computes or
estimates a metric: missing files simply mean "not available / pending".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.common import load_json

# Historical numbers reported by the paper (as quoted in the project brief). These are a REFERENCE for
# discussion only: they come from a different implementation / setup and must not be mixed into the
# measured results table.
PAPER_REFERENCE = {
    "source": "Neural Caption-Image Retrieval (reference paper), values quoted from the project brief",
    "model": "GRU + GloVe",
    "t2i_R@1": 37.0,
    "t2i_R@10": 86.8,
    "note": "Historical result under the paper's own experimental setup; not reproduced by this project.",
}


def _safe_json(path: Path) -> Any | None:
    try:
        return load_json(path)
    except (OSError, ValueError):
        return None


def collect_evaluation(reports_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Gather every ``<name>_<split>_<protocol>.json`` result and each model's training history."""
    metrics_dir = Path(reports_dir) / "metrics"
    results: list[dict[str, Any]] = []
    synthetic = False
    if metrics_dir.is_dir():
        for p in sorted(metrics_dir.glob("*_*_*.json")):
            if p.name.startswith("summary"):
                continue
            data = _safe_json(p)
            if not isinstance(data, dict) or "metrics" not in data:
                continue
            ds = data.get("dataset", {}) or {}
            synthetic = synthetic or bool(ds.get("synthetic"))
            results.append({
                "name": data.get("name"), "model_type": data.get("model_type"), "split": data.get("split"),
                "protocol": data.get("protocol"), "metrics": data["metrics"], "num_images": data.get("num_images"),
                "num_captions": data.get("num_captions"), "embedding_dim": data.get("embedding_dim"),
                "latency_ms_median": data.get("query_latency_ms_median"), "latency_setup": data.get("latency_setup"),
                "training_seconds": data.get("training_seconds"), "dataset": ds.get("name"),
                "is_subset": bool(ds.get("is_subset")), "synthetic": bool(ds.get("synthetic")),
                "environment": data.get("environment", {}), "file": p.name,
            })
    histories: dict[str, Any] = {}
    art = Path(artifacts_dir)
    if art.is_dir():
        for run in sorted(p for p in art.iterdir() if p.is_dir()):
            h = _safe_json(run / "history.json")
            if isinstance(h, list) and h:
                histories[run.name] = [{k: e.get(k) for k in ("epoch", "train_loss", "val_loss", "t2i_R@1", "t2i_R@10", "t2i_rsum")}
                                       for e in h if isinstance(e, dict)]
    if not results:
        msg = ("No evaluation results found yet. Train a model, then run: "
               "python scripts/evaluate.py --config configs/<model>.yaml --split test --allow-test")
        return {"available": False, "message": msg, "results": [], "histories": histories,
                "reference": PAPER_REFERENCE, "synthetic_warning": False}
    return {"available": True, "message": None, "results": results, "histories": histories,
            "reference": PAPER_REFERENCE, "synthetic_warning": synthetic}


def collect_stats(stats_path: Path, metadata_path: Path, models: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset statistics (from ``stats.json``) plus the dataset description stored in ``metadata.json``."""
    stats = _safe_json(Path(stats_path))
    meta = _safe_json(Path(metadata_path))
    indexed = {m["name"]: int(m.get("num_images", 0)) for m in models}
    if not isinstance(stats, dict) or not isinstance(meta, dict):
        return {"available": False,
                "message": "Dataset statistics not found. Run scripts/prepare_data.py first.",
                "dataset": {}, "stats": {}, "indexed_images": indexed, "synthetic": False}
    dataset = meta.get("dataset", {}) or {}
    return {"available": True, "message": None, "dataset": dataset, "stats": stats,
            "indexed_images": indexed, "synthetic": bool(dataset.get("synthetic"))}
