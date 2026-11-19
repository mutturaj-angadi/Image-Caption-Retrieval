"""Report/asset scripts and config helpers: results must come only from real, non-synthetic files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.make_report_assets import load_results, markdown_table  # noqa: E402
from src.utils.common import deep_update, load_config, parse_overrides  # noqa: E402


def _result(name: str, synthetic: bool) -> dict:
    return {"name": name, "model_type": "gru", "split": "test", "protocol": "full", "num_images": 1000, "num_captions": 5000,
            "metrics": {"t2i_R@1": 10.0, "t2i_R@5": 30.0, "t2i_R@10": 45.0, "t2i_mean_rank": 20.0, "i2t_R@1": 12.0,
                        "i2t_R@5": 33.0, "i2t_R@10": 48.0, "i2t_mean_rank": 15.0},
            "query_latency_ms_median": 1.5, "training_seconds": 99, "dataset": {"name": "d", "synthetic": synthetic, "is_subset": True}}


def test_report_assets_ignore_synthetic_and_mark_missing_as_pending(tmp_path):
    metrics = tmp_path / "reports" / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "gru_glove_test_full.json").write_text(json.dumps(_result("gru_glove", synthetic=False)))
    (metrics / "lstm_glove_test_full.json").write_text(json.dumps(_result("lstm_glove", synthetic=True)))
    res = load_results(tmp_path / "reports", tmp_path / "artifacts", "test", "full")
    assert res["gru_glove"]["status"] == "measured" and res["gru_glove"]["subset"] is True
    assert res["lstm_glove"]["status"] == "pending" and "SYNTHETIC" in res["lstm_glove"]["note"]
    assert res["baseline_linreg"]["status"] == "pending"
    table = markdown_table(res)
    assert "10.0" in table and table.count("pending") >= 3 * 8  # three models x eight columns are pending


def test_all_pending_when_nothing_was_run(tmp_path):
    res = load_results(tmp_path / "r", tmp_path / "a", "test", "full")
    assert {e["status"] for e in res.values()} == {"pending"}


def test_parse_overrides_and_deep_update():
    ov = parse_overrides(["train.epochs=3", "data.glove_path=null", "model.l2=[0.1,1.0]", "name=abc"])
    assert ov == {"train": {"epochs": 3}, "data": {"glove_path": None}, "model": {"l2": [0.1, 1.0]}, "name": "abc"}
    merged = deep_update({"train": {"epochs": 30, "lr": 0.1}}, ov)
    assert merged["train"] == {"epochs": 3, "lr": 0.1}
    with pytest.raises(ValueError):
        parse_overrides(["no-equals-sign"])


@pytest.mark.parametrize("cfg", ["baseline", "gru", "lstm", "gru_no_glove"])
def test_shipped_configs_load_and_are_consistent(cfg):
    c = load_config(f"configs/{cfg}.yaml")
    assert c["model"]["type"] in ("baseline", "gru", "lstm") and c["data"]["image_feature_dim"] == 4096
    if c["model"]["type"] != "baseline":
        assert c["model"]["embedding_dim"] == 1024 and c["train"]["margin"] > 0
        assert c["train"]["selection_metric"] == "t2i_rsum"  # model selection is on validation, never test
    assert c["name"] == {"baseline": "baseline_linreg", "gru": "gru_glove", "lstm": "lstm_glove", "gru_no_glove": "gru_random_init"}[cfg]


def test_missing_config_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
