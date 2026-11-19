"""Backend: framework-independent services (always run) and HTTP endpoints (need fastapi + httpx)."""
from __future__ import annotations

import json
import shutil

import pytest

from backend.config import Settings, load_dotenv
from backend.services.evaluation import PAPER_REFERENCE, collect_evaluation, collect_stats
from src.retrieval.search import SearchService


# ------------------------------------------------------------------ services (no web framework needed)
def test_dotenv_loader_does_not_override_existing_env(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nNCIR_MAX_TOP_K=7\nNCIR_LOG_LEVEL="DEBUG"\nBROKEN LINE\n\nNCIR_REPORTS_DIR=somewhere\n')
    import os

    monkeypatch.setenv("NCIR_REPORTS_DIR", "already-set")
    for key in ("NCIR_MAX_TOP_K", "NCIR_LOG_LEVEL"):
        monkeypatch.setenv(key, "placeholder")  # registers "absent" as the value to restore at teardown
        monkeypatch.delenv(key)
    try:
        loaded = load_dotenv(f)
        assert loaded == {"NCIR_MAX_TOP_K": "7", "NCIR_LOG_LEVEL": "DEBUG"}
        assert os.environ["NCIR_REPORTS_DIR"] == "already-set"  # existing variables win
    finally:
        os.environ.pop("NCIR_MAX_TOP_K", None)  # load_dotenv writes os.environ directly; never leak it
        os.environ.pop("NCIR_LOG_LEVEL", None)


def test_settings_defaults_are_relative_to_repo_not_cwd(monkeypatch, tmp_path):
    for k in ("NCIR_ARTIFACTS_DIR", "NCIR_METADATA_PATH", "NCIR_IMAGES_ROOT", "NCIR_REPORTS_DIR", "NCIR_STATS_PATH"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("backend.config.load_dotenv", lambda *a, **k: {})
    s = Settings.from_env()
    assert s.artifacts_dir.name == "artifacts" and s.artifacts_dir.is_absolute()
    assert s.stats_path == s.metadata_path.parent / "stats.json"
    assert "http://localhost:5173" in s.cors_origins and s.max_top_k == 100


def test_evaluation_is_unavailable_until_experiments_exist(tmp_path):
    res = collect_evaluation(tmp_path / "reports", tmp_path / "artifacts")
    assert res["available"] is False and "evaluate.py" in res["message"] and res["results"] == []
    assert res["reference"] == PAPER_REFERENCE  # reference values are kept apart from measured results


def test_evaluation_reads_real_files_and_flags_synthetic(smoke_ws, tmp_path):
    reports = tmp_path / "reports" / "metrics"
    reports.mkdir(parents=True)
    for p in (smoke_ws["workdir"] / "reports" / "metrics").glob("*_test_full.json"):
        shutil.copy(p, reports / p.name)
    (reports / "summary.json").write_text("{}")             # summary files must be ignored
    (reports / "broken_x_y.json").write_text("{not json")   # corrupt files must be skipped, not crash
    res = collect_evaluation(tmp_path / "reports", smoke_ws["artifacts"])
    assert res["available"] is True and len(res["results"]) == 1
    r = res["results"][0]
    assert r["name"] == "smoke_baseline" and r["split"] == "test" and "t2i_R@1" in r["metrics"]
    assert r["synthetic"] is True and res["synthetic_warning"] is True
    assert "smoke_baseline" in res["histories"] or res["histories"] == {}  # baseline history is a dict, not epochs


def test_stats_available_and_unavailable(smoke_ws, tmp_path):
    svc = SearchService.from_artifacts(smoke_ws["artifacts"], smoke_ws["processed"] / "metadata.json", smoke_ws["images_root"])
    ok = collect_stats(smoke_ws["processed"] / "stats.json", smoke_ws["processed"] / "metadata.json", svc.list_models())
    assert ok["available"] and ok["synthetic"] is True and ok["stats"]["splits"]["test"]["num_images"] == 30
    assert ok["indexed_images"] == {"smoke_baseline": 30}
    missing = collect_stats(tmp_path / "no.json", tmp_path / "no2.json", [])
    assert missing["available"] is False and "prepare_data.py" in missing["message"]
