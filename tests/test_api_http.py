"""HTTP endpoint tests. Skipped automatically when fastapi / httpx are not installed
(``pip install -r requirements-dev.txt``). Framework-independent backend tests live in test_api.py."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi is not installed; HTTP endpoint tests skipped")
pytest.importorskip("httpx", reason="httpx is not installed; HTTP endpoint tests skipped")
from fastapi.testclient import TestClient  # noqa: E402

from backend.config import Settings  # noqa: E402
from backend.main import create_app  # noqa: E402
from src.retrieval.search import SearchService  # noqa: E402


@pytest.fixture()
def client(smoke_ws, tmp_path):
    svc = SearchService.from_artifacts(smoke_ws["artifacts"], smoke_ws["processed"] / "metadata.json", smoke_ws["images_root"])
    settings = Settings(artifacts_dir=smoke_ws["artifacts"], metadata_path=smoke_ws["processed"] / "metadata.json",
                        images_root=smoke_ws["images_root"], reports_dir=tmp_path / "reports",
                        stats_path=smoke_ws["processed"] / "stats.json", cors_origins=["http://localhost:5173"])
    with TestClient(create_app(service=svc, settings=settings)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["models_available"] == 1 and body["default_model"] == "smoke_baseline"


def test_models_and_stats_and_evaluation_endpoints(client):
    models = client.get("/api/models").json()
    assert models["models"][0]["available"] is True and models["models"][0]["type"] == "baseline"
    assert client.get("/api/stats").json()["stats"]["total_images"] == 210
    ev = client.get("/api/evaluation").json()
    assert ev["available"] is False and ev["reference"]["t2i_R@1"] == 37.0


def test_search_returns_ranked_results_and_images_are_servable(client):
    r = client.post("/api/search", json={"query": "a red circle on a white background", "top_k": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["num_results"] == 4 and body["score_type"] == "cosine_similarity"
    assert [x["rank"] for x in body["results"]] == [1, 2, 3, 4]
    img = client.get(body["results"][0]["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png" and img.content[:4] == b"\x89PNG"


@pytest.mark.parametrize("payload,status,code", [
    ({"query": "", "top_k": 5}, 422, "invalid_query"),
    ({"query": "   ", "top_k": 5}, 422, "invalid_query"),
    ({"query": "a red circle", "top_k": 0}, 422, "invalid_query"),
    ({"query": "a red circle", "top_k": 5000}, 422, "invalid_query"),
    ({"query": "a red circle", "top_k": "5"}, 422, "invalid_request"),
    ({"query": "a red circle", "top_k": True}, 422, "invalid_request"),
    ({"query": "a red circle", "model": "nope"}, 404, "model_not_found"),
    ({"top_k": 5}, 422, "invalid_request"),
    ({"query": "a red circle", "top_k": "many"}, 422, "invalid_request"),
])
def test_invalid_search_requests_return_consistent_errors(client, payload, status, code):
    r = client.post("/api/search", json=payload)
    assert r.status_code == status and r.json()["error"]["code"] == code and r.json()["error"]["message"]


def test_image_endpoint_rejects_unknown_ids_and_path_tricks(client):
    for bad in ("does-not-exist", "..%2F..%2Fetc%2Fpasswd", "syn_99999"):
        r = client.get(f"/api/images/{bad}")
        assert r.status_code == 404 and r.json()["error"]["code"] in ("image_not_found", "http_error")


def test_unknown_route_and_wrong_method_use_the_error_shape(client):
    assert client.get("/nope").json()["error"]["code"] == "http_error"
    assert client.get("/api/search").status_code == 405


def test_cors_allows_the_dev_frontend_only(client):
    ok = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    other = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_api_starts_degraded_without_artifacts_and_search_returns_503(tmp_path):
    svc = SearchService.from_artifacts(tmp_path / "none", tmp_path / "none.json")
    settings = Settings(tmp_path / "none", tmp_path / "none.json", None, tmp_path, tmp_path / "s.json")
    with TestClient(create_app(service=svc, settings=settings)) as c:
        health = c.get("/health")
        assert health.status_code == 503 and health.json()["status"] == "degraded"
        r = c.post("/api/search", json={"query": "a dog", "top_k": 3})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
        assert c.get("/api/stats").json()["available"] is False
