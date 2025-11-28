import pytest
from fastapi.testclient import TestClient

from morgan.jina.reranking.api import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_rerank_fallback(client):
    payload = {
        "query": "docker deploy",
        "results": [
            {"content": "A short snippet", "score": 0.2, "metadata": {}, "source": "a"},
            {"content": "A much longer snippet about docker and deployment practices", "score": 0.1, "metadata": {}, "source": "b"},
        ],
        "top_k": 1,
    }
    resp = client.post("/rerank", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"]
    # Fallback should prefer longer content
    assert data["results"][0]["source"] == "b"
