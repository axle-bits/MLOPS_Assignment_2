from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_metrics_endpoint_exposes_request_count():
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert b"request_count" in response.content
