import io

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app

client = TestClient(app)


def _dummy_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_valid_image_returns_label():
    files = {"file": ("pet.jpg", _dummy_image_bytes(), "image/jpeg")}

    response = client.post("/predict", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("cat", "dog")
    assert "probabilities" in body


def test_predict_with_non_image_returns_400():
    files = {"file": ("note.txt", b"hello world", "text/plain")}

    response = client.post("/predict", files=files)

    assert response.status_code == 400
