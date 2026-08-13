import io

from PIL import Image

from src.inference.model import SimpleCNN
from src.inference.predict import CLASSES, load_model, predict, preprocess_image


def _dummy_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_load_model_without_weights_returns_eval_mode_model():
    model = load_model(weights_path=None)

    assert isinstance(model, SimpleCNN)
    assert model.training is False


def test_preprocess_image_returns_correct_tensor_shape():
    tensor = preprocess_image(_dummy_image_bytes())

    assert tensor.shape == (1, 3, 224, 224)


def test_predict_returns_valid_label_and_probabilities():
    model = load_model(weights_path=None)
    tensor = preprocess_image(_dummy_image_bytes())

    label, probs = predict(model, tensor)

    assert label in CLASSES
    assert set(probs.keys()) == set(CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-4
