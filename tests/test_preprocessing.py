import numpy as np
from PIL import Image

from src.data.preprocess import is_valid_image, load_and_resize


def test_load_and_resize_returns_224x224_rgb(tmp_path):
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (500, 300), color=(10, 20, 30)).save(img_path)

    result = load_and_resize(img_path, size=(224, 224))

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_load_and_resize_converts_grayscale_to_rgb(tmp_path):
    img_path = tmp_path / "gray.jpg"
    Image.new("L", (100, 100), color=128).save(img_path)

    result = load_and_resize(img_path, size=(224, 224))

    assert result.shape == (224, 224, 3)


def test_is_valid_image_true_for_real_image(tmp_path):
    img_path = tmp_path / "ok.jpg"
    Image.new("RGB", (50, 50)).save(img_path)

    assert is_valid_image(img_path) is True


def test_is_valid_image_false_for_corrupt_file(tmp_path):
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"not an image")

    assert is_valid_image(bad_path) is False
