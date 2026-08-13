from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def load_and_resize(path: Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(size)
        return np.array(img, dtype=np.uint8)
