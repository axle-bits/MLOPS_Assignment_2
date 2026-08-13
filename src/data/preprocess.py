import random
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


def stratified_split(
    items: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
