import argparse
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


def split_items(
    items: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list]:
    """Shuffle `items` and split them by ratio into train/val/test.

    Called once per class by `preprocess_dataset`, which is what makes the
    overall split stratified — each class is divided 80/10/10 independently,
    so every split keeps the source class balance.
    """
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


_CLASS_MAP = {"Cat": "cat", "Dog": "dog"}


def preprocess_dataset(raw_dir: Path, out_dir: Path, size: tuple[int, int] = (224, 224), seed: int = 42) -> None:
    for raw_class, out_class in _CLASS_MAP.items():
        class_dir = raw_dir / raw_class
        valid_files = [p for p in sorted(class_dir.glob("*.jpg")) if is_valid_image(p)]
        if not valid_files:
            raise ValueError(f"no valid images found in {class_dir}")
        splits = split_items(valid_files, seed=seed)

        for split_name, files in splits.items():
            split_dir = out_dir / split_name / out_class
            split_dir.mkdir(parents=True, exist_ok=True)
            for src_path in files:
                # is_valid_image (Task 2) only checks structural integrity via
                # Image.verify(), not a full decode — some files pass that check
                # but still fail to decode here (confirmed during Task 2 review).
                # Skip and log rather than let one bad file abort the whole run.
                try:
                    arr = load_and_resize(src_path, size=size)
                except (OSError, ValueError) as exc:
                    print(f"skipping unreadable file {src_path}: {exc}")
                    continue
                Image.fromarray(arr).save(split_dir / src_path.name, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw PetImages into 224x224 RGB train/val/test splits")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocess_dataset(Path(args.raw_dir), Path(args.out_dir), size=(args.size, args.size), seed=args.seed)


if __name__ == "__main__":
    main()
