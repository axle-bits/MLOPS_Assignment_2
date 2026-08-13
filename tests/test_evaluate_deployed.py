from collections import Counter

from PIL import Image

from scripts.evaluate_deployed import gather_samples


def test_gather_samples_collects_per_class(tmp_path):
    for label in ("cat", "dog"):
        class_dir = tmp_path / label
        class_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (10, 10)).save(class_dir / f"{i}.jpg")

    samples = gather_samples(tmp_path, per_class=2)

    assert len(samples) == 4
    counts = Counter(label for _, label in samples)
    assert counts == {"cat": 2, "dog": 2}
