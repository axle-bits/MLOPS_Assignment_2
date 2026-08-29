from collections import Counter

from PIL import Image

from scripts.evaluate_deployed import compute_metrics, gather_samples


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


def _row(true_label: str, pred_label: str) -> dict:
    return {"file": "x.jpg", "true_label": true_label, "pred_label": pred_label}


def test_compute_metrics_all_correct():
    results = [_row("cat", "cat"), _row("dog", "dog")]

    metrics = compute_metrics(results)

    assert metrics["accuracy"] == 1.0
    assert metrics["n_samples"] == 2
    assert metrics["per_class"]["cat"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
    assert metrics["confusion_matrix"] == {"cat": {"cat": 1, "dog": 0}, "dog": {"cat": 0, "dog": 1}}


def test_compute_metrics_with_known_errors():
    # 3 cats (2 correct, 1 called dog), 3 dogs (1 correct, 2 called cat)
    results = [
        _row("cat", "cat"),
        _row("cat", "cat"),
        _row("cat", "dog"),
        _row("dog", "dog"),
        _row("dog", "cat"),
        _row("dog", "cat"),
    ]

    metrics = compute_metrics(results)

    assert metrics["accuracy"] == 0.5
    # 4 predictions of "cat", 2 of which were really cats
    assert metrics["per_class"]["cat"]["precision"] == 0.5
    # 3 real cats, 2 recovered
    assert round(metrics["per_class"]["cat"]["recall"], 4) == 0.6667
    assert metrics["per_class"]["cat"]["support"] == 3
    assert metrics["per_class"]["dog"]["precision"] == 0.5
    assert round(metrics["per_class"]["dog"]["recall"], 4) == 0.3333
    assert metrics["confusion_matrix"]["dog"] == {"cat": 2, "dog": 1}


def test_compute_metrics_handles_class_never_predicted():
    results = [_row("cat", "cat"), _row("dog", "cat")]

    metrics = compute_metrics(results)

    # No "dog" predictions at all - precision is undefined, reported as 0.0
    # rather than raising ZeroDivisionError.
    assert metrics["per_class"]["dog"]["precision"] == 0.0
    assert metrics["per_class"]["dog"]["recall"] == 0.0
    assert metrics["per_class"]["dog"]["f1"] == 0.0


def test_compute_metrics_on_empty_results():
    metrics = compute_metrics([])

    assert metrics["accuracy"] == 0.0
    assert metrics["n_samples"] == 0
