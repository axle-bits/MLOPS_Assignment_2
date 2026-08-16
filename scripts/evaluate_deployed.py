"""Post-deployment model performance tracking.

Sends a batch of held-out test images through the *deployed* API's /predict
endpoint (not the local model object), compares predictions against their true
labels, and writes both a per-image CSV and a metrics summary. This is the M5
"model performance tracking (post-deployment)" evidence.
"""

import argparse
import csv
import json
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
CLASSES = ("cat", "dog")


def gather_samples(test_dir: Path, per_class: int) -> list[tuple[Path, str]]:
    samples = []
    for label in CLASSES:
        class_dir = test_dir / label
        files = sorted(class_dir.glob("*.jpg"))[:per_class]
        samples.extend((f, label) for f in files)
    return samples


def compute_metrics(results: list[dict]) -> dict:
    """Accuracy, per-class precision/recall/F1, and a confusion matrix.

    `results` rows carry `true_label` and `pred_label`. Precision, recall and F1
    are 0.0 rather than an error when their denominator is zero (a class that
    was never predicted, or never appeared).
    """
    n = len(results)
    if n == 0:
        return {
            "accuracy": 0.0,
            "n_samples": 0,
            "per_class": {c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for c in CLASSES},
            "confusion_matrix": {t: {p: 0 for p in CLASSES} for t in CLASSES},
        }

    confusion = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for row in results:
        confusion[row["true_label"]][row["pred_label"]] += 1

    correct = sum(confusion[c][c] for c in CLASSES)

    per_class = {}
    for c in CLASSES:
        true_positives = confusion[c][c]
        predicted = sum(confusion[t][c] for t in CLASSES)
        actual = sum(confusion[c].values())

        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}

    return {
        "accuracy": correct / n,
        "n_samples": n,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def evaluate(test_dir: Path, per_class: int, out_path: Path, summary_path: Path) -> dict:
    samples = gather_samples(test_dir, per_class)
    results = []

    for path, true_label in samples:
        with open(path, "rb") as f:
            resp = requests.post(f"{BASE_URL}/predict", files={"file": (path.name, f, "image/jpeg")}, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        results.append(
            {
                "file": path.as_posix(),
                "true_label": true_label,
                "pred_label": body["label"],
                "prob_cat": round(body["probabilities"]["cat"], 4),
                "prob_dog": round(body["probabilities"]["dog"], 4),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "true_label", "pred_label", "prob_cat", "prob_dog"])
        writer.writeheader()
        writer.writerows(results)

    summary = compute_metrics(results)
    summary["endpoint"] = f"{BASE_URL}/predict"
    summary["per_image_results"] = out_path.as_posix()

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the deployed API against held-out test images")
    parser.add_argument("--test-dir", default="data/processed/test")
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--out", default="results/evaluation.csv")
    parser.add_argument("--summary", default="results/evaluation_summary.json")
    args = parser.parse_args()

    evaluate(Path(args.test_dir), args.per_class, Path(args.out), Path(args.summary))


if __name__ == "__main__":
    main()
