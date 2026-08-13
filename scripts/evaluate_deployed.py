import argparse
import csv
import json
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"


def gather_samples(test_dir: Path, per_class: int) -> list[tuple[Path, str]]:
    samples = []
    for label in ("cat", "dog"):
        class_dir = test_dir / label
        files = sorted(class_dir.glob("*.jpg"))[:per_class]
        samples.extend((f, label) for f in files)
    return samples


def evaluate(test_dir: Path, per_class: int, out_path: Path) -> dict:
    samples = gather_samples(test_dir, per_class)
    results = []
    correct = 0

    for path, true_label in samples:
        with open(path, "rb") as f:
            resp = requests.post(f"{BASE_URL}/predict", files={"file": (path.name, f, "image/jpeg")}, timeout=10)
        resp.raise_for_status()
        pred_label = resp.json()["label"]
        correct += int(pred_label == true_label)
        results.append({"file": str(path), "true_label": true_label, "pred_label": pred_label})

    accuracy = correct / len(results) if results else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "true_label", "pred_label"])
        writer.writeheader()
        writer.writerows(results)

    summary = {"accuracy": accuracy, "n_samples": len(results)}
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="data/processed/test")
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--out", default="results/evaluation.csv")
    args = parser.parse_args()

    evaluate(Path(args.test_dir), args.per_class, Path(args.out))


if __name__ == "__main__":
    main()
