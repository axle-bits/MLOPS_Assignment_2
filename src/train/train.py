# src/train/train.py
import argparse
import tempfile
from pathlib import Path

# NOTE: torch must be imported before mlflow. On this environment, importing
# mlflow (and the native deps it pulls in) before torch causes torch's DLL
# initialization to fail with OSError: [WinError 1114] on Windows.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mlflow  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from src.inference.model import SimpleCNN  # noqa: E402

CLASS_NAMES = ["cat", "dog"]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float, list, list]:
    """Run `model` over `loader`; return (mean loss, accuracy, preds, labels)."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    preds_out, labels_out = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            seen += images.size(0)
            preds_out.extend(preds.cpu().tolist())
            labels_out.extend(labels.cpu().tolist())

    return total_loss / seen, correct / seen, preds_out, labels_out


def plot_confusion_matrix(labels: list, preds: list, title: str, out_path: Path) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(train_losses: list[float], val_losses: list[float], val_accs: list[float], out_path: Path) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))

    loss_ax.plot(epochs, train_losses, marker="o", label="train loss")
    loss_ax.plot(epochs, val_losses, marker="o", label="val loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Cross-entropy loss")
    loss_ax.set_title("Loss curves")
    loss_ax.legend()
    loss_ax.grid(alpha=0.3)

    acc_ax.plot(epochs, val_accs, marker="o", color="green", label="val accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_ylim(0, 1)
    acc_ax.set_title("Validation accuracy")
    acc_ax.legend()
    acc_ax.grid(alpha=0.3)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    model_out: Path,
    run_name: str | None = None,
) -> SimpleCNN:
    train_ds = datasets.ImageFolder(data_dir / "train", transform=TRAIN_TRANSFORM)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=EVAL_TRANSFORM)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SimpleCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("catsdogs-baseline-cnn")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "architecture": "SimpleCNN",
                "optimizer": "Adam",
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "test_samples": len(test_ds),
            }
        )

        train_losses, val_losses, val_accs = [], [], []
        val_preds, val_labels = [], []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss /= len(train_ds)

            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Final evaluation on the held-out test split, which training never saw.
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
        print(f"final: test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            plot_confusion_matrix(val_labels, val_preds, "Validation confusion matrix", tmp_dir / "confusion_matrix_val.png")
            plot_confusion_matrix(test_labels, test_preds, "Test confusion matrix", tmp_dir / "confusion_matrix_test.png")
            plot_loss_curves(train_losses, val_losses, val_accs, tmp_dir / "loss_curves.png")
            for name in ("confusion_matrix_val.png", "confusion_matrix_test.png", "loss_curves.png"):
                mlflow.log_artifact(str(tmp_dir / name))

        model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_out)
        mlflow.log_artifact(str(model_out))

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-out", default="models/model.pt")
    parser.add_argument("--run-name", default=None, help="Optional MLflow run name, for comparing experiments")
    args = parser.parse_args()

    train(
        Path(args.data_dir),
        args.epochs,
        args.batch_size,
        args.lr,
        args.device,
        Path(args.model_out),
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
