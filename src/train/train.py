# src/train/train.py
import argparse
from pathlib import Path

# NOTE: torch must be imported before mlflow. On this environment, importing
# mlflow (and the native deps it pulls in) before torch causes torch's DLL
# initialization to fail with OSError: [WinError 1114] on Windows.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import confusion_matrix

from src.inference.model import SimpleCNN

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


def train(data_dir: Path, epochs: int, batch_size: int, lr: float, device: str, model_out: Path) -> SimpleCNN:
    train_ds = datasets.ImageFolder(data_dir / "train", transform=TRAIN_TRANSFORM)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SimpleCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("catsdogs-baseline-cnn")
    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr, "architecture": "SimpleCNN"})

        all_preds, all_labels = [], []
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

            model.eval()
            val_loss, correct = 0.0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            val_loss /= len(val_ds)
            val_acc = correct / len(val_ds)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1], labels=["cat", "dog"])
        ax.set_yticks([0, 1], labels=["cat", "dog"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

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
    args = parser.parse_args()

    train(Path(args.data_dir), args.epochs, args.batch_size, args.lr, args.device, Path(args.model_out))


if __name__ == "__main__":
    main()
