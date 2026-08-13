import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.inference.model import SimpleCNN

CLASSES = ["cat", "dog"]

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_model(weights_path: str | None = None, device: str = "cpu") -> SimpleCNN:
    model = SimpleCNN(num_classes=len(CLASSES))
    if weights_path is not None and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _TRANSFORM(image)
    return tensor.unsqueeze(0)


def predict(model: SimpleCNN, tensor: torch.Tensor) -> tuple[str, dict[str, float]]:
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    prob_dict = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    label = CLASSES[int(torch.argmax(probs))]
    return label, prob_dict
