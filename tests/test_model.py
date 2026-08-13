import torch

from src.inference.model import SimpleCNN


def test_forward_pass_output_shape():
    model = SimpleCNN(num_classes=2)
    x = torch.randn(4, 3, 224, 224)

    out = model(x)

    assert out.shape == (4, 2)
