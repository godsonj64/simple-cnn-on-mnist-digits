import torch

from src.model import build_model


def test_small_cnn_forward():
    cfg = {"name": "small_cnn", "num_classes": 10, "pretrained": False}
    model = build_model(cfg)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_efficientnet_forward():
    cfg = {"name": "efficientnet_b0", "num_classes": 10, "pretrained": False}
    model = build_model(cfg)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)
