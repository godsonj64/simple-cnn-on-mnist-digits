import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """A lightweight CNN baseline trained from scratch."""

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_efficientnet_b0(num_classes, pretrained):
    """Build an EfficientNet-B0 with a fresh classification head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


def build_model(model_cfg):
    """Factory that returns the requested model based on config."""
    name = model_cfg["name"]
    num_classes = model_cfg["num_classes"]
    if name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes, model_cfg.get("pretrained", True))
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes, in_channels=3)
    raise ValueError(f"Unknown model name: {name}")
