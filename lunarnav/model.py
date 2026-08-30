"""RGB-only ResNet18 geometry regressor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def build_rgb_resnet18(num_outputs: int = 2) -> nn.Module:
    """Single ImageNet-pretrained ResNet18 with a 2-output regression head."""
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    except AttributeError:
        model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_outputs)
    return model
