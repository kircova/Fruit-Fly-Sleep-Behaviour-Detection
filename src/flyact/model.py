"""The classifier: a 2D ResNet applied to every snippet, with snippet logits averaged.

Motion enters through the input, not the architecture: each snippet is three frames
stacked as channels (see ``config.SamplingConfig``), so an ordinary image network sees
~130 ms of movement in every forward pass. A window of eight snippets then covers two
seconds, and the snippet logits are averaged.

Why not a 3D CNN? On the Apple-silicon GPU this project is developed on, a Kinetics
R(2+1)D-18 trains at ~5 clips/s and this model at ~80, which is the difference between
one experiment a day and a full grouped cross-validation in well under an hour.

The temporal shift module (Lin et al., ICCV 2019) is available as an option. It lets
snippets exchange information inside the backbone at zero parameters: in every residual
block an eighth of the channels is replaced by the same channels of the next snippet and
another eighth by those of the previous one. On this data it changed nothing measurable
(see the ablations in the README), so it is off by default.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck

from .config import ModelConfig


def temporal_shift(x: torch.Tensor, num_snippets: int, fold_div: int = 8) -> torch.Tensor:
    """Shift a fraction of the channels of ``(B*T, C, H, W)`` features along time."""
    bt, c, h, w = x.shape
    x = x.view(bt // num_snippets, num_snippets, c, h, w)
    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]  # these channels look one snippet ahead
    out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # these look one snippet back
    out[:, :, 2 * fold :] = x[:, :, 2 * fold :]
    return out.view(bt, c, h, w)


class TemporalShift(nn.Module):
    """Apply ``temporal_shift`` before a wrapped layer (the residual-branch TSM variant)."""

    def __init__(self, layer: nn.Module, num_snippets: int):
        super().__init__()
        self.layer = layer
        self.num_snippets = num_snippets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(temporal_shift(x, self.num_snippets))


class FlyNet(nn.Module):
    """Per-snippet 2D backbone with optional temporal shift and averaged snippet logits."""

    def __init__(self, cfg: ModelConfig, num_snippets: int, num_classes: int = 3):
        super().__init__()
        weights = "IMAGENET1K_V1" if cfg.pretrained else None
        net = getattr(models, cfg.backbone)(weights=weights)
        if cfg.temporal == "tsm":
            for module in net.modules():
                if isinstance(module, BasicBlock | Bottleneck):
                    module.conv1 = TemporalShift(module.conv1, num_snippets)
        elif cfg.temporal != "none":
            raise ValueError(f"unknown temporal mode '{cfg.temporal}'")
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(feature_dim, num_classes))
        self.num_snippets = num_snippets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, 3, H, W)`` -> ``(B, num_classes)`` logits."""
        b, t = x.shape[:2]
        if t != self.num_snippets:
            raise ValueError(f"expected {self.num_snippets} snippets per window, got {t}")
        features = self.backbone(x.flatten(0, 1))
        return self.head(features).view(b, t, -1).mean(dim=1)
