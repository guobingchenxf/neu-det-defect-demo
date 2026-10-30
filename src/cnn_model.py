"""小 CNN：输入 64×64 灰度 patch，输出"该 patch 含缺陷"的 logit。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(
        self, width: float = 1.0, in_ch: int = 1, dropout: float = 0.1, num_classes: int = 1
    ):
        super().__init__()
        c1, c2, c3 = int(32 * width), int(64 * width), int(128 * width)

        def block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(in_ch, c1), block(c1, c2), block(c2, c3))  # 64 -> 8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )
        self.width = width
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.features(x))  # (B, num_classes)
        return out.squeeze(-1) if self.num_classes == 1 else out

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(width: float = 1.0, dropout: float = 0.1, num_classes: int = 1) -> SmallCNN:
    return SmallCNN(width=width, dropout=dropout, num_classes=num_classes)


def load_model(path: str, map_location: Optional[str] = "cpu") -> SmallCNN:
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = SmallCNN(
        width=float(cfg.get("width", 1.0)),
        dropout=float(cfg.get("dropout", 0.1)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model
