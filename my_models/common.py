"""my_models/common.py

一些拼接模型都会用到的小工具（融合、投影等）。
这些是“胶水层”，核心建模模块仍来自仓库现有 models/ 与 layers/。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScalarGate(nn.Module):
    """根据输入序列生成一个 [B,1,1] 的 gate，用于两路预测加权融合。

    gate = sigmoid(MLP(mean(x_enc)))
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc: [B, T, C]
        summary = x_enc.mean(dim=1)  # [B, C]
        g = self.net(summary)  # [B,1]
        return g[:, None, None]  # [B,1,1]


def maybe_project(pred: torch.Tensor, proj: nn.Module | None) -> torch.Tensor:
    return pred if proj is None else proj(pred)
