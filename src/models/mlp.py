"""PyTorch implementation of a configurable MLP surrogate model."""
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """Configurable fully-connected network for regression tasks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (128, 128, 64),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        act = _make_activation(activation)
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(act.__class__())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)
