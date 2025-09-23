# surrogate.py
# Surrogate models for capacitor design.

from __future__ import annotations

import torch
from torch import nn


class MLPSurrogate(nn.Module):
    """A simple MLP surrogate mapping a feature vector to S-parameters."""

    def __init__(self, in_features: int = 7, out_features: int = 804) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SurrogateCNN(nn.Module):
    """
    A CNN model architecture that is compatible with the state_dict found in
    the pre-trained checkpoint file. The architecture and layer sizes
    are derived from the size mismatch error messages.
    """
    def __init__(self, in_ch=3, out_dim=804):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3, stride=1, padding=1), # 0
            nn.ReLU(inplace=True),                                                            # 1
            nn.MaxPool2d(kernel_size=2),                                                      # 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 3
            nn.ReLU(inplace=True),                                                            # 4
            nn.MaxPool2d(kernel_size=2),                                                      # 5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),# 6
            nn.ReLU(inplace=True),                                                            # 7
            nn.AdaptiveAvgPool2d((8, 8))  # To get 8x8 feature map, 128*8*8 = 8192 features
        )
        
        in_features = 8192
        self.regressor = nn.Sequential(
            nn.Flatten(),                                     # 0
            nn.Linear(in_features, 512),                      # 1
            nn.ReLU(inplace=True),                            # 2
            nn.Linear(512, out_dim),                          # 3
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
