# ml_core/unet.py
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBackbone(nn.Module):
    """Two-level U-Net encoder/decoder that returns feature maps."""

    def __init__(self, in_ch: int = 1, base_channels: int = 32):
        super().__init__()
        self.down1 = Block(in_ch, base_channels)
        self.down2 = Block(base_channels, 2 * base_channels)
        self.bridge = Block(2 * base_channels, 4 * base_channels)
        self.up1 = nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=2, stride=2)
        self.dec1 = Block(4 * base_channels, 2 * base_channels)
        self.up2 = nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=2, stride=2)
        self.dec2 = Block(2 * base_channels, base_channels)
        self.pool = nn.MaxPool2d(2)
        self._out_channels = base_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        b = self.bridge(p2)

        u1 = self.up1(b)
        c1 = torch.cat([u1, d2], dim=1)
        dec1 = self.dec1(c1)
        u2 = self.up2(dec1)
        c2 = torch.cat([u2, d1], dim=1)
        dec2 = self.dec2(c2)
        return dec2


class UNetSmall(nn.Module):
    """A small U-Net architecture for image-to-vector regression."""

    expects_params = False

    def __init__(self, in_ch: int = 1, out_dim: int = 804, img: int = 64, base_channels: int = 32):
        super().__init__()
        del img  # maintained for backwards compatibility, not used directly.
        self.backbone = UNetBackbone(in_ch=in_ch, base_channels=base_channels)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)


class ParamEncoder(nn.Module):
    """Encode parametric seed vectors into a latent representation."""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 64,
        hidden_dims: Sequence[int] | None = (128,),
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_dim <= 0:
            raise ValueError("ParamEncoder requires a positive in_dim")

        layers: list[nn.Module] = []
        prev = in_dim
        if hidden_dims:
            for hidden in hidden_dims:
                layers.append(nn.Linear(prev, hidden))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = hidden

        layers.append(nn.Linear(prev, latent_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.net(params)


class HybridUNetSurrogate(nn.Module):
    """Fuse mask features with parametric encodings for S-parameter regression."""

    expects_params = True

    def __init__(
        self,
        in_ch: int = 1,
        param_dim: int = 4,
        out_dim: int = 804,
        img: int = 64,
        base_channels: int = 32,
        param_hidden_dims: Sequence[int] | None = (128,),
        param_latent_dim: int | None = None,
        fusion_hidden: Sequence[int] | None = (128, 64),
        param_dropout: float = 0.0,
    ):
        super().__init__()
        del img  # placeholder to keep signature similar to UNetSmall
        if param_dim <= 0:
            raise ValueError("HybridUNetSurrogate requires param_dim > 0")

        self.backbone = UNetBackbone(in_ch=in_ch, base_channels=base_channels)
        latent_dim = param_latent_dim or self.backbone.out_channels
        self.param_encoder = ParamEncoder(
            in_dim=param_dim,
            latent_dim=latent_dim,
            hidden_dims=param_hidden_dims,
            dropout=param_dropout,
        )
        self.mask_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        fusion_layers: list[nn.Module] = []
        fusion_in = self.backbone.out_channels + self.param_encoder.latent_dim
        prev = fusion_in
        if fusion_hidden:
            for hidden in fusion_hidden:
                fusion_layers.append(nn.Linear(prev, hidden))
                fusion_layers.append(nn.ReLU())
                prev = hidden
        fusion_layers.append(nn.Linear(prev, out_dim))
        self.regressor = nn.Sequential(*fusion_layers)

    def forward(self, mask: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if params is None:
            raise ValueError("HybridUNetSurrogate expects a param tensor")
        mask_latent = self.mask_pool(self.backbone(mask))
        param_latent = self.param_encoder(params)
        fused = torch.cat([mask_latent, param_latent], dim=-1)
        return self.regressor(fused)
