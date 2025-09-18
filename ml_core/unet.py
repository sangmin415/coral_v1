# ml_core/unet.py
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class UNetSmall(nn.Module):
    """
    A small U-Net architecture for image-to-vector regression.
    This architecture is inferred from the training script and previous error messages.
    """
    def __init__(self, in_ch=1, out_dim=804, img=64):
        super().__init__()
        n = 32 # Initial number of filters
        self.down1 = Block(in_ch, n)
        self.down2 = Block(n, 2*n)
        self.bridge = Block(2*n, 4*n)
        self.up1 = nn.ConvTranspose2d(4*n, 2*n, kernel_size=2, stride=2)
        self.dec1 = Block(4*n, 2*n)
        self.up2 = nn.ConvTranspose2d(2*n, n, kernel_size=2, stride=2)
        self.dec2 = Block(2*n, n)
        self.pool = nn.MaxPool2d(2)

        # Regressor head to map the final feature map to the output vector
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        # Encoder path
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        b = self.bridge(p2)

        # Decoder path
        u1 = self.up1(b)
        c1 = torch.cat([u1, d2], dim=1)
        dec1 = self.dec1(c1)
        u2 = self.up2(dec1)
        c2 = torch.cat([u2, d1], dim=1)
        dec2 = self.dec2(c2)

        # Head
        out = self.head(dec2)
        return out
