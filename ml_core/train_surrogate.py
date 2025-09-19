from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml_core.datasets import HybridMaskDataset
from ml_core.unet import HybridUNetSurrogate, UNetSmall

def passivity_penalty(y_pred: torch.Tensor, margin: float = 1.0):
    excess = torch.clamp(torch.abs(y_pred) - margin, min=0.0)
    return (excess**2).mean()

def train(
    data_dir: str = "data",
    out: str = "models/cnn_surrogate.pt",
    epochs: int = 30,
    batch: int = 32,
    lr: float = 2e-4,
    device: str = "cpu",
    use_hybrid: bool | None = None,
    param_dropout: float = 0.0,
) -> None:
    dataset = HybridMaskDataset(root=data_dir, return_dict=True)
    n = len(dataset)
    assert n > 10, "학습 표본이 너무 적습니다."

    if use_hybrid is None:
        use_hybrid = dataset.has_params
    if use_hybrid and not dataset.has_params:
        warnings.warn("Hybrid training requested but dataset has no parameter vectors; falling back to mask-only mode.")
        use_hybrid = False

    idx = int(n * 0.9)
    if idx >= n:
        idx = max(n - 1, 1)
    tr, va = torch.utils.data.random_split(dataset, [idx, n - idx])
    tl = DataLoader(tr, batch_size=batch, shuffle=True)
    vl = DataLoader(va, batch_size=batch, shuffle=False)

    out_dim = dataset.target_dim or 804
    in_ch = dataset.mask_channels or 1

    if use_hybrid:
        net = HybridUNetSurrogate(
            in_ch=in_ch,
            param_dim=dataset.param_dim,
            out_dim=out_dim,
        ).to(device)
    else:
        net = UNetSmall(in_ch=in_ch, img=64, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    best = 1e9
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        net.train(); loss_sum=0
        for batch in tl:
            if isinstance(batch, dict):
                x = batch["mask"].to(device)
                y = batch["target"].to(device)
                params = batch.get("params")
                if params is not None:
                    params = params.to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                params = None

            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])

            if use_hybrid:
                if params is None:
                    raise RuntimeError("Hybrid surrogate requires parameter tensors in the dataset")
                if param_dropout > 0:
                    dropout_mask = torch.rand_like(params) < param_dropout
                    params = params.masked_fill(dropout_mask, 0.0)
                yp = net(x, params)
            else:
                yp = net(x)

            loss = mse(yp, y) + 1e-3 * passivity_penalty(yp)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0)
        tr_loss = loss_sum/len(tr)

        net.eval(); vloss=0
        with torch.no_grad():
            for batch in vl:
                if isinstance(batch, dict):
                    x = batch["mask"].to(device)
                    y = batch["target"].to(device)
                    params = batch.get("params")
                    if params is not None:
                        params = params.to(device)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    params = None

                if use_hybrid:
                    if params is None:
                        raise RuntimeError("Hybrid surrogate requires parameter tensors in the dataset")
                    yp = net(x, params)
                else:
                    yp = net(x)
                vloss += mse(yp,y).item()*x.size(0)
        vloss /= len(va)
        print(f"[{ep+1:03d}] train {tr_loss:.4f}  val {vloss:.4f}")

        if vloss < best:
            best = vloss
            torch.save(net.state_dict(), out)
    print(f"[OK] saved {out} (best val {best:.4f})")

if __name__ == "__main__":
    train()
