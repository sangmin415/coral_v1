from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ml_core.unet import UNetSmall

class MaskSparamSet(Dataset):
    def __init__(self, data_dir="data", img=64):
        self.mdir = Path(data_dir)/"masks"
        self.rdir = Path(data_dir)/"simulation_results"
        self.items = []
        for m in sorted(self.mdir.glob("mask_*.npy")):
            rid = m.stem.split("_")[-1]
            y = self.rdir/f"result_{rid}.npy"
            if y.exists(): self.items.append((m,y))
        self.img = img
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        mpath, ypath = self.items[idx]
        x = np.load(mpath).astype(np.float32)[None, ...]
        y = np.load(ypath).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

def passivity_penalty(y_pred: torch.Tensor, margin: float = 1.0):
    excess = torch.clamp(torch.abs(y_pred) - margin, min=0.0)
    return (excess**2).mean()

def train(data_dir="data", out="models/cnn_surrogate.pt", epochs=30, batch=32, lr=2e-4, device="cpu"):
    ds = MaskSparamSet(data_dir)
    n = len(ds)
    assert n > 10, "학습 표본이 너무 적습니다."
    idx = int(n*0.9)
    tr, va = torch.utils.data.random_split(ds, [idx, n-idx])
    tl = DataLoader(tr, batch_size=batch, shuffle=True)
    vl = DataLoader(va, batch_size=batch, shuffle=False)

    net = UNetSmall(in_ch=1, img=64, out_dim=804).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    best = 1e9
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        net.train(); loss_sum=0
        for x,y in tl:
            x,y = x.to(device), y.to(device)
            if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[-1])
            yp = net(x)
            loss = mse(yp, y) + 1e-3*passivity_penalty(yp)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0)
        tr_loss = loss_sum/len(tr)

        net.eval(); vloss=0
        with torch.no_grad():
            for x,y in vl:
                x,y = x.to(device), y.to(device)
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
