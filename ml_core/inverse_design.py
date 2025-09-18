from pathlib import Path
import numpy as np
import torch
from ml_core.cnn_model import UNetSmall
import gdstk
from geometry.layout_manager import binary_mask_to_gds

def load_surrogate(weights, device):
    net = UNetSmall(in_ch=1, img=64, out_dim=804).to(device)
    net.load_state_dict(torch.load(weights, map_location=device))
    net.eval()
    return net

def inverse_design_adam(target_npy: str, weights: str, steps=500, lr=0.05, device="cpu"):
    tgt = torch.from_numpy(np.load(target_npy).astype(np.float32)).to(device)
    net = load_surrogate(weights, device)

    p = torch.randn(1,1,64,64, device=device)*0.01
    p.requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)

    for t in range(steps):
        m = torch.sigmoid(p)
        x = (m>0.5).float() if t%10==0 else m
        y = net(x)
        loss = torch.mean((y - tgt)**2)
        opt.zero_grad(); loss.backward(); opt.step()
        if (t+1)%50==0:
            print(f"[{t+1}] loss={loss.item():.5f}")

    m_final = (torch.sigmoid(p).detach()>0.5).float().cpu().numpy()[0,0]
    return m_final

def save_mask_and_gds(mask: np.ndarray, out_dir="results"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out/"best_mask.npy", mask.astype(np.uint8))
    lib = gdstk.Library()
    cell = binary_mask_to_gds(mask.astype(np.uint8), cell_name="IDC_INV")
    lib.add(cell); lib.write_gds(str(out/"best_idc.gds"))
    return str(out/"best_mask.npy"), str(out/"best_idc.gds")
