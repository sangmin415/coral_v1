from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
from geometry.layout_manager import params_to_binary_mask

def generate_design_candidates(
    out_dir: str = "data",
    num_samples: int = 100,
    grid=(64,64),
    seed: int = 42,
):
    rng = random.Random(seed)
    out = Path(out_dir)
    (out/"masks").mkdir(parents=True, exist_ok=True)
    jobs = []

    for i in range(num_samples):
        params = {
            "num_fingers": rng.randint(4, 16),
            "finger_length_um": rng.uniform(40, 120),
            "finger_width_um":  rng.uniform(2, 8),
            "finger_spacing_um":rng.uniform(2, 8),
        }
        mask = params_to_binary_mask(params, grid=grid)
        mask_path = out/"masks"/f"mask_{i:04d}.npy"
        np.save(mask_path, mask)

        jobs.append({
            "job_id": i,
            "mask_path": str(mask_path),
            "cap_type": "idc",
            "sweep": {"f_start_GHz": 1.0, "f_stop_GHz": 20.0, "n_points": 402}
        })

    with open(out/"jobs_to_run.json", "w", encoding="utf-8") as f:
        json.dump({"jobs": jobs}, f, indent=2)
    return str(out/"jobs_to_run.json")

if __name__ == "__main__":
    generate_design_candidates()
