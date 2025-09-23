from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List

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
    mask_dir = out / "masks"
    param_dir = out / "params"
    mask_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    jobs: List[Dict[str, object]] = []
    metadata_rows: List[Dict[str, object]] = []

    for i in range(num_samples):
        params = {
            "num_fingers": rng.randint(4, 16),
            "finger_length_um": rng.uniform(40, 120),
            "finger_width_um":  rng.uniform(2, 8),
            "finger_spacing_um":rng.uniform(2, 8),
        }
        mask = params_to_binary_mask(params, grid=grid)
        mask_path = mask_dir / f"mask_{i:04d}.npy"
        np.save(mask_path, mask)

        param_path = param_dir / f"params_{i:04d}.json"
        with param_path.open("w", encoding="utf-8") as pf:
            json.dump(params, pf, indent=2)

        sweep = {"f_start_GHz": 1.0, "f_stop_GHz": 20.0, "n_points": 402}

        jobs.append({
            "job_id": i,
            "mask_path": str(mask_path),
            "cap_type": "idc",
            "param_path": str(param_path),
            "param_seed": params,
            "sweep": sweep,
        })

        row = {
            "design_id": f"{i:04d}",
            "mask_path": str(mask_path.relative_to(out)),
            "param_path": str(param_path.relative_to(out)),
            "cap_type": "idc",
            "grid_x": grid[0],
            "grid_y": grid[1],
            "f_start_GHz": sweep["f_start_GHz"],
            "f_stop_GHz": sweep["f_stop_GHz"],
            "n_points": sweep["n_points"],
        }
        for key, value in params.items():
            row[f"param_{key}"] = value
        metadata_rows.append(row)

    with open(out/"jobs_to_run.json", "w", encoding="utf-8") as f:
        json.dump({"jobs": jobs}, f, indent=2)

    if metadata_rows:
        metadata_path = out / "metadata.csv"
        fieldnames = list(metadata_rows[0].keys())
        with metadata_path.open("w", newline="", encoding="utf-8") as mf:
            writer = csv.DictWriter(mf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)
    return str(out/"jobs_to_run.json")

if __name__ == "__main__":
    generate_design_candidates()
