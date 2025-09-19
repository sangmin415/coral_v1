# Data Schema Overview

This repository now organizes capacitor design datasets into a hybrid structure
that keeps both pixel masks and the originating parametric seeds.

```
data/
├── masks/              # Binary layout masks saved as mask_XXXX.npy
├── params/             # Parametric seeds saved as params_XXXX.json
├── simulation_results/ # HFSS response vectors saved as result_XXXX.npy
├── jobs_to_run.json    # Batch HFSS job descriptions
└── metadata.csv        # Tabular index with paths, sweep config, and parameters
```

## metadata.csv columns

| Column            | Description |
|-------------------|-------------|
| `design_id`       | Four-digit identifier shared across files. |
| `mask_path`       | Relative path to the binary mask `.npy` file. |
| `param_path`      | Relative path to the param seed `.json` file. |
| `cap_type`        | Capacitor category (currently always `idc`). |
| `grid_x`, `grid_y`| Pixel grid dimensions used to rasterize the mask. |
| `f_start_GHz`     | HFSS sweep start frequency. |
| `f_stop_GHz`      | HFSS sweep stop frequency. |
| `n_points`        | Number of frequency samples in the sweep. |
| `param_*`         | Individual param seed fields (length, spacing, etc.). |

## File naming conventions

All related files share the same four-digit suffix (e.g., `0007`). The
`HybridMaskDataset` loader uses this suffix to join mask, parameter, and
simulation results.
