"""Dataset utilities for hybrid parametric + pixel training."""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class HybridMaskDataset(Dataset):
    """Load mask, parametric seed, and target S-parameter tensors.

    Parameters
    ----------
    root:
        Base directory that contains `masks/`, `params/`, and `simulation_results/`.
    mask_subdir / param_subdir / target_subdir:
        Sub-directory names for mask, parameter, and target data respectively.
    metadata_path:
        Optional path to a CSV metadata table. When omitted the loader looks for
        `<root>/metadata.csv`.
    include_metadata:
        Whether to return metadata dictionaries alongside tensors.
    return_dict:
        If ``True`` (default) returns a dictionary per sample. When ``False`` the
        dataset yields tuples.
    """

    def __init__(
        self,
        root: str | Path = "data",
        mask_subdir: str = "masks",
        param_subdir: str = "params",
        target_subdir: str = "simulation_results",
        metadata_path: str | Path | None = None,
        include_metadata: bool = False,
        return_dict: bool = True,
        mask_transform=None,
        param_transform=None,
        target_transform=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.root = Path(root)
        self.mask_dir = self.root / mask_subdir
        self.param_dir = self.root / param_subdir if param_subdir else None
        self.target_dir = self.root / target_subdir
        self.metadata_path = Path(metadata_path) if metadata_path is not None else self.root / "metadata.csv"
        self.include_metadata = include_metadata
        self.return_dict = return_dict
        self.mask_transform = mask_transform
        self.param_transform = param_transform
        self.target_transform = target_transform
        self.dtype = dtype

        self._metadata_map = self._load_metadata()
        self._param_order: List[str] | None = None

        self.samples = self._scan_samples()
        self.ids = [sample["design_id"] for sample in self.samples]

        self.mask_channels = 0
        self.mask_shape: Optional[Sequence[int]] = None
        self.target_dim = 0
        self.target_shape: Optional[Sequence[int]] = None
        self.param_dim = 0
        self.has_params = False

        if self.samples:
            self._infer_shapes()

    # ------------------------------------------------------------------
    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        if not self.metadata_path.exists() or self.metadata_path.is_dir():
            return {}

        if self.metadata_path.suffix.lower() != ".csv":
            warnings.warn(
                f"Unsupported metadata extension {self.metadata_path.suffix}; only CSV is supported for now.",
                RuntimeWarning,
            )
            return {}

        meta: Dict[str, Dict[str, str]] = {}
        with self.metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                design_id = row.get("design_id") or row.get("id")
                if not design_id:
                    continue
                meta[design_id] = {k: v for k, v in row.items() if k}
        return meta

    def _scan_samples(self) -> List[Dict[str, object]]:
        if not self.mask_dir.exists():
            warnings.warn(f"Mask directory {self.mask_dir} does not exist.", RuntimeWarning)
            return []

        samples: List[Dict[str, object]] = []
        for mask_path in sorted(self.mask_dir.glob("mask_*.npy")):
            design_id = mask_path.stem.split("_")[-1]
            target_path = self.target_dir / f"result_{design_id}.npy"
            if not target_path.exists():
                continue

            param_path = None
            if self.param_dir and self.param_dir.exists():
                json_candidate = self.param_dir / f"params_{design_id}.json"
                npy_candidate = self.param_dir / f"params_{design_id}.npy"
                if json_candidate.exists():
                    param_path = json_candidate
                elif npy_candidate.exists():
                    param_path = npy_candidate

            samples.append(
                {
                    "design_id": design_id,
                    "mask": mask_path,
                    "target": target_path,
                    "param_path": param_path,
                    "metadata": self._metadata_map.get(design_id),
                }
            )
        return samples

    def _infer_shapes(self) -> None:
        first = self.samples[0]
        mask = np.load(first["mask"]).astype(np.float32)
        if mask.ndim == 2:
            mask = mask[None, ...]
        self.mask_shape = tuple(mask.shape)
        self.mask_channels = mask.shape[0]

        target = np.load(first["target"]).astype(np.float32).reshape(-1)
        self.target_dim = int(target.shape[0])
        self.target_shape = (self.target_dim,)

        all_have_params = all(sample.get("param_path") is not None for sample in self.samples)
        if all_have_params and self.samples[0]["param_path"] is not None:
            param = self._load_param_vector(Path(self.samples[0]["param_path"]))
            self.param_dim = int(param.shape[0])
            self.has_params = True
        else:
            self.param_dim = 0
            self.has_params = False

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample = self.samples[idx]
        mask = np.load(sample["mask"]).astype(np.float32)
        if mask.ndim == 2:
            mask = mask[None, ...]
        mask_tensor = torch.from_numpy(mask)
        if self.mask_transform:
            mask_tensor = self.mask_transform(mask_tensor)

        target = np.load(sample["target"]).astype(np.float32).reshape(-1)
        target_tensor = torch.from_numpy(target)
        if self.target_transform:
            target_tensor = self.target_transform(target_tensor)

        params_tensor = None
        if sample.get("param_path") is not None:
            param_vec = self._load_param_vector(Path(sample["param_path"]))
            params_tensor = torch.from_numpy(param_vec.astype(np.float32))
            if self.param_transform:
                params_tensor = self.param_transform(params_tensor)
        elif self.has_params and self.param_dim > 0:
            params_tensor = torch.zeros(self.param_dim, dtype=self.dtype)

        if self.return_dict:
            out = {"mask": mask_tensor.to(self.dtype), "target": target_tensor.to(self.dtype)}
            if params_tensor is not None:
                out["params"] = params_tensor.to(self.dtype)
            if self.include_metadata:
                meta = dict(sample.get("metadata") or {})
                meta.setdefault("design_id", sample["design_id"])
                out["metadata"] = meta
            return out

        if params_tensor is not None:
            return mask_tensor.to(self.dtype), params_tensor.to(self.dtype), target_tensor.to(self.dtype)
        return mask_tensor.to(self.dtype), target_tensor.to(self.dtype)

    # ------------------------------------------------------------------
    def _load_param_vector(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if self._param_order is None:
                    self._param_order = sorted(data.keys())
                missing = set(self._param_order) - set(data.keys())
                if missing:
                    raise KeyError(f"Missing keys {missing} in parameter file {path}")
                values = [float(data[key]) for key in self._param_order]
                return np.asarray(values, dtype=np.float32)
            if isinstance(data, list):
                return np.asarray([float(v) for v in data], dtype=np.float32)
            raise TypeError(f"Unsupported JSON parameter format in {path}")

        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            return np.asarray(arr, dtype=np.float32).reshape(-1)

        raise ValueError(f"Unsupported parameter file format: {path.suffix}")


class MaskSparamSet(Dataset):
    """Backwards compatible wrapper that exposes mask/target pairs only."""

    def __init__(self, data_dir: str = "data") -> None:
        self.dataset = HybridMaskDataset(root=data_dir, return_dict=False)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataset)

    def __getitem__(self, idx: int):  # type: ignore[override]
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) == 3:
            mask, _params, target = item
            return mask, target
        return item
