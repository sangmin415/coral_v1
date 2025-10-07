"""Dataset helpers for capacitor surrogate modeling."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    csv_path: pathlib.Path
    scaler_path: pathlib.Path
    feature_columns: Sequence[str]
    target_columns: Sequence[str]


class CapacitorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, cfg: DatasetConfig) -> None:
        df = pd.read_csv(cfg.csv_path)
        with cfg.scaler_path.open("r", encoding="utf-8") as f:
            scalers = json.load(f)
        self.features = torch.from_numpy(df[list(cfg.feature_columns)].to_numpy(dtype=np.float32))
        self.targets = torch.from_numpy(df[list(cfg.target_columns)].to_numpy(dtype=np.float32))
        self.feature_columns = list(cfg.feature_columns)
        self.target_columns = list(cfg.target_columns)
        self.feature_mean = torch.tensor(scalers["feature_mean"], dtype=torch.float32)
        self.feature_scale = torch.tensor(scalers["feature_scale"], dtype=torch.float32)
        self.target_mean = torch.tensor(scalers["target_mean"], dtype=torch.float32)
        self.target_scale = torch.tensor(scalers["target_scale"], dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def unnormalize_targets(predictions: torch.Tensor, dataset: CapacitorDataset) -> np.ndarray:
    return (predictions.detach().cpu().numpy() * dataset.target_scale.numpy()) + dataset.target_mean.numpy()
