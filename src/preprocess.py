"""Preprocessing utilities for capacitor datasets.

This script normalizes Excel/CSV datasets exported from ADS or lab
measurements, splits them into train/validation/test partitions, and
stores the scalers for later reuse during inference and optimization.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessConfig:
    input_path: pathlib.Path
    output_dir: pathlib.Path
    feature_columns: Sequence[str]
    target_columns: Sequence[str]
    sheet_name: str | int | None
    test_size: float
    val_size: float
    random_state: int


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Normalize and split capacitor datasets")
    parser.add_argument("--input-path", type=pathlib.Path, required=True,
                        help="Path to the Excel/CSV dataset provided by the instructor")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("data/processed"),
                        help="Directory to store normalized CSV files and scaler metadata")
    parser.add_argument("--feature-cols", nargs="+", required=True,
                        help="Column names to use as input features (e.g., L_um W_um)")
    parser.add_argument("--target-cols", nargs="+", required=True,
                        help="Column names to predict (e.g., C_pf Q_factor)")
    parser.add_argument("--sheet", default=None,
                        help="Excel sheet name or index. Ignored for CSV inputs.")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Fraction reserved for the test set")
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Fraction reserved for the validation set (relative to train+val subset)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for deterministic splits")
    args = parser.parse_args()

    if args.input_path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("input-path must be a CSV or Excel file")

    return PreprocessConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        feature_columns=args.feature_cols,
        target_columns=args.target_cols,
        sheet_name=args.sheet,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )


def load_dataframe(cfg: PreprocessConfig) -> pd.DataFrame:
    if cfg.input_path.suffix.lower() == ".csv":
        df = pd.read_csv(cfg.input_path)
    else:
        df = pd.read_excel(cfg.input_path, sheet_name=cfg.sheet_name)
    missing_cols: List[str] = [col for col in (*cfg.feature_columns, *cfg.target_columns) if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in dataset: {missing_cols}")
    return df


def normalize_and_split(df: pd.DataFrame, cfg: PreprocessConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    features = df[list(cfg.feature_columns)].to_numpy(dtype=np.float32)
    targets = df[list(cfg.target_columns)].to_numpy(dtype=np.float32)

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        targets,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    val_fraction = cfg.val_size / (1.0 - cfg.test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1 - val_fraction,
        random_state=cfg.random_state,
    )

    feature_scaler = StandardScaler().fit(x_train)
    target_scaler = StandardScaler().fit(y_train)

    def save_split(x: np.ndarray, y: np.ndarray, split: str) -> None:
        out_path = cfg.output_dir / f"{split}.csv"
        data = np.concatenate([feature_scaler.transform(x), target_scaler.transform(y)], axis=1)
        columns = [*cfg.feature_columns, *cfg.target_columns]
        pd.DataFrame(data, columns=columns).to_csv(out_path, index=False)

    save_split(x_train, y_train, "train")
    save_split(x_val, y_val, "val")
    save_split(x_test, y_test, "test")

    scaler_meta = {
        "feature_columns": list(cfg.feature_columns),
        "target_columns": list(cfg.target_columns),
        "feature_mean": feature_scaler.mean_.tolist(),
        "feature_scale": feature_scaler.scale_.tolist(),
        "target_mean": target_scaler.mean_.tolist(),
        "target_scale": target_scaler.scale_.tolist(),
    }

    with (cfg.output_dir / "scalers.json").open("w", encoding="utf-8") as f:
        json.dump(scaler_meta, f, indent=2)


def main() -> None:
    cfg = parse_args()
    df = load_dataframe(cfg)
    normalize_and_split(df, cfg)


if __name__ == "__main__":
    main()
