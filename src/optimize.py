"""Bayesian optimization stub using Optuna for capacitor inverse design."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Sequence

import optuna
import torch

from data_utils import CapacitorDataset, DatasetConfig, unnormalize_targets
from models.mlp import MLP


def load_model(model_path: pathlib.Path, device: torch.device) -> tuple[MLP, Sequence[str], Sequence[str]]:
    checkpoint = torch.load(model_path, map_location=device)
    model = MLP(
        input_dim=len(checkpoint["feature_columns"]),
        output_dim=len(checkpoint["target_columns"]),
        hidden_dims=checkpoint["hidden_dims"],
        activation=checkpoint["activation"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["feature_columns"], checkpoint["target_columns"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inverse design with a trained MLP surrogate")
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--processed-csv", type=pathlib.Path, default=pathlib.Path("data/processed/train.csv"))
    parser.add_argument("--scaler-path", type=pathlib.Path, default=pathlib.Path("data/processed/scalers.json"))
    parser.add_argument("--target-cols", nargs="+", required=True)
    parser.add_argument("--feature-cols", nargs="+", required=True)
    parser.add_argument("--target-spec", nargs="+", type=float, required=True,
                        help="Desired normalized target values aligned with target-cols order")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, saved_features, saved_targets = load_model(args.model_path, device)

    if list(saved_features) != args.feature_cols:
        raise ValueError("Feature columns provided do not match the trained model")
    if list(saved_targets) != args.target_cols:
        raise ValueError("Target columns provided do not match the trained model")

    dataset = CapacitorDataset(DatasetConfig(
        csv_path=args.processed_csv,
        scaler_path=args.scaler_path,
        feature_columns=args.feature_cols,
        target_columns=args.target_cols,
    ))

    target_spec = torch.tensor(args.target_spec, dtype=torch.float32, device=device)

    def objective(trial: optuna.Trial) -> float:
        candidate = torch.tensor(
            [trial.suggest_float(col, dataset.features[:, idx].min().item(), dataset.features[:, idx].max().item())
             for idx, col in enumerate(args.feature_cols)],
            dtype=torch.float32,
            device=device,
        )
        pred = model(candidate.unsqueeze(0))
        return torch.nn.functional.mse_loss(pred.squeeze(0), target_spec).item()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    best_features = torch.tensor([study.best_params[col] for col in args.feature_cols], dtype=torch.float32)
    pred = model(best_features.unsqueeze(0))
    unnormalized = unnormalize_targets(pred, dataset)[0].tolist()

    result = {
        "best_params": study.best_params,
        "predicted_targets": dict(zip(args.target_cols, unnormalized)),
        "objective": study.best_value,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
