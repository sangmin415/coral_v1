"""Training entry point for the capacitor surrogate MLP."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_utils import CapacitorDataset, DatasetConfig
from models.mlp import MLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the capacitor surrogate MLP")
    parser.add_argument("--train-csv", type=pathlib.Path, default=pathlib.Path("data/processed/train.csv"))
    parser.add_argument("--val-csv", type=pathlib.Path, default=pathlib.Path("data/processed/val.csv"))
    parser.add_argument("--scaler-path", type=pathlib.Path, default=pathlib.Path("data/processed/scalers.json"))
    parser.add_argument("--feature-cols", nargs="+", required=True)
    parser.add_argument("--target-cols", nargs="+", required=True)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 256, 128])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/models"))
    parser.add_argument("--experiment-name", type=str, default="mlp_capacitor")
    return parser.parse_args()


def build_dataset(csv_path: pathlib.Path, scaler_path: pathlib.Path, feature_cols: Sequence[str], target_cols: Sequence[str]) -> CapacitorDataset:
    cfg = DatasetConfig(
        csv_path=csv_path,
        scaler_path=scaler_path,
        feature_columns=feature_cols,
        target_columns=target_cols,
    )
    return CapacitorDataset(cfg)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features)
            loss = criterion(preds, targets)
            total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()

    feature_cols = args.feature_cols
    target_cols = args.target_cols

    train_dataset = build_dataset(args.train_csv, args.scaler_path, feature_cols, target_cols)
    val_dataset = build_dataset(args.val_csv, args.scaler_path, feature_cols, target_cols)

    device = torch.device(args.device)

    model = MLP(
        input_dim=len(feature_cols),
        output_dim=len(target_cols),
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    epochs_without_improvement = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_columns": feature_cols,
                "target_columns": target_cols,
                "hidden_dims": args.hidden_dims,
                "activation": args.activation,
                "dropout": args.dropout,
            }, args.output_dir / f"{args.experiment_name}.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} with best validation loss {best_loss:.6f}")
            break

    with (args.output_dir / f"{args.experiment_name}_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
