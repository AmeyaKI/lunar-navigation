"""Training loop with per-epoch loss reset and validation early stopping."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_path: str | Path,
    max_epochs: int = 15,
    patience: int = 3,
    lr: float = 1e-3,
) -> dict:
    """Train with MSE loss, per-epoch loss reset, and early stopping on val MAE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for rgb, target in train_loader:
            rgb, target = rgb.to(device), target.to(device)
            pred = model(rgb)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        val_mae = _evaluate_mae(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_mae={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return {"history": history, "best_val_mae": best_val_mae}


def _evaluate_mae(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_mae = 0.0
    with torch.no_grad():
        for rgb, target in loader:
            rgb, target = rgb.to(device), target.to(device)
            pred = model(rgb)
            total_mae += torch.mean(torch.abs(pred - target)).item()
    return total_mae / len(loader)
