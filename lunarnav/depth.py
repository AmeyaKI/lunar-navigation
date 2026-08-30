"""MiDaS DPT-Hybrid depth generation with Drive-backed caching."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Min-max normalize depth to [0, 1]; all zeros if range is negligible."""
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min < 1e-5:
        return np.zeros_like(depth)
    return (depth - depth_min) / (depth_max - depth_min)


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_midas(device: torch.device) -> tuple[torch.nn.Module, object]:
    """Load DPT-Hybrid MiDaS model and DPT transform."""
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    return midas, transform


def predict_depth(
    image: np.ndarray,
    midas: torch.nn.Module,
    transform,
    device: torch.device,
) -> np.ndarray:
    input_batch = transform(image).to(device)
    with torch.no_grad():
        pred = midas(input_batch)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    return pred.cpu().numpy()


def generate_depth_maps(
    image_dir: str | Path,
    depth_dir: str | Path,
    device: torch.device,
) -> int:
    """Generate normalized depth maps; skip frames whose .npy already exists."""
    image_dir = Path(image_dir)
    depth_dir = Path(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"))
    midas, transform = load_midas(device)

    generated = 0
    for image_path in tqdm(image_paths, desc="Depth maps"):
        out_path = depth_dir / f"{image_path.stem}.npy"
        if out_path.exists():
            continue
        image = load_image(image_path)
        depth = predict_depth(image, midas, transform, device)
        depth_norm = normalize_depth(depth)
        np.save(out_path, depth_norm)
        generated += 1

    return generated
