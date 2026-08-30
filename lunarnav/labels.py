"""MiDaS pseudo-label geometry from depth maps and bounding boxes.

These targets are **not** ground-truth metric geometry. They are deterministic
functions of MiDaS DPT-Hybrid relative depth and dataset-provided boxes.
"""

from __future__ import annotations

import numpy as np


def compute_relative_height(depth: np.ndarray, bbox: tuple[int, int, int, int], ring: int = 8) -> float:
    """Relative height: median ground ring depth minus median rock depth."""
    x1, y1, x2, y2 = bbox

    rock_depth = float(np.median(depth[y1:y2, x1:x2]))

    y1g = max(0, y1 - ring)
    y2g = min(depth.shape[0], y2 + ring)
    x1g = max(0, x1 - ring)
    x2g = min(depth.shape[1], x2 + ring)

    ground_mask = np.ones((y2g - y1g, x2g - x1g), dtype=bool)
    ground_mask[(y1 - y1g) : (y2 - y1g), (x1 - x1g) : (x2 - x1g)] = False

    ground_depth = float(np.median(depth[y1g:y2g, x1g:x2g][ground_mask]))
    return ground_depth - rock_depth


def compute_relative_distance(depth: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Relative distance: MiDaS depth at bbox center pixel."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return float(depth[cy, cx])


def compute_relative_size(bbox: tuple[int, int, int, int], img_shape: tuple[int, ...]) -> float:
    """Relative box area (used for ridge baseline features, not a model target)."""
    x1, y1, x2, y2 = bbox
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_shape[0] * img_shape[1]
    return box_area / img_area
