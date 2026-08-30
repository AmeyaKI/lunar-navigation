"""Tests for MiDaS pseudo-label geometry math."""

import numpy as np

from lunarnav.labels import compute_relative_distance, compute_relative_height, compute_relative_size


def test_compute_relative_distance_center_pixel():
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    # bbox (0,0)-(2,2) -> center (1,1)
    assert compute_relative_distance(depth, (0, 0, 2, 2)) == depth[1, 1]


def test_compute_relative_height_known_medians():
    depth = np.ones((20, 20), dtype=np.float32)
    depth[5:10, 5:10] = 0.2  # rock region
    # ring around rock at default ring=8 includes pixels still at 1.0
    height = compute_relative_height(depth, (5, 5, 10, 10), ring=8)
    assert np.isclose(height, 1.0 - 0.2)


def test_compute_relative_height_clips_at_image_edge():
    depth = np.ones((10, 10), dtype=np.float32)
    depth[0:3, 0:3] = 0.1
    height = compute_relative_height(depth, (0, 0, 3, 3), ring=8)
    assert np.isfinite(height)


def test_compute_relative_size():
    size = compute_relative_size((0, 0, 100, 50), (480, 720))
    assert np.isclose(size, (100 * 50) / (480 * 720))
