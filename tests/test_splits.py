"""Tests for deterministic frame-level splits."""

import numpy as np
import pandas as pd

from lunarnav.data import bbox_ridge_features, split_frames


def _make_df(n_frames: int = 100) -> pd.DataFrame:
    frames = np.arange(1, n_frames + 1)
    rows = []
    for frame in frames:
        rows.append(
            {
                "Frame": frame,
                "TopLeftCornerX": 10,
                "TopLeftCornerY": 20,
                "Length": 30,
                "Height": 40,
            }
        )
    return pd.DataFrame(rows)


def test_split_is_deterministic():
    df = _make_df()
    train_a, val_a, test_a = split_frames(df, random_state=42)
    train_b, val_b, test_b = split_frames(df, random_state=42)
    assert set(train_a["Frame"]) == set(train_b["Frame"])
    assert set(val_a["Frame"]) == set(val_b["Frame"])
    assert set(test_a["Frame"]) == set(test_b["Frame"])


def test_split_frames_disjoint():
    df = _make_df()
    train_df, val_df, test_df = split_frames(df, random_state=42)
    train_frames = set(train_df["Frame"])
    val_frames = set(val_df["Frame"])
    test_frames = set(test_df["Frame"])
    assert train_frames.isdisjoint(val_frames)
    assert train_frames.isdisjoint(test_frames)
    assert val_frames.isdisjoint(test_frames)


def test_split_proportions_72_8_20():
    df = _make_df(n_frames=1000)
    train_df, val_df, test_df = split_frames(df, random_state=42)
    total = len(df["Frame"].unique())
    train_pct = len(train_df["Frame"].unique()) / total
    val_pct = len(val_df["Frame"].unique()) / total
    test_pct = len(test_df["Frame"].unique()) / total
    assert abs(train_pct - 0.72) < 0.02
    assert abs(val_pct - 0.08) < 0.02
    assert abs(test_pct - 0.20) < 0.02


def test_bbox_ridge_features_shape_and_values():
    features = bbox_ridge_features(0, 0, 72, 48, img_h=480, img_w=720)
    assert features.shape == (5,)
    assert np.isclose(features[0], 72 / 720)
    assert np.isclose(features[1], 48 / 480)
    assert np.isclose(features[2], features[0] * features[1])
    assert np.isclose(features[3], (36) / 720)
    assert np.isclose(features[4], (24) / 480)
