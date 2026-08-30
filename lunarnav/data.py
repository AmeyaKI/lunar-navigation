"""Dataset download, filtering, splits, and RGB-only RockDataset."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from lunarnav.labels import compute_relative_distance, compute_relative_height

from lunarnav.constants import EXPECTED_FRAME_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, TARGET_NAMES

DATASET_SLUG = "romainpessia/artificial-lunar-rocky-landscape-dataset"


def download_dataset(cache_dir: str | Path | None = None) -> Path:
    """Download Kaggle dataset via kagglehub; optional cache directory on Drive."""
    if cache_dir is not None:
        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    return dataset_path


def make_writable_copy(dataset_path: Path, writable_root: Path) -> Path:
    """Copy dataset to a writable location if not already present."""
    writable_root = Path(writable_root)
    if writable_root.exists():
        return writable_root
    shutil.copytree(dataset_path, writable_root)
    return writable_root


def filter_faulty_frames(dataset_path: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    """Remove faulty frames listed in dataset-root *.txt files (773 frames)."""
    dataset_path = Path(dataset_path)
    txt_files = [name for name in os.listdir(dataset_path) if name.endswith(".txt")]
    faulty_images: list[str] = []
    for txt in txt_files:
        with open(dataset_path / txt) as handle:
            for line in handle:
                faulty_images.append(f"render{line.strip()}.png")

    img_path = dataset_path / "images" / "render"
    faulty_image_nums = [int(name[6:10]) for name in faulty_images]

    for img in os.listdir(img_path):
        if img.endswith(".png") and img in faulty_images:
            os.remove(img_path / img)

    filtered_df = df[~df["Frame"].isin(faulty_image_nums)].reset_index(drop=True)
    return filtered_df


def split_frames(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Frame-level train/val/test split: 72% / 8% / 20%."""
    frames = df["Frame"].unique()
    train_val_frames, test_frames = train_test_split(
        frames,
        test_size=0.20,
        random_state=random_state,
        shuffle=True,
    )
    train_frames, val_frames = train_test_split(
        train_val_frames,
        test_size=0.08 / 0.80,
        random_state=random_state,
        shuffle=True,
    )
    train_df = df[df["Frame"].isin(train_frames)].reset_index(drop=True)
    val_df = df[df["Frame"].isin(val_frames)].reset_index(drop=True)
    test_df = df[df["Frame"].isin(test_frames)].reset_index(drop=True)
    return train_df, val_df, test_df


def bbox_ridge_features(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_h: int = IMAGE_HEIGHT,
    img_w: int = IMAGE_WIDTH,
) -> np.ndarray:
    """Bbox-only ridge features: [rel_width, rel_height_px, rel_area, cx/W, cy/H]."""
    rel_width = (x2 - x1) / img_w
    rel_height_px = (y2 - y1) / img_h
    rel_area = rel_width * rel_height_px
    cx_norm = ((x1 + x2) / 2) / img_w
    cy_norm = ((y1 + y2) / 2) / img_h
    return np.array([rel_width, rel_height_px, rel_area, cx_norm, cy_norm], dtype=np.float32)


def _bbox_from_row(row: pd.Series) -> tuple[int, int, int, int]:
    x1 = int(row["TopLeftCornerX"])
    y1 = int(row["TopLeftCornerY"])
    x2 = x1 + int(row["Length"])
    y2 = y1 + int(row["Height"])
    return x1, y1, x2, y2


class RockDataset(Dataset):
    """RGB crop dataset with MiDaS pseudo-label targets (distance, height only)."""

    def __init__(self, df: pd.DataFrame, image_dir: str | Path, depth_dir: str | Path):
        import torch
        from torchvision import transforms as T

        self.df = df
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir)
        self.groups = self.df.groupby("Frame")
        self.index: list[tuple[int, int]] = []
        for frame, group in self.groups:
            for box_idx in range(len(group)):
                self.index.append((frame, box_idx))

        self.rgb_tf = T.Compose(
            [
                T.ToTensor(),
                T.Resize((224, 224)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        import torch

        frame, box_idx = self.index[idx]
        group = self.groups.get_group(frame).iloc[box_idx]

        img_path = self.image_dir / f"render{str(frame).zfill(4)}.png"
        depth_path = self.depth_dir / f"render{str(frame).zfill(4)}.npy"

        rgb_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        x1, y1, x2, y2 = _bbox_from_row(group)
        rgb_box = rgb_img[y1:y2, x1:x2]

        rel_height = compute_relative_height(depth, (x1, y1, x2, y2))
        rel_distance = compute_relative_distance(depth, (x1, y1, x2, y2))

        target = torch.tensor([rel_distance, rel_height], dtype=torch.float32)
        return self.rgb_tf(rgb_box), target


def collect_split_arrays(
    df: pd.DataFrame,
    image_dir: str | Path,
    depth_dir: str | Path,
    img_h: int = IMAGE_HEIGHT,
    img_w: int = IMAGE_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect ridge features, targets, and RGB crops for a split (for baselines/eval)."""
    image_dir = Path(image_dir)
    depth_dir = Path(depth_dir)
    groups = df.groupby("Frame")

    ridge_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    rgb_crops: list[np.ndarray] = []

    for frame, group in groups:
        img_path = image_dir / f"render{str(frame).zfill(4)}.png"
        depth_path = depth_dir / f"render{str(frame).zfill(4)}.npy"
        rgb_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        for _, row in group.iterrows():
            x1, y1, x2, y2 = _bbox_from_row(row)
            ridge_rows.append(bbox_ridge_features(x1, y1, x2, y2, img_h, img_w))
            rel_height = compute_relative_height(depth, (x1, y1, x2, y2))
            rel_distance = compute_relative_distance(depth, (x1, y1, x2, y2))
            target_rows.append([rel_distance, rel_height])
            rgb_crops.append(rgb_img[y1:y2, x1:x2])

    return (
        np.stack(ridge_rows),
        np.stack(target_rows),
        np.array(rgb_crops, dtype=object),
    )


def prepare_data(
    cache_root: str | Path,
    kaggle_cache_dir: str | Path | None = None,
) -> dict[str, Path | pd.DataFrame]:
    """Download, filter, split, and return paths plus train/val/test dataframes."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    raw_path = download_dataset(kaggle_cache_dir)
    writable_path = make_writable_copy(raw_path, cache_root / "dataset")

    bbox_csv = writable_path / "bounding_boxes.csv"
    df = pd.read_csv(bbox_csv)
    filtered_df = filter_faulty_frames(writable_path, df)

    frame_count = len(list((writable_path / "images" / "render").glob("*.png")))
    if frame_count != EXPECTED_FRAME_COUNT:
        raise ValueError(f"Expected {EXPECTED_FRAME_COUNT} frames after filter, got {frame_count}")

    filtered_csv = writable_path / "filtered_bboxes.csv"
    filtered_df.to_csv(filtered_csv, index=False)

    train_df, val_df, test_df = split_frames(filtered_df)

    image_dir = writable_path / "images" / "render"
    depth_dir = writable_path / "images" / "depth_maps"
    depth_dir.mkdir(parents=True, exist_ok=True)

    return {
        "writable_path": writable_path,
        "image_dir": image_dir,
        "depth_dir": depth_dir,
        "filtered_csv": filtered_csv,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "frame_count": frame_count,
    }
