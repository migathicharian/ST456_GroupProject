"""
Data loading, preprocessing, augmentation, and PyTorch Dataset definitions
for the PCam histopathology SSL project.

Usage from notebook:
    from src.data import (
        DataBundle, load_all_data,
        SimCLRTransform, make_classification_transform, make_mae_transform,
        SSLPairDataset, SSLImageDataset, ClassificationDataset,
        make_labelled_subset, make_classification_loader,
        LABEL_FRACTIONS,
    )
"""
from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# Constants

SEED = 42

LABEL_FRACTIONS: Dict[str, float] = {
    "1%": 0.01,
    "5%": 0.05,
    "10%": 0.10,
}

REQUIRED_H5_FILES: List[str] = [
    "camelyonpatch_level_2_split_train_x.h5",
    "camelyonpatch_level_2_split_train_y.h5",
    "camelyonpatch_level_2_split_valid_x.h5",
    "camelyonpatch_level_2_split_valid_y.h5",
    "camelyonpatch_level_2_split_test_x.h5",
    "camelyonpatch_level_2_split_test_y.h5",
]



# File helpers

def ensure_unzipped(src_dir: Path, dst_dir: Path, file_names: List[str]) -> None:
    """Copy or unzip the required H5 files into dst_dir if not already present."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in file_names:
        gz_path = src_dir / f"{name}.gz"
        raw_src_path = src_dir / name
        raw_dst_path = dst_dir / name

        if raw_dst_path.exists():
            continue
        if raw_src_path.exists():
            shutil.copy2(raw_src_path, raw_dst_path)
            continue
        if not gz_path.exists():
            raise FileNotFoundError(
                f"Missing both {raw_src_path.name} and {gz_path.name} in {src_dir}"
            )
        print(f"Unzipping {gz_path.name} -> {raw_dst_path}")
        with gzip.open(gz_path, "rb") as f_in, open(raw_dst_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def load_h5_array(path: Path, key: str = "x", n: Optional[int] = None) -> np.ndarray:
    with h5py.File(path, "r") as f:
        data = f[key]
        if n is None:
            return np.array(data)
        return np.array(data[:n])


def load_labels(path: Path, key: str = "y", n: Optional[int] = None) -> np.ndarray:
    arr = load_h5_array(path, key=key, n=n)
    return arr.reshape(-1)


def load_metadata(path: Path, n: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if n is not None:
        df = df.iloc[:n].copy()
    return df.reset_index(drop=True)



# Preprocessing

def preprocess_images(x: np.ndarray) -> np.ndarray:
    """Convert uint8 [0, 255] images to float32 [0, 1]."""
    return x.astype("float32") / 255.0


def remove_extreme_images(
    x: np.ndarray, y: np.ndarray, meta: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """Drop all-black or all-white patches that carry no signal."""
    black_mask = np.all(x == 0, axis=(1, 2, 3))
    white_mask = np.all(x == 1, axis=(1, 2, 3))
    keep_mask = ~(black_mask | white_mask)
    removed = int((~keep_mask).sum())
    return x[keep_mask], y[keep_mask], meta.loc[keep_mask].reset_index(drop=True), removed


def stratified_subset_indices(
    y: np.ndarray, fraction: float, seed: int = SEED
) -> np.ndarray:
    """Stratified sampling of indices that preserves the class balance in y."""
    n = len(y)
    target_n = max(1, int(round(n * fraction)))
    if target_n >= n:
        return np.arange(n)
    indices = np.arange(n)
    chosen, _ = train_test_split(
        indices, train_size=target_n, stratify=y, random_state=seed
    )
    return np.sort(chosen)


def make_labelled_subset(
    x: np.ndarray, y: np.ndarray, fraction: float, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray]:
    idx = stratified_subset_indices(y, fraction, seed=seed)
    return x[idx], y[idx]



# DataBundle: one object holding all arrays + channel stats

@dataclass
class DataBundle:
    """Container for everything Phase 1 and Phase 2 need from the dataset.

    Using a single object.
    """
    x_pretrain: np.ndarray
    y_pretrain: np.ndarray
    x_train_pool: np.ndarray
    y_train_pool: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    channel_mean: np.ndarray
    channel_std: np.ndarray
    meta_pretrain: Optional[pd.DataFrame] = None
    meta_train_pool: Optional[pd.DataFrame] = None
    meta_val: Optional[pd.DataFrame] = None
    meta_test: Optional[pd.DataFrame] = None

    def summary(self) -> None:
        """Print a short summary of the loaded data."""
        print(f"  x_pretrain:    {self.x_pretrain.shape}")
        print(f"  x_train_pool:  {self.x_train_pool.shape}")
        print(f"  x_val:         {self.x_val.shape}")
        print(f"  x_test:        {self.x_test.shape}")
        print(f"  channel_mean:  {self.channel_mean}")
        print(f"  channel_std:   {self.channel_std}")


def load_all_data(
    drive_data_dir: Path,
    data_dir: Path,
    pretrain_fraction: float = 0.15,
    downstream_pool_fraction: float = 0.15,
    use_subset: bool = True,
    seed: int = SEED,
) -> DataBundle:
    """Load PCam H5 files, sample subsets, preprocess, and return a DataBundle.

    Args:
        drive_data_dir: Folder where the raw .h5 / .h5.gz files locate.
        data_dir: Scratch folder where decompressed files are cached.
        pretrain_fraction: Fraction of training set used for SSL pre-training.
        downstream_pool_fraction: Fraction of training set kept as a pool from
            which 1, 5, 10 percent labelled subsets will later be drawn.
        use_subset: If False, use the full training split everywhere.
        seed: Random seed for stratified sampling.
    """
    drive_data_dir = Path(drive_data_dir)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not drive_data_dir.exists():
        raise FileNotFoundError(f"Raw data folder not found: {drive_data_dir}")

    ensure_unzipped(drive_data_dir, data_dir, REQUIRED_H5_FILES)

    # Paths
    train_x = data_dir / "camelyonpatch_level_2_split_train_x.h5"
    train_y = data_dir / "camelyonpatch_level_2_split_train_y.h5"
    valid_x = data_dir / "camelyonpatch_level_2_split_valid_x.h5"
    valid_y = data_dir / "camelyonpatch_level_2_split_valid_y.h5"
    test_x = data_dir / "camelyonpatch_level_2_split_test_x.h5"
    test_y = data_dir / "camelyonpatch_level_2_split_test_y.h5"

    train_meta_path = drive_data_dir / "camelyonpatch_level_2_split_train_meta.csv"
    valid_meta_path = drive_data_dir / "camelyonpatch_level_2_split_valid_meta.csv"
    test_meta_path = drive_data_dir / "camelyonpatch_level_2_split_test_meta.csv"

    # Load raw arrays
    x_train_full = load_h5_array(train_x, key="x")
    y_train_full = load_labels(train_y, key="y")
    meta_train_full = load_metadata(train_meta_path) if train_meta_path.exists() else None

    x_val = load_h5_array(valid_x, key="x")
    y_val = load_labels(valid_y, key="y")
    meta_val = load_metadata(valid_meta_path) if valid_meta_path.exists() else None

    x_test = load_h5_array(test_x, key="x")
    y_test = load_labels(test_y, key="y")
    meta_test = load_metadata(test_meta_path) if test_meta_path.exists() else None

    # Split the training set into a pretrain subset and a downstream pool.
    if use_subset:
        pretrain_idx = stratified_subset_indices(y_train_full, pretrain_fraction, seed=seed)
        downstream_pool_idx = stratified_subset_indices(
            y_train_full, downstream_pool_fraction, seed=seed + 1
        )
    else:
        pretrain_idx = np.arange(len(y_train_full))
        downstream_pool_idx = np.arange(len(y_train_full))

    x_pretrain = x_train_full[pretrain_idx]
    y_pretrain = y_train_full[pretrain_idx]
    meta_pretrain = (
        meta_train_full.iloc[pretrain_idx].reset_index(drop=True)
        if meta_train_full is not None else None
    )

    x_train_pool = x_train_full[downstream_pool_idx]
    y_train_pool = y_train_full[downstream_pool_idx]
    meta_train_pool = (
        meta_train_full.iloc[downstream_pool_idx].reset_index(drop=True)
        if meta_train_full is not None else None
    )

    # Preprocess to float [0, 1]
    x_pretrain = preprocess_images(x_pretrain)
    x_train_pool = preprocess_images(x_train_pool)
    x_val = preprocess_images(x_val)
    x_test = preprocess_images(x_test)

    # Drop extreme (all-black, all-white) images
    if meta_pretrain is not None:
        x_pretrain, y_pretrain, meta_pretrain, _ = remove_extreme_images(
            x_pretrain, y_pretrain, meta_pretrain
        )
    if meta_train_pool is not None:
        x_train_pool, y_train_pool, meta_train_pool, _ = remove_extreme_images(
            x_train_pool, y_train_pool, meta_train_pool
        )
    if meta_val is not None:
        x_val, y_val, meta_val, _ = remove_extreme_images(x_val, y_val, meta_val)
    if meta_test is not None:
        x_test, y_test, meta_test, _ = remove_extreme_images(x_test, y_test, meta_test)

    # Channel mean and std are computed only from the SSL pretrain subset
    channel_mean = x_pretrain.mean(axis=(0, 1, 2))
    channel_std = x_pretrain.std(axis=(0, 1, 2)) + 1e-8

    return DataBundle(
        x_pretrain=x_pretrain,
        y_pretrain=y_pretrain,
        x_train_pool=x_train_pool,
        y_train_pool=y_train_pool,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        channel_mean=channel_mean,
        channel_std=channel_std,
        meta_pretrain=meta_pretrain,
        meta_train_pool=meta_train_pool,
        meta_val=meta_val,
        meta_test=meta_test,
    )



# Transforms

class SimCLRTransform:
    """Two-view augmentation for SimCLR contrastive pre-training.

    tailored=True uses histopathology-specific augmentations.
    tailored=False uses a generic ImageNet-style recipe as an ablation control.
    """

    def __init__(
        self,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        size: int = 96,
        tailored: bool = True,
    ):
        mean = tuple(channel_mean)
        std = tuple(channel_std)

        if tailored:
            aug = [
                T.ToPILImage(),
                T.RandomResizedCrop(size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=90),
                T.RandomApply(
                    [T.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.4, hue=0.08)],
                    p=0.8,
                ),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        else:
            aug = [
                T.ToPILImage(),
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [T.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,
                ),
                T.GaussianBlur(kernel_size=3),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]

        self.transform = T.Compose(aug)

    def __call__(self, image: np.ndarray):
        return self.transform(image), self.transform(image)


def make_classification_transform(channel_mean: np.ndarray, channel_std: np.ndarray):
    """Minimal preprocessing for downstream classification evaluation.

    no augmentation
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(tuple(channel_mean), tuple(channel_std)),
    ])


def make_mae_transform(channel_mean: np.ndarray, channel_std: np.ndarray):
    """Small augmentation for MAE pre-training. 
    """
    return T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.Normalize(tuple(channel_mean), tuple(channel_std)),
    ])



# Dataset classes

class SSLPairDataset(Dataset):
    """Returns two differently-augmented views of the same image for SimCLR."""

    def __init__(self, images: np.ndarray, transform):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        return self.transform(image)


class SSLImageDataset(Dataset):
    """Returns one tensor per image for MAE."""

    def __init__(self, images: np.ndarray, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.transform is None:
            return torch.from_numpy(image.transpose(2, 0, 1)).float()
        return self.transform(image)


class ClassificationDataset(Dataset):
    """Image and label dataset for downstream supervised training and evaluation."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels.astype("int64")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = int(self.labels[index])
        if self.transform is None:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        else:
            image = self.transform(image)
        return image, label


# DataLoader helpers

def make_classification_loader(
    x: np.ndarray,
    y: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    transform = make_classification_transform(channel_mean, channel_std)
    dataset = ClassificationDataset(x, y, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def build_downstream_loaders(
    data: DataBundle,
    label_fractions: Dict[str, float] = LABEL_FRACTIONS,
    train_batch_size: int = 64,
    eval_batch_size: int = 128,
    seed: int = SEED,
    num_workers: int = 2,
) -> Tuple[Dict[str, DataLoader], DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for all label budgets.

    Returns:
        train_loaders: dict mapping label budget name -> DataLoader.
        val_loader, test_loader: shared across all budgets.
    """
    train_loaders: Dict[str, DataLoader] = {}
    for name, frac in label_fractions.items():
        sx, sy = make_labelled_subset(
            data.x_train_pool, data.y_train_pool, frac, seed=seed
        )
        train_loaders[name] = make_classification_loader(
            sx, sy, data.channel_mean, data.channel_std,
            batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
        )

    val_loader = make_classification_loader(
        data.x_val, data.y_val, data.channel_mean, data.channel_std,
        batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = make_classification_loader(
        data.x_test, data.y_test, data.channel_mean, data.channel_std,
        batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loaders, val_loader, test_loader
