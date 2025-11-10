import json
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def normalize_minmax(vol: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    vmin = float(vol.min())
    vmax = float(vol.max())
    rng = max(vmax - vmin, eps)
    norm = (vol - vmin) / rng
    return norm.astype(np.float32), vmin, vmax


def add_gaussian_noise(x: torch.Tensor, sigma: Tuple[float, float] = (0.01, 0.05)) -> torch.Tensor:
    s = torch.empty(1, device=x.device).uniform_(sigma[0], sigma[1]).item()
    noise = torch.randn_like(x) * s
    return torch.clamp(x + noise, 0.0, 1.0)


def add_speckle_noise(x: torch.Tensor, sigma: Tuple[float, float] = (0.01, 0.05)) -> torch.Tensor:
    s = torch.empty(1, device=x.device).uniform_(sigma[0], sigma[1]).item()
    noise = torch.randn_like(x) * s
    return torch.clamp(x + x * noise, 0.0, 1.0)


def gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g


def gaussian_blur_3d(x: torch.Tensor, sigma: Tuple[float, float] = (0.5, 1.5), ksize: int = 7) -> torch.Tensor:
    # x: [B,1,D,H,W]
    s = torch.empty(1, device=x.device).uniform_(sigma[0], sigma[1]).item()
    k = gaussian_kernel_1d(ksize, s, x.device)
    kx = k.view(1, 1, -1, 1, 1)
    ky = k.view(1, 1, 1, -1, 1)
    kz = k.view(1, 1, 1, 1, -1)
    x = F.conv3d(x, kx, padding=(ksize // 2, 0, 0), groups=1)
    x = F.conv3d(x, ky, padding=(0, ksize // 2, 0), groups=1)
    x = F.conv3d(x, kz, padding=(0, 0, ksize // 2), groups=1)
    return x


def augment_volume_patch(x: torch.Tensor) -> torch.Tensor:
    # x: [1,D,H,W] in [0,1]
    x = x.unsqueeze(0)  # [B=1,1,D,H,W]
    if torch.rand(1).item() < 0.7:
        x = gaussian_blur_3d(x)
    if torch.rand(1).item() < 0.9:
        if torch.rand(1).item() < 0.5:
            x = add_gaussian_noise(x)
        else:
            x = add_speckle_noise(x)
    return x.squeeze(0)


@dataclass
class PatchSpec:
    dz: int
    dy: int
    dx: int


class RandomPatchDataset(Dataset):
    def __init__(self, volume: np.ndarray, patch: PatchSpec, count: int, seed: int = 42,
                 augment: bool = True):
        assert volume.ndim == 3
        self.vol = torch.from_numpy(volume).float()  # [D,H,W]
        self.patch = patch
        self.count = count
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.D, self.H, self.W = self.vol.shape

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        z0 = self.rng.integers(0, self.D - self.patch.dz + 1)
        y0 = self.rng.integers(0, self.H - self.patch.dy + 1)
        x0 = self.rng.integers(0, self.W - self.patch.dx + 1)
        patch = self.vol[z0:z0 + self.patch.dz, y0:y0 + self.patch.dy, x0:x0 + self.patch.dx]
        target = patch.clone()
        patch = patch.unsqueeze(0)  # [1,D,H,W]
        if self.augment:
            patch = augment_volume_patch(patch)
        return patch, target.unsqueeze(0)


def save_split_json(path: str, split: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(split, f, indent=2)


def compute_time_slice_split(D: int, train: float = 0.7, val: float = 0.15):
    idx = np.arange(D)
    n_train = int(train * D)
    n_val = int(val * D)
    split = {
        'train': idx[:n_train].tolist(),
        'val': idx[n_train:n_train + n_val].tolist(),
        'test': idx[n_train + n_val:].tolist(),
    }
    return split


def generate_augmented_patches(volume: np.ndarray, patch: PatchSpec, n_train: int, n_val: int,
                                outdir: str, seed: int = 42):
    os.makedirs(outdir, exist_ok=True)
    train_ds = RandomPatchDataset(volume, patch, n_train, seed=seed, augment=True)
    val_ds = RandomPatchDataset(volume, patch, n_val, seed=seed + 1, augment=True)

    def dump(ds: RandomPatchDataset, out_path: str):
        X = torch.zeros((len(ds), 1, ds.patch.dz, ds.patch.dy, ds.patch.dx), dtype=torch.float32)
        Y = torch.zeros_like(X)
        for i in range(len(ds)):
            x, y = ds[i]
            X[i] = x
            Y[i] = y
        np.savez_compressed(out_path, X=X.numpy(), Y=Y.numpy())

    dump(train_ds, os.path.join(outdir, 'train_patches.npz'))
    dump(val_ds, os.path.join(outdir, 'val_patches.npz'))


class NPZPatchDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data['X']).float()
        self.Y = torch.from_numpy(data['Y']).float()
        assert self.X.shape == self.Y.shape

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

