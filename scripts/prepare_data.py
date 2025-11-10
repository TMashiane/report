import argparse
import json
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Rest of your imports
from src.utils.seg_loader import load_volume
from src.data.dataset import normalize_minmax, compute_time_slice_split, PatchSpec, generate_augmented_patches, save_split_json


def save_sample_images(volume: np.ndarray, outdir: str, n: int = 3):
    os.makedirs(outdir, exist_ok=True)
    D, H, W = volume.shape
    zs = np.linspace(0, D - 1, n, dtype=int)
    for i, z in enumerate(zs):
        plt.figure(figsize=(8, 6))
        plt.imshow(volume[z], cmap='gray')
        plt.title(f'Original time-slice z={z}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'original_slice_{i}_z{z}.png'), dpi=150)
        plt.close()


def save_augmented_patch_examples(volume: np.ndarray, patch_size, outdir: str, n: int = 3, seed: int = 123):
    from src.data.dataset import RandomPatchDataset
    os.makedirs(outdir, exist_ok=True)
    ds = RandomPatchDataset(volume, PatchSpec(*patch_size), count=n, seed=seed, augment=True)
    for i in range(n):
        x, y = ds[i]  # x: [1,D,H,W], y: [1,D,H,W]
        xd = x.squeeze(0).numpy()
        yd = y.squeeze(0).numpy()
        d = xd.shape[0] // 2
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(xd[d], cmap='gray'); plt.axis('off'); plt.title('Augmented patch (mid z)')
        plt.subplot(1, 2, 2)
        plt.imshow(yd[d], cmap='gray'); plt.axis('off'); plt.title('Original patch (mid z)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'augmented_pair_{i}.png'), dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description='Prepare Orange Basin data: split, augment, save patches and samples.')
    ap.add_argument('--input', required=True, help='Path to SEG/SEGY or NPY/NPZ')
    ap.add_argument('--outdir', default='data')
    ap.add_argument('--patch-size', type=int, nargs=3, default=[64, 64, 64], help='dz dy dx')
    ap.add_argument('--train-patches', type=int, default=2000)
    ap.add_argument('--val-patches', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    raw_dir = os.path.join(args.outdir, 'raw')
    split_dir = os.path.join(args.outdir, 'splits')
    aug_dir = os.path.join(args.outdir, 'augmented')
    sample_dir = os.path.join(args.outdir, 'samples')
    os.makedirs(raw_dir, exist_ok=True)

    vol = load_volume(args.input)
    vol = np.asarray(vol)
    vol = vol.astype(np.float32)
    vol_norm, vmin, vmax = normalize_minmax(vol)
    np.save(os.path.join(raw_dir, 'orange_basin.npy'), vol_norm)

    D, H, W = vol_norm.shape
    split = compute_time_slice_split(D)
    save_split_json(os.path.join(split_dir, 'split.json'), split)

    patch = PatchSpec(dz=args.patch_size[0], dy=args.patch_size[1], dx=args.patch_size[2])
    generate_augmented_patches(vol_norm, patch, args.train_patches, args.val_patches, aug_dir, seed=args.seed)

    # Save a few original samples (augmented samples are saved later during training/eval as needed)
    save_sample_images(vol_norm, sample_dir, n=3)
    save_augmented_patch_examples(vol_norm, (patch.dz, patch.dy, patch.dx), sample_dir, n=3, seed=args.seed + 123)

    print('Prepared data:')
    print(f'- Raw volume saved: {os.path.join(raw_dir, "orange_basin.npy")}')
    print(f'- Split saved: {os.path.join(split_dir, "split.json")}')
    print(f'- Augmented patches: {aug_dir}')
    print(f'- Sample images: {sample_dir}')


if __name__ == '__main__':
    main()
