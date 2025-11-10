import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import NPZPatchDataset
from src.metrics.metrics import mse, psnr, ssim3d
from src.models.unet3d import UNet3D
from src.models.res_attention_unet3d import ResAttUNet3D
from src.models.vit3d import ViT3DAE
from src.utils.train_utils import get_device, timestamp, ensure_dir, save_checkpoint


def build_model(name: str, patch_size):
    if name == 'unet3d':
        return UNet3D(in_ch=1, out_ch=1, base_ch=32)
    if name == 'resattunet3d':
        return ResAttUNet3D(in_ch=1, out_ch=1, base_ch=32)
    if name == 'vit3d':
        return ViT3DAE(in_ch=1, out_ch=1, embed_dim=256, depth=6, num_heads=8, patch_size=patch_size)
    raise ValueError(f'Unknown model: {name}')


def main():
    ap = argparse.ArgumentParser(description='Train model on augmented seismic patches')
    ap.add_argument('--model', required=True, choices=['unet3d', 'resattunet3d', 'vit3d'])
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patch-size', type=int, nargs=3, default=[32,32,32])
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = get_device() if args.device is None else torch.device(args.device)
    run_dir = os.path.join('runs', args.model, timestamp())
    ensure_dir(run_dir)

    train_npz = os.path.join(args.data_dir, 'augmented', 'train_patches.npz')
    val_npz = os.path.join(args.data_dir, 'augmented', 'val_patches.npz')
    train_ds = NPZPatchDataset(train_npz)
    val_ds = NPZPatchDataset(val_npz)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, tuple(args.patch_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_psnr = -1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]')
        total_loss = 0.0
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_ds)

        model.eval()
        val_mse = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} [val]'):
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_mse += mse(pred, yb).item() * xb.size(0)
                val_psnr += psnr(pred, yb).item() * xb.size(0)
                val_ssim += ssim3d(pred, yb).item() * xb.size(0)
                n_val += xb.size(0)
        val_mse /= n_val
        val_psnr /= n_val
        val_ssim /= n_val

        print(f'Epoch {epoch}: train_loss={avg_loss:.6f} val_mse={val_mse:.6f} val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f}')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, opt, epoch, best_psnr, run_dir, name='best.pt')

    print(f'Training complete. Best PSNR: {best_psnr:.4f}. Checkpoints: {run_dir}')


if __name__ == '__main__':
    main()

