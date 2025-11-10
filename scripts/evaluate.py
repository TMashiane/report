import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import NPZPatchDataset
from src.metrics.metrics import mse, rmse, psnr, ssim3d, ms_ssim3d
from src.models.unet3d import UNet3D
from src.models.res_attention_unet3d import ResAttUNet3D
from src.models.vit3d import ViT3DAE
from src.utils.train_utils import get_device, load_checkpoint


def build_model(name: str, patch_size):
    if name == 'unet3d':
        return UNet3D(in_ch=1, out_ch=1, base_ch=32)
    if name == 'resattunet3d':
        return ResAttUNet3D(in_ch=1, out_ch=1, base_ch=32)
    if name == 'vit3d':
        return ViT3DAE(in_ch=1, out_ch=1, embed_dim=256, depth=6, num_heads=8, patch_size=patch_size)
    raise ValueError(f'Unknown model: {name}')


def save_sample_preds(x, y, pred, outdir, idx):
    # Save mid-slice along depth for quick visuals
    os.makedirs(outdir, exist_ok=True)
    x = x.squeeze().cpu().numpy()
    y = y.squeeze().cpu().numpy()
    p = pred.squeeze().detach().cpu().numpy()
    d = x.shape[0] // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(x[d], cmap='gray'); plt.axis('off'); plt.title('Augmented (input)')
    plt.subplot(1, 3, 2)
    plt.imshow(y[d], cmap='gray'); plt.axis('off'); plt.title('Target (original)')
    plt.subplot(1, 3, 3)
    plt.imshow(p[d], cmap='gray'); plt.axis('off'); plt.title('Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'sample_{idx}.png'), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Evaluate a trained model and compute metrics')
    ap.add_argument('--model', required=True, choices=['unet3d', 'resattunet3d', 'vit3d'])
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--patch-size', type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument('--device', default=None)
    ap.add_argument('--save-samples', default='runs/eval_samples')
    ap.add_argument('--out-json', default=None, help='Optional path to save metrics as JSON')
    args = ap.parse_args()

    device = get_device() if args.device is None else torch.device(args.device)

    val_npz = os.path.join(args.data_dir, 'augmented', 'val_patches.npz')
    val_ds = NPZPatchDataset(val_npz)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, tuple(args.patch_size)).to(device)
    load_checkpoint(model, None, args.ckpt)
    model.eval()

    m_mse = 0.0
    m_rmse = 0.0
    m_psnr = 0.0
    m_ssim = 0.0
    m_msssim = 0.0
    n = 0

    with torch.no_grad():
        for i, (xb, yb) in enumerate(tqdm(val_loader, desc='Evaluating')):
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            m_mse += mse(pred, yb).item() * xb.size(0)
            m_rmse += rmse(pred, yb).item() * xb.size(0)
            m_psnr += psnr(pred, yb).item() * xb.size(0)
            m_ssim += ssim3d(pred, yb).item() * xb.size(0)
            m_msssim += ms_ssim3d(pred, yb).item() * xb.size(0)
            n += xb.size(0)
            if i < 5:
                save_sample_preds(xb[0], yb[0], pred[0], args.save_samples, i)

    m_mse /= n
    m_rmse /= n
    m_psnr /= n
    m_ssim /= n
    m_msssim /= n

    results = {
        'MSE': float(m_mse),
        'RMSE': float(m_rmse),
        'PSNR': float(m_psnr),
        'SSIM': float(m_ssim),
        'MS-SSIM': float(m_msssim),
    }
    print('Evaluation results:')
    print(f"- MSE    : {results['MSE']:.6f}")
    print(f"- RMSE   : {results['RMSE']:.6f}")
    print(f"- PSNR   : {results['PSNR']:.4f} dB")
    print(f"- SSIM   : {results['SSIM']:.4f}")
    print(f"- MS-SSIM: {results['MS-SSIM']:.4f}")

    if args.out_json:
        import json
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics JSON: {args.out_json}")


if __name__ == '__main__':
    main()
