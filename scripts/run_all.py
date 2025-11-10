import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(cmd, cwd=None):
    print("[run]", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.stdout


def latest_run_dir(model: str) -> Path:
    root = Path('runs') / model
    if not root.exists():
        raise FileNotFoundError(f"No run directory found for model {model}")
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No sub-runs under {root}")
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def ensure_data(args):
    train_npz = Path(args.outdir) / 'augmented' / 'train_patches.npz'
    val_npz = Path(args.outdir) / 'augmented' / 'val_patches.npz'
    if train_npz.exists() and val_npz.exists():
        print("Data already prepared; skipping prepare step.")
        return
    input_path = args.input or str(Path('src') / 'data' / 'Orange Basin_cropped 1')
    cmd = [sys.executable, 'scripts/prepare_data.py',
           '--input', input_path,
           '--outdir', args.outdir,
           '--patch-size', str(args.patch_size[0]), str(args.patch_size[1]), str(args.patch_size[2]),
           '--train-patches', str(args.train_patches),
           '--val-patches', str(args.val_patches),
           '--seed', str(args.seed)]
    print(cmd)
    run(cmd)


def train_one(model: str, args):
    batch = args.batch_size
    lr = args.lr
    if model == 'vit3d':
        batch = min(batch, 1)
        lr = min(lr, 1e-4)
    cmd = [sys.executable, 'scripts/train.py',
           '--model', model,
           '--data-dir', args.outdir,
           '--batch-size', str(batch),
           '--epochs', str(args.epochs),
           '--lr', str(lr),
           '--patch-size', str(args.patch_size[0]), str(args.patch_size[1]), str(args.patch_size[2])]
    if args.device:
        cmd += ['--device', args.device]
    t0 = time.time()
    run(cmd)
    t1 = time.time()
    rdir = latest_run_dir(model)
    ckpt = rdir / 'best.pt'
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return str(ckpt), str(rdir), (t1 - t0)


def evaluate_one(model: str, ckpt: str, args):
    out_json = Path('runs') / model / 'metrics.json'
    save_samples = Path('runs') / 'eval_samples' / model
    cmd = [sys.executable, 'scripts/evaluate.py',
           '--model', model,
           '--data-dir', args.outdir,
           '--ckpt', ckpt,
           '--batch-size', str(args.eval_batch_size),
           '--patch-size', str(args.patch_size[0]), str(args.patch_size[1]), str(args.patch_size[2]),
           '--save-samples', str(save_samples),
           '--out-json', str(out_json)]
    if args.device:
        cmd += ['--device', args.device]
    t0 = time.time()
    run(cmd)
    t1 = time.time()
    with open(out_json, 'r') as f:
        metrics = json.load(f)
    return metrics, (t1 - t0)


def main():
    ap = argparse.ArgumentParser(description='Run all models end-to-end: prepare, train, evaluate, compare')
    ap.add_argument('--input', default=None, help='Raw data path or directory (default: src/data/Orange Basin_cropped 1)')
    ap.add_argument('--outdir', default='data')
    ap.add_argument('--patch-size', type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument('--train-patches', type=int, default=2000)
    ap.add_argument('--val-patches', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--eval-batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', nargs='*', default=['unet3d', 'resattunet3d', 'vit3d'],
                    choices=['unet3d', 'resattunet3d', 'vit3d'])
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    print("== Prepare data ==")
    t_prepare0 = time.time()
    ensure_data(args)
    t_prepare1 = time.time()

    summary = {}
    for m in args.models:
        print(f"== Train {m} ==")
        ckpt, rdir, t_train = train_one(m, args)
        print(f"== Evaluate {m} ==")
        metrics, t_eval = evaluate_one(m, ckpt, args)
        summary[m] = {
            **metrics,
            'train_seconds': round(t_train, 3),
            'eval_seconds': round(t_eval, 3),
            'total_seconds': round(t_train + t_eval, 3),
            'run_dir': rdir,
        }

    # Print comparison
    print("\nModel comparison (val):")
    header = f"{'Model':<15} {'MSE':>12} {'RMSE':>12} {'PSNR(dB)':>12} {'SSIM':>12} {'MS-SSIM':>12} {'Train(s)':>10} {'Eval(s)':>10} {'Total(s)':>10}"
    print(header)
    print('-' * len(header))
    for m, met in summary.items():
        print(f"{m:<15} {met['MSE']:>12.6f} {met['RMSE']:>12.6f} {met['PSNR']:>12.4f} {met['SSIM']:>12.4f} {met['MS-SSIM']:>12.4f} {met['train_seconds']:>10.1f} {met['eval_seconds']:>10.1f} {met['total_seconds']:>10.1f}")

    # Save CSV/JSON
    os.makedirs('runs', exist_ok=True)
    summary_package = {
        'prepared_seconds': round(t_prepare1 - t_prepare0, 3),
        'models': summary,
    }
    with open('runs/summary.json', 'w') as f:
        json.dump(summary_package, f, indent=2)
    with open('runs/summary.csv', 'w') as f:
        f.write('model,MSE,RMSE,PSNR,SSIM,MS-SSIM,train_seconds,eval_seconds,total_seconds\n')
        for m, met in summary.items():
            f.write(f"{m},{met['MSE']:.6f},{met['RMSE']:.6f},{met['PSNR']:.6f},{met['SSIM']:.6f},{met['MS-SSIM']:.6f},{met['train_seconds']:.3f},{met['eval_seconds']:.3f},{met['total_seconds']:.3f}\n")
    print("\nSaved summary with timings: runs/summary.json and runs/summary.csv")


if __name__ == '__main__':
    main()
