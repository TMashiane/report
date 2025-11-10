Orange Basin 3D Seismic Reconstruction (UNet3D, ResAttUNet3D, ViT3D)

Goal: Compare how well three models reconstruct augmented seismic volumes back to the original Orange Basin 3D dataset using PSNR, MSE, RMSE, SSIM, and MS-SSIM.

Models:
- UNet3D (baseline)
- ResAttUNet3D (Residual + Channel/Spatial Attention)
- ViT3D (Transformer-based patch autoencoder)

Pipeline:
1) Split dataset and persist split
2) Augment training data (noise + blur) and save augmented patches
3) Train selected model on augmented patches
4) Evaluate metrics on val/test
5) Save sample images comparing augmented vs original and model outputs

Dataset:
- Orange Basin provided as SEG/SEGY file (or .npy). Target volume dims ~ 560 x 773 x 805 (inline x crossline x time).

Directory layout:

- data/
  - raw/                 # raw volume saved as .npy
  - splits/              # train/val/test split JSON
  - augmented/           # saved augmented patch datasets (npz)
  - samples/             # a few saved images (aug vs orig)
- runs/
  - <model>/<timestamp>/ # checkpoints and logs
- src/
  - data/
    - dataset.py         # loaders, patch extraction, augmentation
    - Orange Basin_cropped 1 # the dataset (Place it here)
  - metrics/
    - metrics.py         # MSE, RMSE, PSNR, SSIM3D, MS-SSIM3D
  - models/
    - unet3d.py
    - res_attention_unet3d.py
    - vit3d.py
  - utils/
    - seg_loader.py      # optional SEG-Y loading via segyio
    - train_utils.py
- scripts/
  - prepare_data.py      # load > split > augment > save
  - train.py             # train one model
  - evaluate.py          # evaluate metrics + save predictions
- requirements.txt

Dependencies (install with pip):
- torch, torchvision, torchaudio (match your CUDA)
- numpy, scipy, matplotlib, tqdm
- segyio (optional; for direct SEG/SEGY reading)

Example installs (CPU only):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install numpy scipy matplotlib tqdm segyio

Quick start:
1) Prepare data (from SEG/SEGY or pre-saved .npy):
   - If SEG/SEGY: scripts/prepare_data.py --input path/to/data.segy --format segy
   - If NumPy:    scripts/prepare_data.py --input path/to/volume.npy --format numpy

   Common args:
   - --outdir data
   - --patch-size 64 64 64
   - --train-patches 2000 --val-patches 400

   Example:
   - python scripts/prepare_data.py --input C:/data/orange_basin.segy --format segy --outdir data --patch-size 64 64 64 --train-patches 2000 --val-patches 400
   
   - Or if one wants to run on the Wits Cluster run "run_cluster_prepare_data.sh"

2) Train a model:
   - python scripts/train.py --model unet3d --data-dir data --batch-size 32 --epochs 50 --lr 1e-3 --patch-size 64 64 64
   - python scripts/train.py --model resattunet3d --data-dir data --batch-size 4 --epochs 50 --lr 1e-3 --patch-size 64 64 64
   - python scripts/train.py --model vit3d --data-dir data --batch-size 2 --epochs 50 --lr 1e-4 --patch-size 64 64 64

   - Or if one wants to run on the Wits Cluster run:
     - "run_cluster_unet3d.sh" for running the UNet3D
     - "run_cluster_resattunet3d.sh" for running the ResAttUNet3D
     - "run_cluster_unet3d.sh" for running the ViT3D

3) Evaluate:
   - python scripts/evaluate.py --model unet3d --data-dir data --ckpt runs/unet3d/<timestamp>/best.pt

   - Or if one wants to run on the Wits Cluster run "run_cluster_evaluate.sh"

Run all models (highly not advised):
   - End-to-end (prepare, train all, evaluate, summarize):
     python scripts/run_all.py --input "src/data/Orange Basin_cropped 1" --format auto --outdir data --patch-size 64 64 64 --train-patches 2000 --val-patches 400 --epochs 50 --batch-size 32 
     - Uses smaller batch and LR for ViT3D automatically.
     - Writes metrics to runs/<model>/metrics.json and summary to runs/summary.(json|csv) with per-model timing (train/eval/total) and prepare time.

   - Or if one wants to run on the Wits Cluster run "run_cluster.sh"

Notes:
- Volumes are normalized to [-1,1] using min/max before patching.
- Augmentation includes Gaussian noise, speckle noise, and 3D Gaussian blur.
- SSIM/MS-SSIM are computed in 3D with a Gaussian window. For very large volumes, patch-wise evaluation is used.
- ViT3D uses non-overlapping 3D patches; its batch size may need to be small.

Troubleshooting:
- If segyio is not installed, convert your SEG/SEGY to .npy first or install segyio.
- If out-of-memory, reduce --patch-size and/or --batch-size.
- On Windows paths, quote paths with spaces.
