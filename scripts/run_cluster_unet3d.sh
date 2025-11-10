#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --job-name=finisherunet
#SBATCH --output=/home-mscluster/tmashiane/finisherunet_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finisher

# Train UNet3D
python scripts/train.py \
    --model unet3d \
    --data-dir data \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --patch-size 64 64 64