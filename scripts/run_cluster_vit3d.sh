#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --job-name=finishervit
#SBATCH --output=/home-mscluster/tmashiane/finishervit_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finisher

# Train ViT3D (with reduced batch size and learning rate as per run_all.py)
python scripts/train.py \
    --model vit3d \
    --data-dir data \
    --batch-size 2 \
    --epochs 50 \
    --lr 1e-4 \
    --patch-size 64 64 64