#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --job-name=finisherres
#SBATCH --output=/home-mscluster/tmashiane/finisherres_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finisher

# Train ResAttUNet3D
python scripts/train.py \
    --model resattunet3d \
    --data-dir data \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-3 \
    --patch-size 64 64 64