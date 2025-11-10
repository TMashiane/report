#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --job-name=finisherdata_prepared
#SBATCH --output=/home-mscluster/tmashiane/finisherdata_prepared_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finisher

INPUT_PATH="src/data/Orange Basin_cropped 1"
OUTDIR="data"
PATCH_SIZE="64 64 64"
TRAIN_PATCHES=2000
VAL_PATCHES=400
SEED=42

if [ -f "$OUTDIR/augmented/train_patches.npz" ] && [ -f "$OUTDIR/augmented/val_patches.npz" ]; then
    echo "Data already prepared; skipping prepare step."
    exit 0
fi

python scripts/prepare_data.py --input "$INPUT_PATH" --outdir "$OUTDIR" --patch-size $PATCH_SIZE --train-patches $TRAIN_PATCHES --val-patches $VAL_PATCHES --seed $SEED