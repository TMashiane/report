#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --job-name=finishereval
#SBATCH --output=/home-mscluster/tmashiane/finishereval_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finisher

# Function to find latest checkpoint
find_latest_ckpt() {
    model=$1
    find runs/$model -name "best.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

# Evaluate all models
for model in unet3d resattunet3d vit3d; do
    echo "== Evaluate $model =="
    ckpt=$(find_latest_ckpt $model)
    if [ -n "$ckpt" ]; then
        python scripts/evaluate.py \
            --model $model \
            --data-dir data \
            --ckpt "$ckpt" \
            --batch-size 32 \
            --patch-size 64 64 64 \
            --save-samples runs/eval_samples/$model \
            --out-json runs/$model/metrics.json
    else
        echo "No checkpoint found for $model"
    fi
done

echo "Evaluation complete. Check runs/*/metrics.json for results."