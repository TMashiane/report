#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --job-name=finisher
#SBATCH --output=/home-mscluster/tmashiane/finisher_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finisher

# Set headless mode for cluster (no display)
export DISPLAY=

# Enable subprocess vectorization for cluster (Linux)
export USE_SUBPROCESS=1

# Verify environment
echo "========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU Check:"
nvidia-smi 2>/dev/null || echo "No GPU found (will use CPU)"
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch check failed"
echo "Environment variables:"
echo "USE_SUBPROCESS: $USE_SUBPROCESS"
echo "========================================="
echo ""
python scripts\run_all.py


