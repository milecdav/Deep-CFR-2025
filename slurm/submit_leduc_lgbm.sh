#!/bin/bash
#SBATCH --job-name=leduc-lgbm
#SBATCH --output=slurm_out/leduc-lgbm-%j.out
#SBATCH --error=slurm_out/leduc-lgbm-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=amd

# Adjust module load for your cluster's Python version
ml Python/3.11.5-GCCcore-13.2.0

source venv/bin/activate

# Belt-and-suspenders: cap BLAS threads before Python starts.
# Worker subprocesses also call torch.set_num_threads(1) themselves.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_WORKERS=$((${SLURM_CPUS_PER_TASK:-16} - 2))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

# Optional: Set number of iterations (default: None = run until stopped)
# N_ITERATIONS=50

# Build command
CMD="python -u paper_experiment_leduc_exploitability_comparison.py --run-lgbm --n-workers $N_WORKERS --device-training cpu --device-parameter-server cpu --device-inference cpu"

# Add iterations if specified
if [ -n "$N_ITERATIONS" ]; then
    CMD="$CMD --n-iterations $N_ITERATIONS"
fi

echo "Running: $CMD"
$CMD

