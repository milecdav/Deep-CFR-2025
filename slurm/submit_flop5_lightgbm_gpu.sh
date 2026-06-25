#!/bin/bash
#SBATCH --job-name=deep-cfr-flop5-lgbm-gpu
#SBATCH --output=slurm_out/deep-cfr-flop5-lgbm-gpu-%j.out
#SBATCH --error=slurm_out/deep-cfr-flop5-lgbm-gpu-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:1        # GPU required for LightGBM GPU training
#SBATCH --time=72:00:00
#SBATCH --partition=gpulong

# Adjust module load for your cluster's Python version
ml Python/3.11.5-GCCcore-13.2.0
# Load CUDA module for OpenCL libraries (required for LightGBM GPU)
ml CUDA/13.0.2 2>/dev/null || ml CUDA/12.4.0 2>/dev/null || ml CUDA 2>/dev/null || echo "Warning: Could not load CUDA module"

source venv/bin/activate

# Set library path to find OpenCL (CUDA provides OpenCL libraries)
if [ -n "$CUDA_HOME" ]; then
    export LD_LIBRARY_PATH=${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
elif [ -d "/mnt/appl/software/CUDA/13.0.2" ]; then
    export LD_LIBRARY_PATH=/mnt/appl/software/CUDA/13.0.2/targets/x86_64-linux/lib:/mnt/appl/software/CUDA/13.0.2/lib64:${LD_LIBRARY_PATH}
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Filter out OpenCL compiler "warning generated" messages (they're harmless)
# Redirect stderr through a filter that removes these specific warnings
exec 2> >(grep -v "^1 warning generated\.$" >&2 || cat)

N_WORKERS=$((${SLURM_CPUS_PER_TASK:-16} - 2))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

# Accept run-id from environment variable or command line argument
RUN_ID=${RUN_ID:-${1:-}}

if [ -n "$RUN_ID" ]; then
    python -u paper_experiment_sdcfr_vs_deepcfr_h2h.py --n-workers "$N_WORKERS" --device-training cpu --device-parameter-server cpu --device-inference cpu --adv-model-type lightgbm --adv-lgbm-device-type gpu --run-id "$RUN_ID"
else
    python -u paper_experiment_sdcfr_vs_deepcfr_h2h.py --n-workers "$N_WORKERS" --device-training cpu --device-parameter-server cpu --device-inference cpu --adv-model-type lightgbm --adv-lgbm-device-type gpu
fi

