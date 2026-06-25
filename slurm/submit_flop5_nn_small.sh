#!/bin/bash
#SBATCH --job-name=deep-cfr-flop5-nn-small
#SBATCH --output=slurm_out/deep-cfr-flop5-nn-small-%j.out
#SBATCH --error=slurm_out/deep-cfr-flop5-nn-small-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:1        # recommended for Flop5Holdem training
#SBATCH --time=72:00:00
#SBATCH --partition=amdgpulong

# Adjust module load for your cluster's Python version
ml Python/3.11.5-GCCcore-13.2.0

source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_WORKERS=$((${SLURM_CPUS_PER_TASK:-16} - 2))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

# Accept run-id from environment variable or command line argument
RUN_ID=${RUN_ID:-${1:-}}

if [ -n "$RUN_ID" ]; then
    python -u paper_experiment_sdcfr_vs_deepcfr_h2h.py --n-workers "$N_WORKERS" --device-training cpu --device-parameter-server cuda:0 --device-inference cpu --adv-model-type nn --nn-size small --run-id "$RUN_ID"
else
    python -u paper_experiment_sdcfr_vs_deepcfr_h2h.py --n-workers "$N_WORKERS" --device-training cpu --device-parameter-server cuda:0 --device-inference cpu --adv-model-type nn --nn-size small
fi
