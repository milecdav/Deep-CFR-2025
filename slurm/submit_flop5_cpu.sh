#!/bin/bash
#SBATCH --job-name=deep-cfr-flop5-cpu
#SBATCH --output=slurm_out/deep-cfr-flop5-cpu-%j.out
#SBATCH --error=slurm_out/deep-cfr-flop5-cpu-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=amdlong

# Adjust module load for your cluster's Python version
ml Python/3.11.5-GCCcore-13.2.0

source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_WORKERS=$((${SLURM_CPUS_PER_TASK:-16} - 2))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

python -u paper_experiment_sdcfr_vs_deepcfr_h2h.py --n-workers "$N_WORKERS" --device-training cpu --device-parameter-server cpu --device-inference cpu
