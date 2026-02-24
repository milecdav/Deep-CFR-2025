#!/bin/bash
#SBATCH --job-name=deep-cfr
#SBATCH --output=slurm_out/deep-cfr-%j.out
#SBATCH --error=slurm_out/deep-cfr-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1        # optional — remove if no GPU needed
#SBATCH --time=48:00:00

# Adjust module load for your cluster's Python version
ml Python/3.11.5-GCCcore-13.2.0

source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Use one fewer worker than CPUs to leave headroom for the main process
N_WORKERS=$((${SLURM_CPUS_PER_TASK:-8} - 1))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

python -u leduc_example.py --n-workers "$N_WORKERS"

