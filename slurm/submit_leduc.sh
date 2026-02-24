#!/bin/bash
#SBATCH --job-name=deep-cfr-leduc
#SBATCH --output=slurm_out/deep-cfr-leduc-%j.out
#SBATCH --error=slurm_out/deep-cfr-leduc-%j.err
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

N_WORKERS=$((${SLURM_CPUS_PER_TASK:-4} - 1))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

python -u paper_experiment_leduc_exploitability.py --n-workers "$N_WORKERS" --device-training cpu --device-parameter-server cpu --device-inference cpu
