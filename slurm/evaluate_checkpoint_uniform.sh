#!/bin/bash
#SBATCH --job-name=deep-cfr-eval-uniform-%j
#SBATCH --output=slurm_out/deep-cfr-evaluate-checkpoints-uniform-%j.out
#SBATCH --error=slurm_out/deep-cfr-evaluate-checkpoints-uniform-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --partition=cpufast

# ============================================================================
# This script evaluates a single checkpoint against uniform random
# Usage: sbatch evaluate_checkpoint_uniform.sh EXPERIMENT ITERATION [N_HANDS] [DATA_PATH]
# ============================================================================

EXPERIMENT=$1
ITERATION=$2
N_HANDS=${3:-200000}
DATA_PATH=${4:-""}

if [ -z "$EXPERIMENT" ] || [ -z "$ITERATION" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch evaluate_checkpoint_uniform.sh EXPERIMENT ITERATION [N_HANDS] [DATA_PATH]"
    exit 1
fi

ml Python/3.11.5-GCCcore-13.2.0 
source venv/bin/activate 

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_WORKERS=$((${SLURM_CPUS_PER_TASK:-16} - 2))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

CMD="python -u evaluate_checkpoint.py --n-workers $N_WORKERS --mode vs_uniform --experiment $EXPERIMENT --iteration $ITERATION --n-hands $N_HANDS"

if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data-path $DATA_PATH"
fi

echo "Running: $CMD"
echo "Job ID: $SLURM_JOB_ID"
echo "Evaluating vs_uniform: $EXPERIMENT (iter $ITERATION)"
$CMD

