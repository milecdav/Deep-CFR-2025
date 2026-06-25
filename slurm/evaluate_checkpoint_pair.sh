#!/bin/bash
#SBATCH --job-name=deep-cfr-eval-%j
#SBATCH --output=slurm_out/deep-cfr-evaluate-checkpoints-%j.out
#SBATCH --error=slurm_out/deep-cfr-evaluate-checkpoints-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --partition=amdfast

# ============================================================================
# This script evaluates a single h2h checkpoint pair
# Usage: sbatch evaluate_checkpoint_pair.sh EXPERIMENT1 ITERATION1 EXPERIMENT2 ITERATION2 [N_HANDS] [DATA_PATH]
# ============================================================================

EXPERIMENT1=$1
ITERATION1=$2
EXPERIMENT2=$3
ITERATION2=$4
N_HANDS=${5:-200000}
DATA_PATH=${6:-""}

if [ -z "$EXPERIMENT1" ] || [ -z "$ITERATION1" ] || [ -z "$EXPERIMENT2" ] || [ -z "$ITERATION2" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch evaluate_checkpoint_pair.sh EXPERIMENT1 ITERATION1 EXPERIMENT2 ITERATION2 [N_HANDS] [DATA_PATH]"
    exit 1
fi

ml Python/3.11.5-GCCcore-13.2.0 
source venv/bin/activate 

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_WORKERS=$((${SLURM_CPUS_PER_TASK:-16} - 2))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

CMD="python -u evaluate_checkpoint.py --n-workers $N_WORKERS --mode h2h --experiment1 $EXPERIMENT1 --experiment2 $EXPERIMENT2 --iteration1 $ITERATION1 --iteration2 $ITERATION2 --n-hands $N_HANDS"

if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data-path $DATA_PATH"
fi

echo "Running: $CMD"
echo "Job ID: $SLURM_JOB_ID"
echo "Evaluating H2H: $EXPERIMENT1 (iter $ITERATION1) vs $EXPERIMENT2 (iter $ITERATION2)"
$CMD

