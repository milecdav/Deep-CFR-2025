#!/bin/bash
#SBATCH --job-name=deep-cfr-evaluate-checkpoints-leduc
#SBATCH --output=slurm_out/deep-cfr-evaluate-checkpoints-leduc-%j.out
#SBATCH --error=slurm_out/deep-cfr-evaluate-checkpoints-leduc-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --partition=cpufast

# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================

# Evaluation mode: "vs_uniform", "lbr", or "h2h"
MODE="h2h"

# For single agent evaluation (vs_uniform or lbr):
EXPERIMENT="EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run5"
ITERATION=30

# For head-to-head evaluation (h2h):
EXPERIMENT1="LEDUC_EXPLOITABILITY_LightGBM"
EXPERIMENT2="LEDUC_EXPLOITABILITY_NN"
ITERATION1=225
ITERATION2=225

# Evaluation parameters
N_HANDS=20000          # Number of hands for vs_uniform or h2h
N_LBR_HANDS=30000       # Number of LBR hands per seat (for lbr mode)

# Optional: Override data path (default: ~/poker_ai_data)
# DATA_PATH=""

# ============================================================================
# Script execution (usually no need to modify below)
# ============================================================================

ml Python/3.11.5-GCCcore-13.2.0 
source venv/bin/activate 

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_WORKERS=$((${SLURM_CPUS_PER_TASK:-16} - 2))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

# Build command based on mode
CMD="python -u evaluate_checkpoint.py --n-workers $N_WORKERS --mode $MODE"

if [ "$MODE" = "h2h" ]; then
    CMD="$CMD --experiment1 $EXPERIMENT1 --experiment2 $EXPERIMENT2 --iteration1 $ITERATION1 --iteration2 $ITERATION2 --n-hands $N_HANDS"
else
    CMD="$CMD --experiment $EXPERIMENT --iteration $ITERATION"
    if [ "$MODE" = "vs_uniform" ]; then
        CMD="$CMD --n-hands $N_HANDS"
    elif [ "$MODE" = "lbr" ]; then
        CMD="$CMD --n-lbr-hands $N_LBR_HANDS"
    fi
fi

# Add data path if specified
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data-path $DATA_PATH"
fi

# Execute command
echo "Running: $CMD"
$CMD