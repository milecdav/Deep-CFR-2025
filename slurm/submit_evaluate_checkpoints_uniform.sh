#!/bin/bash
# ============================================================================
# Submit multiple vs_uniform checkpoint evaluation jobs
# ============================================================================

# Evaluation parameters
N_HANDS=200000          # Number of hands for vs_uniform evaluation

# Optional: Override data path (default: ~/poker_ai_data)
# DATA_PATH=""

# ============================================================================
# Define checkpoints to evaluate
# Format: "EXPERIMENT:ITERATION"
# ============================================================================

CHECKPOINTS=(
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:15"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:30"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:45"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:60"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:75"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:90"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:105"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:120"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:135"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0:150"
)

# ============================================================================
# Submit jobs for each checkpoint
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIFORM_SCRIPT="$SCRIPT_DIR/evaluate_checkpoint_uniform.sh"

if [ ! -f "$UNIFORM_SCRIPT" ]; then
    echo "Error: $UNIFORM_SCRIPT not found!"
    exit 1
fi

echo "Submitting ${#CHECKPOINTS[@]} vs_uniform evaluation jobs..."

for checkpoint in "${CHECKPOINTS[@]}"; do
    # Parse checkpoint: "EXPERIMENT:ITERATION"
    IFS=':' read -r EXPERIMENT ITERATION <<< "$checkpoint"
    DESCRIPTION="vs_uniform: $EXPERIMENT (iter $ITERATION)"
    
    # Submit job
    if [ -n "$DATA_PATH" ]; then
        JOB_ID=$(sbatch "$UNIFORM_SCRIPT" "$EXPERIMENT" "$ITERATION" "$N_HANDS" "$DATA_PATH" | awk '{print $4}')
    else
        JOB_ID=$(sbatch "$UNIFORM_SCRIPT" "$EXPERIMENT" "$ITERATION" "$N_HANDS" | awk '{print $4}')
    fi
    
    echo "Submitted job $JOB_ID: $DESCRIPTION"
done

echo "All jobs submitted!"

