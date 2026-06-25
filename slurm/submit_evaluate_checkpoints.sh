#!/bin/bash
# ============================================================================
# Submit multiple h2h checkpoint evaluation jobs
# ============================================================================

# Evaluation parameters
N_HANDS=200000          # Number of hands for h2h evaluation

# Optional: Override data path (default: ~/poker_ai_data)
# DATA_PATH=""

# ============================================================================
# Define iterations to evaluate (same for all pairs)
# Can be a single value or an array
# ============================================================================

ITERATIONS=(15)

# ============================================================================
# Define experiment pairs to evaluate (without iterations)
# Format: "EXPERIMENT1:EXPERIMENT2"
# ============================================================================

CHECKPOINT_PAIRS=(
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run7:EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_Medium_run15"
    # "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run1:EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_run1"
    # "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run2:EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_run2"
    # "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run3:EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_run3"
    # "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run4:EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_run4"
)

# ============================================================================
# Submit jobs for each checkpoint pair and iteration combination
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIR_SCRIPT="$SCRIPT_DIR/evaluate_checkpoint_pair.sh"

if [ ! -f "$PAIR_SCRIPT" ]; then
    echo "Error: $PAIR_SCRIPT not found!"
    exit 1
fi

TOTAL_JOBS=$((${#CHECKPOINT_PAIRS[@]} * ${#ITERATIONS[@]}))
echo "Submitting $TOTAL_JOBS h2h evaluation jobs (${#CHECKPOINT_PAIRS[@]} pairs × ${#ITERATIONS[@]} iterations)..."

for iteration in "${ITERATIONS[@]}"; do
    for pair in "${CHECKPOINT_PAIRS[@]}"; do
        # Parse checkpoint pair: "EXPERIMENT1:EXPERIMENT2"
        IFS=':' read -r EXPERIMENT1 EXPERIMENT2 <<< "$pair"
        DESCRIPTION="H2H: $EXPERIMENT1 (iter $iteration) vs $EXPERIMENT2 (iter $iteration)"
        
        # Submit job
        if [ -n "$DATA_PATH" ]; then
            JOB_ID=$(sbatch "$PAIR_SCRIPT" "$EXPERIMENT1" "$iteration" "$EXPERIMENT2" "$iteration" "$N_HANDS" "$DATA_PATH" | awk '{print $4}')
        else
            JOB_ID=$(sbatch "$PAIR_SCRIPT" "$EXPERIMENT1" "$iteration" "$EXPERIMENT2" "$iteration" "$N_HANDS" | awk '{print $4}')
        fi
        
        echo "Submitted job $JOB_ID: $DESCRIPTION"
    done
done

echo "All jobs submitted!"
