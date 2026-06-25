#!/bin/bash
# ============================================================================
# Submit job(s) to evaluate all combinations from two checkpoint lists.
# Each checkpoint from list 1 is evaluated against each checkpoint from list 2.
#
# Usage:
#   1) Single job with explicit ITERATIONS array (default):
#      ./submit_evaluate_checkpoints_matrix.sh
#
#   2) N jobs, one per iteration (e.g. 0, 5, 10, ..., 145):
#      ./submit_evaluate_checkpoints_matrix.sh --iter-range 0 145 5
# ============================================================================

# Evaluation parameters
N_HANDS=2000000          # Number of hands for h2h evaluation

# Optional: Override data path (default: ~/poker_ai_data)
# DATA_PATH=""

# ============================================================================
# Define iterations to evaluate (used when --iter-range is not passed)
# Can be a single value or an array
# ============================================================================

ITERATIONS=(149)

# ============================================================================
# Define two lists of experiments to evaluate (without iterations)
# Format: "EXPERIMENT"
# ============================================================================

CHECKPOINTS_LIST1=(
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run11"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run12"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run13"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run14"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run15"
)

CHECKPOINTS_LIST2=(
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_Large_run10"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_Large_run11"
    "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_Large_run12"
)

# ============================================================================
# Optional: iteration range (submit one job per iteration)
# Usage: --iter-range START END STEP   e.g. --iter-range 0 145 5  →  0,5,10,...,145
# ============================================================================

ITER_RANGE_START=""
ITER_RANGE_END=""
ITER_RANGE_STEP=""
if [ "$1" = "--iter-range" ] && [ $# -ge 4 ]; then
    ITER_RANGE_START="$2"
    ITER_RANGE_END="$3"
    ITER_RANGE_STEP="$4"
    shift 4
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_SCRIPT="$SCRIPT_DIR/evaluate_checkpoint_matrix.sh"

if [ ! -f "$MATRIX_SCRIPT" ]; then
    echo "Error: $MATRIX_SCRIPT not found!"
    exit 1
fi

# Build base args (list1, list2) once
BASE_ARGS=()
for checkpoint in "${CHECKPOINTS_LIST1[@]}"; do
    BASE_ARGS+=("--list1" "$checkpoint")
done
for checkpoint in "${CHECKPOINTS_LIST2[@]}"; do
    BASE_ARGS+=("--list2" "$checkpoint")
done

submit_one_job() {
    local iters=("$@")
    local iter_args=()
    for i in "${iters[@]}"; do
        iter_args+=("--iteration" "$i")
    done
    if [ -n "$DATA_PATH" ]; then
        sbatch \
            --job-name="h2h-matrix" \
            --output="slurm_out/h2h-matrix-%j.out" \
            --error="slurm_out/h2h-matrix-%j.out" \
            "$MATRIX_SCRIPT" "${BASE_ARGS[@]}" "${iter_args[@]}" --n-hands "$N_HANDS" --data-path "$DATA_PATH"
    else
        sbatch \
            --job-name="h2h-matrix" \
            --output="slurm_out/h2h-matrix-%j.out" \
            --error="slurm_out/h2h-matrix-%j.out" \
            "$MATRIX_SCRIPT" "${BASE_ARGS[@]}" "${iter_args[@]}" --n-hands "$N_HANDS"
    fi
}

if [ -n "$ITER_RANGE_START" ] && [ -n "$ITER_RANGE_END" ] && [ -n "$ITER_RANGE_STEP" ]; then
    # Submit one job per iteration: 0, 5, 10, ..., 145
    COUNT=0
    for iter in $(seq "$ITER_RANGE_START" "$ITER_RANGE_STEP" "$ITER_RANGE_END"); do
        SUBMITTED=$(submit_one_job "$iter" | tee /dev/stderr | awk '{print $4}')
        [ -n "$SUBMITTED" ] && echo "  iteration $iter → job $SUBMITTED" && COUNT=$((COUNT + 1))
    done
    echo "Submitted $COUNT jobs (iterations $ITER_RANGE_START to $ITER_RANGE_END step $ITER_RANGE_STEP)"
else
    # Single job with ITERATIONS array
    TOTAL_COMBINATIONS=$((${#CHECKPOINTS_LIST1[@]} * ${#CHECKPOINTS_LIST2[@]} * ${#ITERATIONS[@]}))
    echo "Submitting single job for $TOTAL_COMBINATIONS H2H evaluations (${#CHECKPOINTS_LIST1[@]} × ${#CHECKPOINTS_LIST2[@]} × ${#ITERATIONS[@]} iterations)..."

    OUTPUT=$(submit_one_job "${ITERATIONS[@]}")
    JOB_ID=$(echo "$OUTPUT" | awk '{print $4}')
    echo "Submitted job $JOB_ID: Matrix evaluation ($TOTAL_COMBINATIONS combinations)"
    echo "Output will be written to: slurm_out/h2h-matrix-${JOB_ID}.out"
fi

