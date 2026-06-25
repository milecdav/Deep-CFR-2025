#!/bin/bash
#SBATCH --job-name=deep-cfr-eval-matrix
#SBATCH --output=slurm_out/deep-cfr-evaluate-matrix-%j.out
#SBATCH --error=slurm_out/deep-cfr-evaluate-matrix-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=amd

# ============================================================================
# This script evaluates all combinations from two checkpoint lists
# Usage: sbatch evaluate_checkpoint_matrix.sh --list1 EXP1 --list1 EXP2 ... --list2 EXP3 ... --iteration ITER [--n-hands N] [--data-path PATH]
# ============================================================================

# Parse arguments
CHECKPOINTS_LIST1=()
CHECKPOINTS_LIST2=()
ITERATIONS=()
N_HANDS=2000000
DATA_PATH=""

CURRENT_LIST=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --list1)
            CURRENT_LIST="list1"
            shift
            ;;
        --list2)
            CURRENT_LIST="list2"
            shift
            ;;
        --iteration)
            ITERATIONS+=("$2")
            shift 2
            ;;
        --n-hands)
            N_HANDS="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        *)
            if [ "$CURRENT_LIST" = "list1" ]; then
                CHECKPOINTS_LIST1+=("$1")
            elif [ "$CURRENT_LIST" = "list2" ]; then
                CHECKPOINTS_LIST2+=("$1")
            else
                echo "Error: Unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ ${#CHECKPOINTS_LIST1[@]} -eq 0 ] || [ ${#CHECKPOINTS_LIST2[@]} -eq 0 ]; then
    echo "Error: Both --list1 and --list2 must be provided with at least one experiment each"
    exit 1
fi

if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "Error: At least one --iteration must be provided (use 0 for initialized/untrained models)"
    exit 1
fi

ml Python/3.11.5-GCCcore-13.2.0
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_WORKERS=$((${SLURM_CPUS_PER_TASK:-16} - 2))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

TOTAL_COMBINATIONS=$((${#CHECKPOINTS_LIST1[@]} * ${#CHECKPOINTS_LIST2[@]} * ${#ITERATIONS[@]}))

echo "============================================================================"
echo "Matrix H2H Evaluation"
echo "============================================================================"
echo "Iterations: ${ITERATIONS[*]}"
echo ""
echo "List 1 (${#CHECKPOINTS_LIST1[@]} experiments):"
for checkpoint in "${CHECKPOINTS_LIST1[@]}"; do
    echo "  - $checkpoint"
done
echo ""
echo "List 2 (${#CHECKPOINTS_LIST2[@]} experiments):"
for checkpoint in "${CHECKPOINTS_LIST2[@]}"; do
    echo "  - $checkpoint"
done
echo ""
echo "Total combinations: $TOTAL_COMBINATIONS (${#CHECKPOINTS_LIST1[@]} × ${#CHECKPOINTS_LIST2[@]} × ${#ITERATIONS[@]})"
echo "Hands per evaluation: $N_HANDS"
echo "Workers: $N_WORKERS"
echo "============================================================================"
echo ""

# Build base command
BASE_CMD="python -u evaluate_checkpoint.py --n-workers $N_WORKERS --mode h2h --n-hands $N_HANDS"
if [ -n "$DATA_PATH" ]; then
    BASE_CMD="$BASE_CMD --data-path $DATA_PATH"
fi

# Track results
COUNTER=0
RESULTS_FILE="/tmp/h2h_matrix_results_$$.txt"
SUMMARY_FILE="/tmp/h2h_matrix_summary_$$.txt"
> "$RESULTS_FILE"
> "$SUMMARY_FILE"

# Evaluate all combinations
for iteration in "${ITERATIONS[@]}"; do
    for checkpoint1 in "${CHECKPOINTS_LIST1[@]}"; do
        EXPERIMENT1="$checkpoint1"
        
        for checkpoint2 in "${CHECKPOINTS_LIST2[@]}"; do
            EXPERIMENT2="$checkpoint2"
            
            COUNTER=$((COUNTER + 1))
            
            echo ""
            echo "============================================================================"
            echo "[$COUNTER/$TOTAL_COMBINATIONS] $EXPERIMENT1 (iter $iteration) vs $EXPERIMENT2 (iter $iteration)"
            echo "============================================================================"
            echo ""
            
            CMD="$BASE_CMD --experiment1 $EXPERIMENT1 --iteration1 $iteration --experiment2 $EXPERIMENT2 --iteration2 $iteration"
            
            # Run evaluation and capture output
            OUTPUT=$(eval "$CMD" 2>&1)
            EXIT_CODE=$?
            
            # Save full output
            echo "$OUTPUT" >> "$RESULTS_FILE"
            echo "$OUTPUT"
            
            # Extract key metrics for summary (format: "Agent 1 (...): X.XXXXXX MBB_per_G" and "95% CI: [LOWER, UPPER]")
            MEAN_LINE=$(echo "$OUTPUT" | grep "Agent 1" | head -1)
            CI_LINE=$(echo "$OUTPUT" | grep "95% CI" | head -1)
            
            if [ -n "$MEAN_LINE" ]; then
                MEAN=$(echo "$MEAN_LINE" | sed -n 's/.*: \(-*[0-9][0-9.]*\) MBB_per_G.*/\1/p' | head -1)
            else
                MEAN="N/A"
            fi
            
            if [ -n "$CI_LINE" ]; then
                CI_VALUES=$(echo "$CI_LINE" | sed -n 's/.*\[\(-*[0-9][0-9.]*\), \(-*[0-9][0-9.]*\)\].*/\1 \2/p' | head -1)
                LOWER=$(echo "$CI_VALUES" | awk '{print $1}')
                UPPER=$(echo "$CI_VALUES" | awk '{print $2}')
            else
                LOWER="N/A"
                UPPER="N/A"
            fi
            
            if [ -z "$MEAN" ]; then MEAN="N/A"; fi
            if [ -z "$LOWER" ]; then LOWER="N/A"; fi
            if [ -z "$UPPER" ]; then UPPER="N/A"; fi
            
            if [ "$EXIT_CODE" -eq 0 ]; then
                STATUS="✓"
            else
                STATUS="✗"
            fi
            
            # Add to summary
            printf "%-5s | %-50s | %-50s | Mean: %10s [%10s, %10s]\n" \
                "$STATUS" \
                "$EXPERIMENT1 (iter $iteration)" \
                "$EXPERIMENT2 (iter $iteration)" \
                "$MEAN" "$LOWER" "$UPPER" >> "$SUMMARY_FILE"
            
            echo ""
            if [ "$EXIT_CODE" -eq 0 ]; then
                echo "✓ Completed successfully"
            else
                echo "✗ Failed with exit code $EXIT_CODE"
            fi
            echo ""
        done
    done
done

echo ""
echo "============================================================================"
echo "All evaluations complete!"
echo "============================================================================"
echo ""
echo "Summary Table:"
echo "============================================================================"
printf "%-5s | %-50s | %-50s | %s\n" "STATUS" "EXPERIMENT1" "EXPERIMENT2" "RESULT (Mean [Lower, Upper])"
echo "============================================================================"
cat "$SUMMARY_FILE"
echo "============================================================================"

rm -f "$RESULTS_FILE" "$SUMMARY_FILE"

