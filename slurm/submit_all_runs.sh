#!/bin/bash
# Master script to submit 5 runs of LightGBM and 5 runs of NN SingleDeepCFR
# Each run will have a unique run-id (0-4) to avoid checkpoint overwrites

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Submitting 10 training runs (5 LightGBM GPU + 5 NN)..."
echo "Project directory: $PROJECT_DIR"
echo ""

# Submit 5 LightGBM GPU runs (run-ids 0-4)
echo "Submitting 5 LightGBM CPU runs..."
for i in {0..4}; do
    echo "  Submitting LightGBM CPU run $i..."
    cd "$PROJECT_DIR"
    RUN_ID=$i sbatch --export=ALL,RUN_ID=$i slurm/submit_flop5_lightgbm_cpu.sh $i
    sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo ""
echo "Submitting 5 NN (Neural Network) runs..."
# Submit 5 NN runs (run-ids 0-4)
for i in {10..14}; do
    echo "  Submitting NN run $i..."
    cd "$PROJECT_DIR"
    RUN_ID=$i sbatch --export=ALL,RUN_ID=$i slurm/submit_flop5_nn_large.sh $i
    sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo ""
echo "All 10 jobs submitted!"
echo ""
echo "Check job status with: squeue -u \$USER"
echo ""
echo "Run names will be:"
echo "  LightGBM GPU: EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run{0-4}"
echo "  NN:           EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_run{0-4}"

