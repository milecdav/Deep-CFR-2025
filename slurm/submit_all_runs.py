#!/usr/bin/env python3
"""
Python script to submit 5 runs of LightGBM and 5 runs of NN SingleDeepCFR.
Each run will have a unique run-id (0-4) to avoid checkpoint overwrites.
"""

import subprocess
import sys
import os
from pathlib import Path

def submit_slurm_job(script_path, run_id, job_type="NN"):
    """Submit a SLURM job with the given run_id."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Change to project directory
    os.chdir(project_dir)
    
    # Build sbatch command
    cmd = [
        "sbatch",
        "--export=ALL",
        f"RUN_ID={run_id}",
        str(script_path),
        str(run_id)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        job_id = result.stdout.strip().split()[-1] if result.stdout else "unknown"
        print(f"  ✓ Submitted {job_type} run {run_id} (Job ID: {job_id})")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to submit {job_type} run {run_id}: {e.stderr}")
        return None

def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    print("=" * 60)
    print("Submitting 10 training runs (5 LightGBM GPU + 5 NN)")
    print("=" * 60)
    print(f"Project directory: {project_dir}")
    print()
    
    # Submit 5 LightGBM GPU runs (run-ids 0-4)
    print("Submitting 5 LightGBM GPU runs...")
    lightgbm_gpu_script = script_dir / "submit_flop5_lightgbm_gpu.sh"
    lightgbm_gpu_job_ids = []
    
    for i in range(5):
        job_id = submit_slurm_job(lightgbm_gpu_script, i, "LightGBM-GPU")
        if job_id:
            lightgbm_gpu_job_ids.append(job_id)
    
    print()
    
    # Submit 5 NN runs (run-ids 0-4)
    print("Submitting 5 NN (Neural Network) runs...")
    nn_script = script_dir / "submit_flop5.sh"
    nn_job_ids = []
    
    for i in range(5):
        job_id = submit_slurm_job(nn_script, i, "NN")
        if job_id:
            nn_job_ids.append(job_id)
    
    print()
    print("=" * 60)
    print("Submission Summary")
    print("=" * 60)
    print(f"LightGBM GPU jobs submitted: {len(lightgbm_gpu_job_ids)}/5")
    print(f"NN jobs submitted: {len(nn_job_ids)}/5")
    print()
    print("Run names will be:")
    print("  LightGBM GPU: EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run{0-4}")
    print("  NN:           EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_run{0-4}")
    print()
    print("Check job status with: squeue -u $USER")
    print()
    
    if lightgbm_gpu_job_ids or nn_job_ids:
        print("Job IDs:")
        if lightgbm_gpu_job_ids:
            print(f"  LightGBM GPU: {', '.join(lightgbm_gpu_job_ids)}")
        if nn_job_ids:
            print(f"  NN: {', '.join(nn_job_ids)}")

if __name__ == "__main__":
    main()

