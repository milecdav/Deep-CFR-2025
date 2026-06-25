#!/usr/bin/env python3
"""
Aggregate H2H matrix results from slurm_out h2h-matrix-*.out files.

Computes:
- Per-run averages for each LGBM run (run5, run6, ...)
- Per-run averages for each NN run (run0, run1, ...)
- Total average across all matches

Outputs JSON and optionally a summary text file.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional


def parse_h2h_file(filepath: Path) -> Optional[dict]:
    """Parse a single h2h-matrix output file. Returns dict with iteration, matches, or None if parse fails."""
    text = filepath.read_text()
    
    # Extract iteration from header
    iter_match = re.search(r'^Iterations:\s*(\d+)', text, re.MULTILINE)
    if not iter_match:
        return None
    iteration = int(iter_match.group(1))
    
    # Parse summary table lines: "| ... | Mean: VAL" (optional [L, U] after are ignored)
    pattern = re.compile(
        r'[✓✗]\s*\|\s*(.+?)\s*\(iter\s*\d+\)\s*\|\s*(.+?)\s*\(iter\s*\d+\)\s*\|\s*Mean:\s*([-\d.]+)'
    )
    
    matches = []
    for m in pattern.finditer(text):
        exp1, exp2, mean_str = m.group(1).strip(), m.group(2).strip(), m.group(3)
        try:
            mean_val = float(mean_str)
        except ValueError:
            continue
        matches.append({
            'exp1': exp1,
            'exp2': exp2,
            'mean': mean_val,
        })
    
    if not matches:
        return None
    
    return {'iteration': iteration, 'matches': matches}


def extract_run_id(exp_name: str, prefix: str) -> Optional[str]:
    """Extract run ID like run5, run0 from experiment name."""
    # e.g. EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run5 -> run5
    m = re.search(rf'{re.escape(prefix)}_run(\d+)', exp_name, re.IGNORECASE)
    if m:
        return f"run{m.group(1)}"
    m = re.search(r'_run(\d+)(?:\s|$)', exp_name)
    if m:
        return f"run{m.group(1)}"
    return None


def aggregate_matches(matches: list) -> dict:
    """
    Aggregate match results.
    exp1 = LGBM (Agent 1), exp2 = NN (Agent 2).
    Mean is from Agent 1 perspective: positive = LGBM wins.
    """
    lgbm_scores = defaultdict(list)  # run_id -> list of means (LGBM perspective)
    nn_scores = defaultdict(list)    # run_id -> list of -means (NN perspective, since Agent 2 = -Agent 1)
    
    all_means = []

    match_results = []  # list of {"label": "run5 vs run0", "mean": x} for each pair
    for m in matches:
        mean = m['mean']
        all_means.append(mean)
        lgbm_run = extract_run_id(m['exp1'], 'LightGBM') or extract_run_id(m['exp1'], 'LGBM')
        nn_run = extract_run_id(m['exp2'], 'NN')
        pair_label = f"{lgbm_run or 'LGBM'} vs {nn_run or 'NN'}"
        match_results.append({'label': pair_label, 'mean': mean})
        if lgbm_run:
            lgbm_scores[lgbm_run].append(mean)
        if nn_run:
            nn_scores[nn_run].append(-mean)  # NN perspective: negative of LGBM score
    
    lgbm_avgs = {k: sum(v) / len(v) for k, v in lgbm_scores.items() if v}
    nn_avgs = {k: sum(v) / len(v) for k, v in nn_scores.items() if v}
    n = len(all_means)
    total_avg = sum(all_means) / n if all_means else None

    return {
        'lgbm_run_averages': dict(sorted(lgbm_avgs.items())),
        'nn_run_averages': dict(sorted(nn_avgs.items())),
        'total_average': total_avg,
        'n_matches': n,
        'match_results': match_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Aggregate H2H matrix results')
    parser.add_argument('input_dir', type=str, default='slurm_out/big_lgbm',
                        nargs='?', help='Directory containing h2h-matrix-*.out files')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON path (default: input_dir/h2h_aggregated.json)')
    parser.add_argument('--summary', type=str, default=None,
                        help='Output summary text path (default: input_dir/h2h_summary.txt)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Directory not found: {input_dir}")
    
    out_json = Path(args.output) if args.output else input_dir / 'h2h_aggregated.json'
    out_summary = Path(args.summary) if args.summary else input_dir / 'h2h_summary.txt'
    
    files = sorted(input_dir.glob('h2h-matrix-*.out'))
    if not files:
        raise SystemExit(f"No h2h-matrix-*.out files found in {input_dir}")
    
    results_by_iteration = {}
    
    for f in files:
        parsed = parse_h2h_file(f)
        if parsed is None:
            print(f"Warning: Could not parse {f.name}, skipping")
            continue
        it = parsed['iteration']
        agg = aggregate_matches(parsed['matches'])
        results_by_iteration[it] = agg
        print(f"Parsed {f.name}: iteration {it}, {agg['n_matches']} matches")
    
    # Sort by iteration
    sorted_iters = sorted(results_by_iteration.keys())
    output = {
        'iterations': sorted_iters,
        'data': {it: results_by_iteration[it] for it in sorted_iters},
    }
    
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_json}")
    
    # Write summary
    lines = [
        "H2H Aggregated Results",
        "=" * 60,
        "",
    ]
    for it in sorted_iters:
        d = results_by_iteration[it]
        lines.append(f"Iteration {it}:")
        lines.append(f"  Total average (LGBM perspective): {d['total_average']:.2f} MBB_per_G")
        lines.append(f"  LGBM run averages: {d['lgbm_run_averages']}")
        lines.append(f"  NN run averages:   {d['nn_run_averages']}")
        lines.append("")
    
    with open(out_summary, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {out_summary}")


if __name__ == '__main__':
    main()
