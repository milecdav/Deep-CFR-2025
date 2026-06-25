#!/usr/bin/env python3
"""
Plot H2H aggregated results (LGBM perspective only).

- Total average: bold line
- Per-run LGBM averages: slightly transparent lines (each = one LGBM run vs all NN opponents)
- Y-axis: MBB_per_G (positive = LGBM wins)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Plot H2H aggregated results')
    parser.add_argument('input', type=str, nargs='?', default='slurm_out/big_lgbm/h2h_aggregated.json',
                        help='Path to h2h_aggregated.json (from aggregate_h2h_results.py)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output plot path (default: same dir as input, h2h_plot.png)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Transparency for per-run lines (default: 0.4)')
    parser.add_argument('--linewidth', type=float, default=2.0,
                        help='Line width for total average (default: 2.0)')
    parser.add_argument('--plot-all-pairs', action='store_true',
                        help='Plot each of the 25 LGBM vs NN pair results as separate lines')
    parser.add_argument('--pairs-alpha', type=float, default=0.25,
                        help='Transparency for all-pairs lines when --plot-all-pairs (default: 0.25)')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path}. Run aggregate_h2h_results.py first.")
    
    with open(input_path) as f:
        data = json.load(f)
    
    iterations = np.array(data['iterations'])
    n = len(iterations)
    
    # JSON keys are strings
    def get_data(it):
        return data['data'][str(int(it))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    first_data = get_data(iterations[0]) if n > 0 else {}
    
    # Plot all 25 pair lines (if option set and data available)
    if args.plot_all_pairs and n > 0:
        match_results = first_data.get('match_results', [])
        for pair_idx, pair in enumerate(match_results):
            vals = []
            for it in iterations:
                mr = get_data(it).get('match_results', [])
                if pair_idx < len(mr):
                    vals.append(mr[pair_idx].get('mean'))
                else:
                    vals.append(None)
            if all(v is not None for v in vals):
                ax.plot(iterations, vals, alpha=args.pairs_alpha, color='#95a5a6', linewidth=0.8)
        if match_results:
            ax.plot([], [], alpha=args.pairs_alpha, color='#95a5a6', linewidth=0.8,
                    label=f'All pairs ({len(match_results)})')
    
    # Plot per-run LGBM lines (transparent) — each run's average vs all NN opponents
    lgbm_runs = sorted(first_data.get('lgbm_run_averages', {}).keys())
    for i, run in enumerate(lgbm_runs):
        vals = [get_data(it).get('lgbm_run_averages', {}).get(run) for it in iterations]
        if all(v is not None for v in vals):
            leg_label = 'LGBM (per run)' if i == 0 else None
            ax.plot(iterations, vals, alpha=args.alpha, color='#2ecc71', linewidth=1, label=leg_label)
    
    # Plot total average (bold)
    total_vals = [get_data(it)['total_average'] for it in iterations]
    ax.plot(iterations, total_vals, color='#2c3e50', linewidth=args.linewidth, label='Total average', zorder=10)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MBB per G (LGBM perspective, positive = LGBM wins)')
    ax.set_title('H2H: LightGBM vs NN (LGBM perspective)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    out_path = Path(args.output) if args.output else input_path.parent / 'h2h_plot.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
