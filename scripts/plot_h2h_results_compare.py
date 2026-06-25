#!/usr/bin/env python3
"""
Plot two or three H2H aggregated JSON files on one figure for comparison.

Examples:
  # Two runs:
  python scripts/plot_h2h_results_compare.py \
    slurm_out/small/h2h_aggregated.json "Small NN" \
    slurm_out/medium/h2h_aggregated.json "Medium NN" \
    -o slurm_out/h2h_compare.png

  # Three runs:
  python scripts/plot_h2h_results_compare.py \
    slurm_out/small/h2h_aggregated.json "Small NN" \
    slurm_out/medium/h2h_aggregated.json "Medium NN" \
    slurm_out/large/h2h_aggregated.json "Large NN" \
    -o slurm_out/h2h_compare.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration matching reference script style
ALPHA = 0.3
FIGX = 6
FIGY = 4.5
CONFIDENCE = 0.95
LINEWIDTH = 1.0


def main():
    parser = argparse.ArgumentParser(
        description='Plot two or more H2H aggregated results on one figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input1', type=str, help='First h2h_aggregated.json path')
    parser.add_argument('label1', type=str, help='Label for first series (e.g. "Small NN")')
    parser.add_argument('input2', type=str, help='Second h2h_aggregated.json path')
    parser.add_argument('label2', type=str, help='Label for second series (e.g. "Medium NN")')
    parser.add_argument('input3', type=str, nargs='?', default=None, help='Third h2h_aggregated.json path (optional)')
    parser.add_argument('label3', type=str, nargs='?', default=None, help='Label for third series (e.g. "Large NN")')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output plot path (default: slurm_out/h2h_compare.png)')
    parser.add_argument('--alpha', type=float, default=0.55,
                        help='Transparency for per-run lines (default: 0.35)')
    parser.add_argument('--linewidth', type=float, default=1.5,
                        help='Line width for total average lines (default: 2.5)')
    parser.add_argument('--no-per-run', action='store_true',
                        help='Do not plot per-run LGBM lines, only total averages')
    parser.add_argument('--plot-all-pairs', action='store_true',
                        help='Plot each of the 25 LGBM vs NN pair results as separate lines per dataset')
    parser.add_argument('--pairs-alpha', type=float, default=0.6,
                        help='Transparency for all-pairs lines when --plot-all-pairs (default: 0.2)')
    args = parser.parse_args()

    # Validate that input3 and label3 are provided together
    if (args.input3 is None) != (args.label3 is None):
        parser.error("input3 and label3 must be provided together (both or neither)")

    # Build paths_labels list based on provided inputs
    paths_labels = [(args.input1, args.label1), (args.input2, args.label2)]
    if args.input3 is not None and args.label3 is not None:
        paths_labels.append((args.input3, args.label3))
    
    # Colors for up to 3 series
    base_colors = ['#dd8452', '#55a868', '#c44e52']   # orange, green, red (main averages)
    run_colors = ['#4c72b0', '#8172b3', '#ccb974']    # blue, violet, yellow (run averages)
    pairs_colors = ['#f8d6c5', '#cfeedd', '#f4c2c2']   # light orange, light green, light red (per run)

    fig, ax = plt.subplots(figsize=(FIGX, FIGY))

    for idx, (input_path, label) in enumerate(paths_labels):
        path = Path(input_path)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        with open(path) as f:
            data = json.load(f)
        iterations = np.array(data['iterations'])
        n = len(iterations)
        avg_color = base_colors[idx % len(base_colors)]
        run_color = run_colors[idx % len(run_colors)]
        pairs_color = pairs_colors[idx % len(pairs_colors)]

        def get_data(it):
            return data['data'][str(int(it))]

        # All 25 pair lines (if option set) - all pairs from same file use same color
        if args.plot_all_pairs and n > 0:
            first_data = get_data(iterations[0])
            match_results = first_data.get('match_results', [])
            for pair_idx in range(len(match_results)):
                vals = []
                for it in iterations:
                    mr = get_data(it).get('match_results', [])
                    if pair_idx < len(mr):
                        vals.append(mr[pair_idx].get('mean'))
                    else:
                        vals.append(None)
                if all(v is not None for v in vals):
                    # All pairs from this aggregation file use the same color
                    ax.plot(iterations, vals, alpha=args.pairs_alpha, color=pairs_color, linewidth=0.8)
            if match_results:
                ax.plot([], [], alpha=args.pairs_alpha, color=pairs_color, linewidth=0.8,
                        label=f'{label} (all pairs)')

        # Per-run LGBM lines (same color as average, transparent)
        if not args.no_per_run and n > 0:
            first_data = get_data(iterations[0])
            lgbm_runs = sorted(first_data.get('lgbm_run_averages', {}).keys())
            for i, run in enumerate(lgbm_runs):
                vals = [get_data(it).get('lgbm_run_averages', {}).get(run) for it in iterations]
                if all(v is not None for v in vals):
                    # Use same color as average, just with transparency
                    run_label = f'{label} (per run)' if i == 0 else None
                    ax.plot(iterations, vals, alpha=args.alpha, color=run_color, linewidth=1, label=run_label)

        # Total average (bold, base color)
        total_vals = [get_data(it)['total_average'] for it in iterations]
        ax.plot(iterations, total_vals, color=avg_color, linewidth=args.linewidth, label=label, zorder=10)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('LUGL-DeepCFR-LightGBM winnings in mbb/h')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    out_path = Path(args.output) if args.output else Path('slurm_out/h2h_compare.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    
    # Save with high DPI (matching reference script style)
    # Save to requested format
    fig.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    
    # Also save PDF version for publication quality (common in papers)
    if out_path.suffix.lower() != '.pdf':
        pdf_path = out_path.with_suffix('.pdf')
        fig.savefig(pdf_path, dpi=300)
        print(f"Saved plot to {pdf_path}")
    
    plt.close()


if __name__ == '__main__':
    main()
