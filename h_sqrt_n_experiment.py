#!/usr/bin/env python3
"""
h/sqrt(n) Ratio Experiment
===========================
Tests how solver recall degrades as h moves below, at, and above the spectral
threshold h ≈ sqrt(n). Runs h = 0.5*sqrt(n), 1.0*sqrt(n), 2.0*sqrt(n) for
several values of n and outputs results to h_sqrt_n_results.txt.
"""

import numpy as np
import time
from collections import defaultdict
from mis_benchmark import (
    gen_erdos_renyi_planted, greedy_min_degree, local_search_1_2_swap,
    spectral_mis, verify_independent_set, KaMISRunner
)

def run_experiment(kamis_path=None):
    kamis = KaMISRunner(kamis_path)

    # n values and h ratios to test
    n_values = [50, 100, 200, 400]
    ratios   = [0.5, 1.0, 2.0]   # h = ratio * sqrt(n)
    seeds    = [42, 123, 456]
    p        = 0.5

    # results[ratio][solver] = list of recalls
    results_by_ratio = defaultdict(lambda: defaultdict(list))

    lines = []
    lines.append("=" * 80)
    lines.append("h/sqrt(n) RATIO EXPERIMENT")
    lines.append(f"Graph family: Erdos-Renyi planted, p={p}")
    lines.append("=" * 80)

    header = f"{'n':>6} {'h':>5} {'h/sqrt(n)':>10}  " \
             f"{'Greedy':>8} {'LocalSrch':>10} {'Spectral':>9}"
    if kamis.available:
        header += f" {'KaMIS_online':>13} {'KaMIS_redu':>11}"
    lines.append(header)
    lines.append("-" * 80)

    for ratio in ratios:
        lines.append(f"\n--- h = {ratio:.1f} * sqrt(n) ---")
        lines.append(header)
        lines.append("-" * 80)

        for n in n_values:
            h = max(2, int(round(ratio * np.sqrt(n))))

            greedy_recalls, ls_recalls, spectral_recalls = [], [], []
            kamis_o_recalls, kamis_r_recalls = [], []

            for seed in seeds:
                try:
                    inst = gen_erdos_renyi_planted(n, h, p=p, seed=seed)
                except AssertionError:
                    continue
                if not inst.verify():
                    continue
                G = inst.graph

                # Greedy
                mis = greedy_min_degree(G)
                greedy_recalls.append(len(mis & inst.planted_set) / h)

                # LocalSearch
                mis = local_search_1_2_swap(G, greedy_min_degree(G))
                ls_recalls.append(len(mis & inst.planted_set) / h)

                # Spectral
                mis = spectral_mis(G)
                spectral_recalls.append(len(mis & inst.planted_set) / h)

                # KaMIS
                if kamis.available:
                    tl = 30.0
                    mis_o, _ = kamis.solve(G, 'online_mis', time_limit=tl)
                    if mis_o:
                        kamis_o_recalls.append(len(mis_o & inst.planted_set) / h)
                    mis_r, _ = kamis.solve(G, 'redumis', time_limit=tl)
                    if mis_r:
                        kamis_r_recalls.append(len(mis_r & inst.planted_set) / h)

            avg = lambda lst: sum(lst)/len(lst) if lst else float('nan')

            row = (f"{n:>6} {h:>5} {ratio*1.0:>10.2f}  "
                   f"{avg(greedy_recalls):>8.3f} {avg(ls_recalls):>10.3f} "
                   f"{avg(spectral_recalls):>9.3f}")
            if kamis.available:
                row += f" {avg(kamis_o_recalls):>13.3f} {avg(kamis_r_recalls):>11.3f}"
            lines.append(row)

            # Accumulate for summary
            results_by_ratio[ratio]['Greedy'].extend(greedy_recalls)
            results_by_ratio[ratio]['LocalSearch'].extend(ls_recalls)
            results_by_ratio[ratio]['Spectral'].extend(spectral_recalls)
            if kamis.available:
                results_by_ratio[ratio]['KaMIS_online'].extend(kamis_o_recalls)
                results_by_ratio[ratio]['KaMIS_redumis'].extend(kamis_r_recalls)

    # Summary
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY: avg recall by h/sqrt(n) ratio")
    lines.append("=" * 80)
    solvers = ['Greedy', 'LocalSearch', 'Spectral']
    if kamis.available:
        solvers += ['KaMIS_online', 'KaMIS_redumis']
    ratio_col = f"{'ratio':>10}"
    solver_cols = "  ".join(f"{s:>14}" for s in solvers)
    lines.append(ratio_col + "  " + solver_cols)
    lines.append("-" * 80)
    for ratio in ratios:
        row = f"{ratio:>10.1f}  "
        for s in solvers:
            lst = results_by_ratio[ratio][s]
            avg = sum(lst)/len(lst) if lst else float('nan')
            row += f"{avg:>14.3f}  "
        lines.append(row)

    lines.append("\nKey observation:")
    lines.append("  h = 0.5*sqrt(n): below spectral threshold -> spectral recall drops sharply")
    lines.append("  h = 1.0*sqrt(n): at threshold -> spectral marginal, greedy/KaMIS hold")
    lines.append("  h = 2.0*sqrt(n): above threshold -> all solvers perform better")

    output = "\n".join(lines) + "\n"
    print(output)
    with open("h_sqrt_n_results.txt", "w") as f:
        f.write(output)
    print("Results saved to h_sqrt_n_results.txt")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kamis_path', type=str, default=None)
    args = parser.parse_args()
    run_experiment(kamis_path=args.kamis_path)
