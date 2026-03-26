#!/usr/bin/env python3
"""
exp4_lp_suites.py
=================
Task 4: Apply KaMIS LP relaxation to other solver suites.

Compares baseline vs LP-preprocessed variants:
  Greedy_MinDeg       vs  LP+Greedy_MinDeg
  LocalSearch_1_2     vs  LP+LocalSearch_1_2
  SimulatedAnnealing  vs  LP+SimulatedAnnealing

LP preprocessing (via scipy.optimize.linprog / HiGHS):
  - Solve MIS LP relaxation.
  - Fix LP=1 nodes (include in IS), LP=0 nodes (exclude).
  - Run solver on residual graph of fractional nodes.

Scalability: LP skipped automatically for m > 100,000 edges.
Effective range: sparse instances (p ≤ 0.05, n ≤ 1000).

Results: oneal/benchmark_results/exp4_results.json
"""

import os, sys, argparse, time
from itertools import chain

_ONEAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ONEAL)

from benchmark_suite import (
    FAMILY_A_N, FAMILY_A_P, FAMILY_A_SEEDS, TIME_LIMIT_S,
    iter_family_a, iter_family_b,
    ExperimentResult, make_result,
    save_results, load_or_init_results, get_completed_ids,
    print_comparison_table,
)
from mis_benchmark_combined import (
    greedy_min_degree, local_search_1_2_swap,
    simulated_annealing_mis,
)
from lp_reduction import lp_preprocess_then_solve, lp_stats

EXPERIMENT  = "exp4_lp_suites"
OUTPUT_FILE = os.path.join(_ONEAL, "benchmark_results", "exp4_results.json")

# Only sparse instances where LP is effective
_LP_P_VALUES  = [0.001, 0.01, 0.05]
_LP_N_MAX     = 1000


def _greedy(G):    return greedy_min_degree(G)
def _ls(G):        return local_search_1_2_swap(G, greedy_min_degree(G))
def _sa(G):        return simulated_annealing_mis(G)


_BASE_SOLVERS = {
    "Greedy_MinDeg":    _greedy,
    "LocalSearch_1_2":  _ls,
    "SimulatedAnnealing": _sa,
}
_LP_SOLVERS = {
    "LP+Greedy_MinDeg":     _greedy,
    "LP+LocalSearch_1_2":   _ls,
    "LP+SimAnneal":         _sa,
}
_ALL_SOLVERS = list(_BASE_SOLVERS) + list(_LP_SOLVERS)


def run_exp4(
    output: str = OUTPUT_FILE,
    resume: bool = True,
    n_max: int = _LP_N_MAX,
    time_limit: float = TIME_LIMIT_S,
) -> None:
    results = load_or_init_results(output) if resume else []
    done    = {s: get_completed_ids(results, s) for s in _ALL_SOLVERS}

    instances = list(chain(
        iter_family_a(
            n_values=[n for n in FAMILY_A_N if n <= n_max],
            p_values=_LP_P_VALUES,
        ),
        iter_family_b(n_values=[n for n in [100, 200, 500] if n <= n_max]),
    ))
    print(f"[exp4] {len(instances)} instances × {len(_ALL_SOLVERS)} solvers")

    for inst in instances:
        G = inst.graph
        m = G.number_of_edges()

        # ── Baseline solvers ──────────────────────────────────────────────────
        for sname, fn in _BASE_SOLVERS.items():
            if inst.instance_id in done[sname]:
                continue
            t0 = time.time()
            sol = fn(G)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit)
            results.append(r); done[sname].add(inst.instance_id)
            print(f"  {sname:25s} {inst.instance_id[:38]:38s} "
                  f"size={r.solution_size} recall={r.recall:.3f} t={rt:.2f}s")

        # ── LP-preprocessed solvers ───────────────────────────────────────────
        for sname, base_fn in _LP_SOLVERS.items():
            if inst.instance_id in done[sname]:
                continue

            t0 = time.time()
            sol, total_t, meta = lp_preprocess_then_solve(G, base_fn)
            rt = time.time() - t0

            note = (
                f"lp_fixed_in={meta['lp_fixed_in']},"
                f"lp_fixed_out={meta['lp_fixed_out']},"
                f"lp_frac={meta['lp_fractional']},"
                f"lp_ratio={meta['lp_reduction_ratio']:.3f},"
                f"lp_t={meta['lp_time']:.2f}s"
            )
            r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit, notes=note)
            results.append(r); done[sname].add(inst.instance_id)
            print(f"  {sname:25s} {inst.instance_id[:38]:38s} "
                  f"size={r.solution_size} recall={r.recall:.3f} "
                  f"lp_ratio={meta['lp_reduction_ratio']:.2f} t={rt:.2f}s")

        save_results(results, output)

    print(f"[exp4] Done. Results: {output}")


def analyze_exp4(results_file: str = OUTPUT_FILE) -> None:
    results = load_or_init_results(results_file)
    if not results:
        print("[exp4] No results to analyze.")
        return

    import numpy as np
    from collections import defaultdict

    print("\n=== Exp4: LP Relaxation Preprocessing ===")
    print("\n--- Recall by n (all p values) ---")
    print_comparison_table(results, _ALL_SOLVERS, group_field="n", metric="recall")

    # Extract LP metadata from notes field
    print("\n--- LP reduction ratio by instance ---")
    for r in results:
        if r.solver.startswith("LP+") and "lp_ratio=" in r.notes:
            try:
                ratio = float(r.notes.split("lp_ratio=")[1].split(",")[0])
                print(f"  {r.instance_id[:45]:45s} lp_ratio={ratio:.3f}")
            except Exception:
                pass

    # Delta recall: LP+X vs X
    pairs = [
        ("Greedy_MinDeg",    "LP+Greedy_MinDeg"),
        ("LocalSearch_1_2",  "LP+LocalSearch_1_2"),
        ("SimulatedAnnealing", "LP+SimAnneal"),
    ]
    print("\n--- Recall improvement from LP preprocessing ---")
    for base_s, lp_s in pairs:
        by_id: Dict = defaultdict(dict)
        for r in results:
            if r.solver in (base_s, lp_s):
                by_id[r.instance_id][r.solver] = r.recall
        deltas = []
        for iid, sv in by_id.items():
            if base_s in sv and lp_s in sv:
                deltas.append(sv[lp_s] - sv[base_s])
        if deltas:
            print(f"  {base_s:22s} -> {lp_s:25s}: "
                  f"avg delta_recall = {np.mean(deltas):+.4f}  "
                  f"({sum(d>0 for d in deltas)}/{len(deltas)} improved)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",       default=OUTPUT_FILE)
    parser.add_argument("--no_resume",    action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--n_max",        type=int, default=_LP_N_MAX)
    parser.add_argument("--time_limit",   type=float, default=TIME_LIMIT_S)
    args = parser.parse_args()

    if args.analyze_only:
        analyze_exp4(args.output)
    else:
        run_exp4(args.output,
                 resume=not args.no_resume,
                 n_max=args.n_max,
                 time_limit=args.time_limit)
        analyze_exp4(args.output)
