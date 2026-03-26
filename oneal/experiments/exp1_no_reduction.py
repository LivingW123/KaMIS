#!/usr/bin/env python3
"""
exp1_no_reduction.py
====================
Task 1: Does KaMIS without full reduction outperform on non-sparse graphs?

Compares:
  - KaMIS_online_mis    : online_mis binary  (ARW ILS + light online reductions)
  - KaMIS_redumis       : redumis binary     (full kernelisation + evolutionary)
  - KaMIS_mmwis_nored   : mmwis binary with --disable_* flags (no reductions)

Hypothesis: redumis dominates at sparse p ≤ 0.01; online_mis is competitive
or better at dense p ≥ 0.2 where reduction rules rarely fire.

Results: oneal/benchmark_results/exp1_results.json
"""

import os, sys, argparse
from itertools import chain

_ONEAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ONEAL)

from benchmark_suite import (
    FAMILY_A_N, FAMILY_A_P, FAMILY_A_SEEDS, TIME_LIMIT_S,
    iter_family_a, iter_family_b,
    WSLRunner, ExperimentResult, make_result,
    save_results, load_or_init_results, get_completed_ids,
    print_comparison_table,
)

EXPERIMENT   = "exp1_no_reduction"
OUTPUT_FILE  = os.path.join(_ONEAL, "benchmark_results", "exp1_results.json")

# Disable-all-reductions flags for mmwis
_MMWIS_NO_RED_FLAGS = [
    "--disable_fold1",
    "--disable_neighborhood",
    "--disable_twin",
    "--disable_clique",
    "--disable_triangle",
    "--disable_v_shape_min",
    "--disable_v_shape_mid",
    "--disable_v_shape_max",
    "--disable_basic_se",
    "--disable_extended_se",
    "--disable_generalized_fold",
    "--disable_heavy_set",
    "--disable_critical_set",
    "--disable_clique_neighborhood",
    "--disable_generalized_neighborhood",
]


def run_exp1(
    kamis_path: str = None,
    output: str = OUTPUT_FILE,
    resume: bool = True,
    n_max: int = 2000,
    time_limit: float = TIME_LIMIT_S,
) -> None:
    runner  = WSLRunner(kamis_path)
    results = load_or_init_results(output) if resume else []

    solvers = ["KaMIS_online_mis", "KaMIS_redumis", "KaMIS_mmwis_nored"]
    done    = {s: get_completed_ids(results, s) for s in solvers}

    instances = list(chain(
        iter_family_a(n_values=[n for n in FAMILY_A_N if n <= n_max]),
        iter_family_b(),
    ))
    total = len(instances) * len(solvers)
    done_count = sum(len(v) for v in done.values())
    print(f"[exp1] {total} solver×instance combinations ({done_count} already done)")

    if not runner.is_available:
        print("[exp1] WARNING: KaMIS binaries unavailable (WSL not found or no deploy/). "
              "All results will be empty.")

    for inst in instances:
        if inst.n > n_max:
            continue

        for solver in solvers:
            if inst.instance_id in done[solver]:
                continue

            sol = set()
            rt  = -1.0
            note = ""

            if runner.is_available:
                if solver == "KaMIS_online_mis":
                    sol, rt = runner.solve_unweighted(inst, "online_mis", time_limit)
                elif solver == "KaMIS_redumis":
                    sol, rt = runner.solve_unweighted(inst, "redumis", time_limit)
                elif solver == "KaMIS_mmwis_nored":
                    sol, rt = runner.solve_unweighted(
                        inst, "mmwis", time_limit,
                        extra_args=_MMWIS_NO_RED_FLAGS + ["--weight_source=unit"],
                    )
                    note = "no_reductions"
            else:
                note = "wsl_unavailable"

            r = make_result(EXPERIMENT, inst, solver, sol, rt, time_limit, notes=note)
            results.append(r)
            done[solver].add(inst.instance_id)

            status = f"size={r.solution_size}" if sol else "no_sol"
            print(f"  {solver:28s} {inst.instance_id[:40]:40s} {status} t={rt:.1f}s")

        save_results(results, output)

    print(f"[exp1] Done. Results: {output}")


def analyze_exp1(results_file: str = OUTPUT_FILE) -> None:
    results = load_or_init_results(results_file)
    if not results:
        print("[exp1] No results to analyze.")
        return

    solvers = ["KaMIS_online_mis", "KaMIS_redumis", "KaMIS_mmwis_nored"]

    print("\n=== Exp1: KaMIS with vs without reductions ===")
    print("\n--- Average recall by graph density (p) ---")
    for p in sorted({r.params.get("p") for r in results if "p" in r.params}):
        subset = [r for r in results if r.params.get("p") == p]
        if not subset:
            continue
        print(f"\n  p = {p}")
        print_comparison_table(subset, solvers, group_field="n", metric="recall")

    print("\n--- Average IS size by density ---")
    print_comparison_table(results, solvers, group_field="n", metric="solution_size")

    # Ratio: online_mis / redumis
    import numpy as np
    from collections import defaultdict
    ratio: Dict = defaultdict(list)
    by_id: Dict = defaultdict(dict)
    for r in results:
        if r.solver in solvers and r.solution_size > 0:
            by_id[r.instance_id][r.solver] = r.solution_size
    for iid, sv in by_id.items():
        if "KaMIS_online_mis" in sv and "KaMIS_redumis" in sv and sv["KaMIS_redumis"] > 0:
            rat = sv["KaMIS_online_mis"] / sv["KaMIS_redumis"]
            # get n and p from results
            for r in results:
                if r.instance_id == iid and r.solver == "KaMIS_redumis":
                    ratio[(r.n, r.params.get("p"))].append(rat)
                    break

    print("\n--- online_mis / redumis size ratio (>1 = online wins) ---")
    print(f"  {'(n, p)':20s}  {'ratio':>8s}")
    for key in sorted(ratio):
        vals = ratio[key]
        print(f"  {str(key):20s}  {np.mean(vals):8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kamis_path",   default=None)
    parser.add_argument("--output",       default=OUTPUT_FILE)
    parser.add_argument("--no_resume",    action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--n_max",        type=int, default=2000)
    parser.add_argument("--time_limit",   type=float, default=TIME_LIMIT_S)
    args = parser.parse_args()

    if args.analyze_only:
        analyze_exp1(args.output)
    else:
        run_exp1(args.kamis_path, args.output,
                 resume=not args.no_resume,
                 n_max=args.n_max,
                 time_limit=args.time_limit)
        analyze_exp1(args.output)
