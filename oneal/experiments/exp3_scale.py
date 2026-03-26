#!/usr/bin/env python3
"""
exp3_scale.py
=============
Task 3: Does HILS(+Red) perform better than alternatives up to 10^8?

Three sub-runs:
  (a) Python solvers: n ≤ 2000, weighted instances.
      Solvers: Python_HILS_W, Python_RedHILS_W
  (b) C++ binaries: n up to 50000, weighted instances.
      Solvers: mmwis, weighted_local_search, weighted_branch_reduce
  (c) Overlap region n in {500, 1000, 2000}: both Python and C++ solvers.

Note: 10^8 nodes is infeasible in Python; C++ binaries can approach 10^5-10^6
within 30s. We validate the trend and extrapolate.

Results: oneal/benchmark_results/exp3_results.json
"""

import os, sys, argparse, time
from itertools import chain

_ONEAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ONEAL)

from benchmark_suite import (
    FAMILY_A_N, FAMILY_A_P, FAMILY_A_SEEDS, LARGE_N, LARGE_P, LARGE_SEEDS,
    TIME_LIMIT_S,
    iter_family_a, iter_large_scale, iter_weighted, add_weights,
    WSLRunner, ExperimentResult, make_result,
    save_results, load_or_init_results, get_completed_ids,
    print_comparison_table,
)
from hils import hils_weighted, red_hils_weighted, HilsConfig

EXPERIMENT  = "exp3_scale"
OUTPUT_FILE = os.path.join(_ONEAL, "benchmark_results", "exp3_results.json")

_CPP_WEIGHTED = ["mmwis", "weighted_local_search", "weighted_branch_reduce"]
_PY_WEIGHTED  = ["Python_HILS_W", "Python_RedHILS_W"]
_ALL_SOLVERS  = _PY_WEIGHTED + [f"KaMIS_{b}" for b in _CPP_WEIGHTED]


def run_exp3(
    kamis_path: str = None,
    output: str = OUTPUT_FILE,
    resume: bool = True,
    n_max_python: int = 2000,
    n_max_cpp: int = 50000,
    time_limit: float = TIME_LIMIT_S,
    overlap_only: bool = False,
) -> None:
    runner  = WSLRunner(kamis_path)
    results = load_or_init_results(output) if resume else []
    done    = {s: get_completed_ids(results, s) for s in _ALL_SOLVERS}

    cfg = HilsConfig(max_iter=2_000_000, time_limit=time_limit, seed=42)

    # ── (a)+(c) Python solvers: Family A, n <= n_max_python ──────────────────
    if not overlap_only:
        py_instances = list(iter_weighted(iter_family_a(
            n_values=[n for n in FAMILY_A_N if n <= n_max_python],
        )))
    else:
        py_instances = list(iter_weighted(iter_family_a(
            n_values=[n for n in [500, 1000, 2000] if n <= n_max_python],
        )))

    print(f"[exp3] Python instances: {len(py_instances)}")

    for winst in py_instances:
        G = winst.graph

        # Python HILS weighted
        sname = "Python_HILS_W"
        if winst.instance_id not in done[sname]:
            t0 = time.time()
            sol, _ = hils_weighted(G, winst.weights, cfg)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, winst, sname, sol, rt, time_limit)
            results.append(r); done[sname].add(winst.instance_id)
            print(f"  {sname:22s} n={winst.n:5d} p={winst.params.get('p','?'):5} "
                  f"w={r.solution_weight} t={rt:.1f}s")

        # Python Red+HILS weighted
        sname = "Python_RedHILS_W"
        if winst.instance_id not in done[sname]:
            t0 = time.time()
            sol, _ = red_hils_weighted(G, winst.weights, cfg)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, winst, sname, sol, rt, time_limit)
            results.append(r); done[sname].add(winst.instance_id)
            print(f"  {sname:22s} n={winst.n:5d} p={winst.params.get('p','?'):5} "
                  f"w={r.solution_weight} t={rt:.1f}s")

        save_results(results, output)

    # ── (b)+(c) C++ solvers: up to n_max_cpp ─────────────────────────────────
    if not runner.is_available:
        print("[exp3] WARNING: C++ binaries unavailable.")
        return

    cpp_instances_fa = list(iter_weighted(iter_family_a(
        n_values=[n for n in FAMILY_A_N if n <= n_max_cpp],
    )))
    cpp_instances_lg = list(iter_weighted(iter_large_scale(
        n_values=[n for n in LARGE_N if n <= n_max_cpp],
    )))
    cpp_instances = cpp_instances_fa + cpp_instances_lg
    print(f"[exp3] C++ instances: {len(cpp_instances)}")

    for winst in cpp_instances:
        for binary in _CPP_WEIGHTED:
            sname = f"KaMIS_{binary}"
            if winst.instance_id in done[sname]:
                continue
            sol, rt = runner.solve_weighted(winst, binary, time_limit,
                                            extra_args=["--weight_source=file"])
            r = make_result(EXPERIMENT, winst, sname, sol, rt, time_limit)
            results.append(r); done[sname].add(winst.instance_id)
            print(f"  {sname:30s} n={winst.n:5d} w={r.solution_weight} t={rt:.1f}s")
        save_results(results, output)

    print(f"[exp3] Done. Results: {output}")


def analyze_exp3(results_file: str = OUTPUT_FILE) -> None:
    results = load_or_init_results(results_file)
    if not results:
        print("[exp3] No results to analyze.")
        return

    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    print("\n=== Exp3: Scaling to large n ===")
    print("\n--- Weighted solution quality (weight_recall) by n ---")
    print_comparison_table(results, _ALL_SOLVERS, group_field="n", metric="weight_recall")
    print("\n--- Runtime (s) by n ---")
    print_comparison_table(results, _ALL_SOLVERS, group_field="n", metric="runtime")

    # Plot: runtime vs n (log-log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10.colors

    by_solver_n: Dict = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.solution_weight is not None and r.runtime > 0:
            by_solver_n[r.solver][r.n].append((r.weight_recall or 0, r.runtime))

    for ax_idx, (metric_idx, ylabel, title) in enumerate([
        (0, "Weight recall", "Solution quality vs n"),
        (1, "Runtime (s)", "Runtime vs n (log-log)"),
    ]):
        ax = axes[ax_idx]
        for ci, solver in enumerate(_ALL_SOLVERS):
            if solver not in by_solver_n:
                continue
            ns = sorted(by_solver_n[solver].keys())
            ys = [np.mean([v[metric_idx] for v in by_solver_n[solver][n]]) for n in ns]
            ax.plot(ns, ys, marker="o", label=solver, color=colors[ci % len(colors)])
        ax.set_xlabel("n (graph size)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.set_xscale("log")
        if metric_idx == 1:
            ax.set_yscale("log")

    out_plot = os.path.join(_ONEAL, "benchmark_results", "exp3_scaling.png")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=120)
    print(f"\n  Plot saved: {out_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kamis_path",    default=None)
    parser.add_argument("--output",        default=OUTPUT_FILE)
    parser.add_argument("--no_resume",     action="store_true")
    parser.add_argument("--analyze_only",  action="store_true")
    parser.add_argument("--n_max_python",  type=int, default=2000)
    parser.add_argument("--n_max_cpp",     type=int, default=50000)
    parser.add_argument("--time_limit",    type=float, default=TIME_LIMIT_S)
    parser.add_argument("--overlap_only",  action="store_true",
                        help="Only run overlap region n in {500,1000,2000}")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_exp3(args.output)
    else:
        run_exp3(args.kamis_path, args.output,
                 resume=not args.no_resume,
                 n_max_python=args.n_max_python,
                 n_max_cpp=args.n_max_cpp,
                 time_limit=args.time_limit,
                 overlap_only=args.overlap_only)
        analyze_exp3(args.output)
