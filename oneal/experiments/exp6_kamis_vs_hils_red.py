#!/usr/bin/env python3
"""
exp6_kamis_vs_hils_red.py
=========================
Task 6: Compare KaMIS redumis vs Python Red+HILS.

KaMIS_redumis:  evolutionary algorithm on a kernelised graph (C++ binary).
Python_RedHILS: Python Red+HILS with unit weights (comparable to unweighted MIS).

Both run on the canonical dataset with 30s time limit.
Python Red+HILS limited to n ≤ 2000.

Analysis: quality gap = redumis_size - red_hils_size, crossover n, per p-value.

Results: oneal/benchmark_results/exp6_results.json
"""

import os, sys, argparse, time
from itertools import chain

_ONEAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ONEAL)

from benchmark_suite import (
    FAMILY_A_N, TIME_LIMIT_S,
    iter_family_a, iter_family_b,
    WSLRunner, ExperimentResult, make_result,
    save_results, load_or_init_results, get_completed_ids,
    print_comparison_table,
)
from hils import red_hils_weighted, HilsConfig

EXPERIMENT  = "exp6_kamis_vs_hils_red"
OUTPUT_FILE = os.path.join(_ONEAL, "benchmark_results", "exp6_results.json")

_SOLVERS = ["KaMIS_redumis", "Python_RedHILS"]


def run_exp6(
    kamis_path: str = None,
    output: str = OUTPUT_FILE,
    resume: bool = True,
    n_max: int = 2000,
    time_limit: float = TIME_LIMIT_S,
) -> None:
    runner  = WSLRunner(kamis_path)
    results = load_or_init_results(output) if resume else []
    done    = {s: get_completed_ids(results, s) for s in _SOLVERS}

    cfg = HilsConfig(max_iter=2_000_000, time_limit=time_limit, seed=42)

    instances = list(chain(
        iter_family_a(n_values=[n for n in FAMILY_A_N if n <= n_max]),
        iter_family_b(),
    ))
    print(f"[exp6] {len(instances)} instances × {len(_SOLVERS)} solvers")

    for inst in instances:
        G = inst.graph

        # ── KaMIS redumis ─────────────────────────────────────────────────────
        sname = "KaMIS_redumis"
        if inst.instance_id not in done[sname]:
            if runner.is_available:
                sol, rt = runner.solve_unweighted(inst, "redumis", time_limit)
            else:
                sol, rt = set(), -1.0
            note = "wsl_unavailable" if not runner.is_available else ""
            r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit, notes=note)
            results.append(r); done[sname].add(inst.instance_id)
            print(f"  {sname:22s} {inst.instance_id[:40]:40s} "
                  f"size={r.solution_size} recall={r.recall:.3f} t={rt:.1f}s")

        # ── Python Red+HILS (unit weights) ────────────────────────────────────
        sname = "Python_RedHILS"
        if inst.instance_id not in done[sname]:
            t0 = time.time()
            w1  = {v: 1.0 for v in G.nodes()}
            sol, _ = red_hils_weighted(G, w1, cfg)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit)
            results.append(r); done[sname].add(inst.instance_id)
            print(f"  {sname:22s} {inst.instance_id[:40]:40s} "
                  f"size={r.solution_size} recall={r.recall:.3f} t={rt:.1f}s")

        save_results(results, output)

    print(f"[exp6] Done. Results: {output}")


def analyze_exp6(results_file: str = OUTPUT_FILE) -> None:
    results = load_or_init_results(results_file)
    if not results:
        print("[exp6] No results to analyze.")
        return

    import numpy as np
    from collections import defaultdict

    print("\n=== Exp6: KaMIS redumis vs Python Red+HILS ===")
    print("\n--- Solution size by n ---")
    print_comparison_table(results, _SOLVERS, group_field="n", metric="solution_size")
    print("\n--- Recall by n ---")
    print_comparison_table(results, _SOLVERS, group_field="n", metric="recall")
    print("\n--- Runtime by n ---")
    print_comparison_table(results, _SOLVERS, group_field="n", metric="runtime")

    # Quality gap per p value
    print("\n--- Quality gap (redumis - RedHILS size) by (n, p) ---")
    by_id: Dict = defaultdict(dict)
    for r in results:
        if r.solver in _SOLVERS and r.solution_size > 0:
            by_id[r.instance_id][r.solver] = (r.solution_size, r.n, r.params.get("p"))

    gap_by_np: Dict = defaultdict(list)
    for iid, sv in by_id.items():
        if "KaMIS_redumis" in sv and "Python_RedHILS" in sv:
            kamis_sz, n, p = sv["KaMIS_redumis"]
            hils_sz, _, _ = sv["Python_RedHILS"]
            gap_by_np[(n, p)].append(kamis_sz - hils_sz)

    crossover_n: Dict = defaultdict(list)
    for (n, p), gaps in sorted(gap_by_np.items()):
        avg_gap = np.mean(gaps)
        print(f"  n={n:5d} p={str(p):6s}: avg_gap={avg_gap:+.1f}  "
              f"redumis_wins={sum(g>0 for g in gaps)}/{len(gaps)}")
        if avg_gap <= 0.5:
            crossover_n[p].append(n)

    print("\n--- Crossover n (where redumis gap <= 0.5 nodes) by p ---")
    for p, ns in sorted(crossover_n.items()):
        print(f"  p={p}: crossover at n <= {max(ns) if ns else '—'}")


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
        analyze_exp6(args.output)
    else:
        run_exp6(args.kamis_path, args.output,
                 resume=not args.no_resume,
                 n_max=args.n_max,
                 time_limit=args.time_limit)
        analyze_exp6(args.output)
