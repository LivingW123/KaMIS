#!/usr/bin/env python3
"""
exp5_arw_vs_hils.py
===================
Task 5: Compare ARW (online_mis binary) vs Python HILS on unweighted instances.

ARW = Andrade, Resende, Werneck ILS for maximum independent set.
      Implemented in KaMIS as online_mis (lib/mis/ils/ils.h).
      Note: in unweighted mode, HILS N1 is inactive; only N2 (1,2-swap) fires,
      making Python HILS equivalent to the ARW (1,2)-swap ILS.

Both run 30s time limit.
Python HILS limited to n ≤ 2000; ARW also runs on large-scale instances.

Results: oneal/benchmark_results/exp5_results.json
"""

import os, sys, argparse, time
from itertools import chain

_ONEAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ONEAL)

from benchmark_suite import (
    FAMILY_A_N, TIME_LIMIT_S,
    iter_family_a, iter_family_b, iter_large_scale,
    WSLRunner, ExperimentResult, make_result,
    save_results, load_or_init_results, get_completed_ids,
    print_comparison_table,
)
from hils import hils_unweighted, HilsConfig

EXPERIMENT  = "exp5_arw_vs_hils"
OUTPUT_FILE = os.path.join(_ONEAL, "benchmark_results", "exp5_results.json")

_SOLVERS = ["ARW_online_mis", "Python_HILS_unw"]


def run_exp5(
    kamis_path: str = None,
    output: str = OUTPUT_FILE,
    resume: bool = True,
    n_max_python: int = 2000,
    time_limit: float = TIME_LIMIT_S,
) -> None:
    runner  = WSLRunner(kamis_path)
    results = load_or_init_results(output) if resume else []
    done    = {s: get_completed_ids(results, s) for s in _SOLVERS}

    cfg = HilsConfig(max_iter=2_000_000, time_limit=time_limit, seed=42)

    # Family A + Family B + large-scale (ARW only for large)
    small_inst = list(chain(
        iter_family_a(n_values=[n for n in FAMILY_A_N if n <= n_max_python]),
        iter_family_b(),
    ))
    # Large instances are kept as a lazy iterator (graphs are generated on demand)
    # to avoid generating all large graphs upfront (O(n²) for n=50000 is infeasible).
    large_inst = iter_large_scale()

    print(f"[exp5] Small instances: {len(small_inst)}, large: lazy iterator (ARW only)")

    # ── Small instances: both ARW and Python HILS ─────────────────────────────
    for inst in small_inst:
        # ARW (online_mis binary)
        sname = "ARW_online_mis"
        if inst.instance_id not in done[sname]:
            if runner.is_available:
                sol, rt = runner.solve_unweighted(inst, "online_mis", time_limit)
            else:
                sol, rt = set(), -1.0
            note = "wsl_unavailable" if not runner.is_available else ""
            r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit, notes=note)
            results.append(r); done[sname].add(inst.instance_id)
            print(f"  {sname:22s} {inst.instance_id[:40]:40s} "
                  f"size={r.solution_size} recall={r.recall:.3f} t={rt:.1f}s")

        # Python HILS (unweighted)
        sname = "Python_HILS_unw"
        if inst.instance_id not in done[sname]:
            t0 = time.time()
            sol, _ = hils_unweighted(inst.graph, cfg)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit)
            results.append(r); done[sname].add(inst.instance_id)
            print(f"  {sname:22s} {inst.instance_id[:40]:40s} "
                  f"size={r.solution_size} recall={r.recall:.3f} t={rt:.1f}s")

        save_results(results, output)

    # ── Large instances: ARW only ─────────────────────────────────────────────
    if runner.is_available:
        for inst in large_inst:
            sname = "ARW_online_mis"
            if inst.instance_id not in done[sname]:
                sol, rt = runner.solve_unweighted(inst, "online_mis", time_limit)
                r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit,
                                notes="large_scale")
                results.append(r); done[sname].add(inst.instance_id)
                print(f"  {sname:22s} {inst.instance_id[:40]:40s} "
                      f"n={inst.n} size={r.solution_size} t={rt:.1f}s")
            save_results(results, output)

    print(f"[exp5] Done. Results: {output}")


def analyze_exp5(results_file: str = OUTPUT_FILE) -> None:
    results = load_or_init_results(results_file)
    if not results:
        print("[exp5] No results to analyze.")
        return

    import numpy as np
    from collections import defaultdict

    print("\n=== Exp5: ARW vs Python HILS (unweighted) ===")
    print("\n--- Recall by n ---")
    print_comparison_table(results, _SOLVERS, group_field="n", metric="recall")
    print("\n--- Runtime by n ---")
    print_comparison_table(results, _SOLVERS, group_field="n", metric="runtime")

    # Quality gap per instance
    by_id: Dict = defaultdict(dict)
    for r in results:
        if r.solver in _SOLVERS and r.solution_size > 0:
            by_id[r.instance_id][r.solver] = (r.solution_size, r.recall, r.n)

    gaps = []
    for iid, sv in by_id.items():
        if "ARW_online_mis" in sv and "Python_HILS_unw" in sv:
            arw_sz, arw_rec, n = sv["ARW_online_mis"]
            hils_sz, hils_rec, _ = sv["Python_HILS_unw"]
            if arw_sz > 0:
                gaps.append((n, arw_sz, hils_sz, arw_sz - hils_sz))

    if gaps:
        gaps.sort()
        print("\n--- ARW size - HILS size gap (positive = ARW larger) ---")
        gap_by_n: Dict = defaultdict(list)
        for n, a, h, g in gaps:
            gap_by_n[n].append(g)
        for n in sorted(gap_by_n):
            vals = gap_by_n[n]
            print(f"  n={n:5d}: avg_gap={np.mean(vals):+.1f}  "
                  f"ARW_wins={sum(g>0 for g in vals)}/{len(vals)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kamis_path",    default=None)
    parser.add_argument("--output",        default=OUTPUT_FILE)
    parser.add_argument("--no_resume",     action="store_true")
    parser.add_argument("--analyze_only",  action="store_true")
    parser.add_argument("--n_max_python",  type=int, default=2000)
    parser.add_argument("--time_limit",    type=float, default=TIME_LIMIT_S)
    args = parser.parse_args()

    if args.analyze_only:
        analyze_exp5(args.output)
    else:
        run_exp5(args.kamis_path, args.output,
                 resume=not args.no_resume,
                 n_max_python=args.n_max_python,
                 time_limit=args.time_limit)
        analyze_exp5(args.output)
