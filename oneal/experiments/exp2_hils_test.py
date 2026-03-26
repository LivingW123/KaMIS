#!/usr/bin/env python3
"""
exp2_hils_test.py
=================
Task 2: Validate the Python HILS and Red+HILS implementation.

Compares (all unweighted for fair comparison with existing baselines):
  - Greedy_MinDeg         : greedy_min_degree (from mis_benchmark_combined)
  - LocalSearch_1_2       : local_search_1_2_swap
  - SimulatedAnnealing    : simulated_annealing_mis
  - Python_HILS           : hils_unweighted (this work)
  - Python_RedHILS        : red_hils_weighted with weights=1 (this work)

Also runs weighted variants (Python_HILS_W, Python_RedHILS_W) on same instances.

n_max default: 500 (Python HILS is slow; hits 30s time limit for n≥500)

Results: oneal/benchmark_results/exp2_results.json
"""

import os, sys, argparse, time
from itertools import chain

_ONEAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ONEAL)

from benchmark_suite import (
    FAMILY_A_N, TIME_LIMIT_S,
    iter_family_a, iter_family_b, iter_weighted,
    WSLRunner, ExperimentResult, make_result,
    save_results, load_or_init_results, get_completed_ids,
    print_comparison_table, add_weights,
)
from mis_benchmark_combined import (
    greedy_min_degree, local_search_1_2_swap,
    simulated_annealing_mis, verify_independent_set,
)
from hils import hils_unweighted, red_hils_weighted, HilsConfig, correctness_check

EXPERIMENT  = "exp2_hils_test"
OUTPUT_FILE = os.path.join(_ONEAL, "benchmark_results", "exp2_results.json")

_UNWEIGHTED_SOLVERS = {
    "Greedy_MinDeg": lambda G, inst: (greedy_min_degree(G), None),
    "LocalSearch_1_2": lambda G, inst: (local_search_1_2_swap(G, greedy_min_degree(G)), None),
    "SimulatedAnnealing": lambda G, inst: (simulated_annealing_mis(G, seed=inst.seed), None),
}

def _make_hils_config(time_limit):
    return HilsConfig(max_iter=2_000_000, time_limit=time_limit, seed=42)


def run_exp2(
    output: str = OUTPUT_FILE,
    resume: bool = True,
    n_max: int = 500,
    time_limit: float = TIME_LIMIT_S,
) -> None:
    # Run correctness check first
    print("[exp2] Running correctness checks...")
    try:
        ok = correctness_check()
        print(f"[exp2] Correctness check: {'PASS' if ok else 'FAIL'}")
    except AssertionError as e:
        print(f"[exp2] Correctness check FAILED: {e}")
        return

    results = load_or_init_results(output) if resume else []

    # Solvers to run: unweighted + weighted HILS variants
    all_solvers = list(_UNWEIGHTED_SOLVERS.keys()) + [
        "Python_HILS", "Python_RedHILS",
        "Python_HILS_W", "Python_RedHILS_W",
    ]
    done = {s: get_completed_ids(results, s) for s in all_solvers}

    instances = list(chain(
        iter_family_a(n_values=[n for n in FAMILY_A_N if n <= n_max]),
        iter_family_b(n_values=[n for n in [100, 200, 500] if n <= n_max]),
    ))
    print(f"[exp2] {len(instances)} instances × {len(all_solvers)} solvers "
          f"(n_max={n_max}, t={time_limit}s)")

    for inst in instances:
        G = inst.graph
        winst = add_weights(inst)
        cfg = _make_hils_config(time_limit)

        # ── Unweighted baselines ──────────────────────────────────────────────
        for sname, fn in _UNWEIGHTED_SOLVERS.items():
            if inst.instance_id in done[sname]:
                continue
            t0 = time.time()
            sol, _ = fn(G, inst)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit)
            results.append(r)
            done[sname].add(inst.instance_id)
            print(f"  {sname:22s} {inst.instance_id[:40]:40s} "
                  f"size={r.solution_size} recall={r.recall:.3f} t={rt:.2f}s")

        # ── Python HILS (unweighted) ──────────────────────────────────────────
        if inst.instance_id not in done["Python_HILS"]:
            t0 = time.time()
            sol, _ = hils_unweighted(G, cfg)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, inst, "Python_HILS", sol, rt, time_limit)
            results.append(r)
            done["Python_HILS"].add(inst.instance_id)
            print(f"  {'Python_HILS':22s} {inst.instance_id[:40]:40s} "
                  f"size={r.solution_size} recall={r.recall:.3f} t={rt:.2f}s")

        # ── Python Red+HILS (unweighted via unit weights) ─────────────────────
        if inst.instance_id not in done["Python_RedHILS"]:
            t0 = time.time()
            w1 = {v: 1.0 for v in G.nodes()}
            sol, _ = red_hils_weighted(G, w1, cfg)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, inst, "Python_RedHILS", sol, rt, time_limit)
            results.append(r)
            done["Python_RedHILS"].add(inst.instance_id)
            print(f"  {'Python_RedHILS':22s} {inst.instance_id[:40]:40s} "
                  f"size={r.solution_size} recall={r.recall:.3f} t={rt:.2f}s")

        # ── Weighted HILS ─────────────────────────────────────────────────────
        from hils import hils_weighted
        if winst.instance_id not in done["Python_HILS_W"]:
            t0 = time.time()
            sol, w_val = hils_weighted(G, winst.weights, cfg)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, winst, "Python_HILS_W", sol, rt, time_limit)
            results.append(r)
            done["Python_HILS_W"].add(winst.instance_id)
            print(f"  {'Python_HILS_W':22s} {winst.instance_id[:36]:36s} "
                  f"w={r.solution_weight} recall={r.recall:.3f} t={rt:.2f}s")

        # ── Weighted Red+HILS ─────────────────────────────────────────────────
        if winst.instance_id not in done["Python_RedHILS_W"]:
            t0 = time.time()
            sol, w_val = red_hils_weighted(G, winst.weights, cfg)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, winst, "Python_RedHILS_W", sol, rt, time_limit)
            results.append(r)
            done["Python_RedHILS_W"].add(winst.instance_id)
            print(f"  {'Python_RedHILS_W':22s} {winst.instance_id[:36]:36s} "
                  f"w={r.solution_weight} recall={r.recall:.3f} t={rt:.2f}s")

        save_results(results, output)

    print(f"[exp2] Done. Results: {output}")


def analyze_exp2(results_file: str = OUTPUT_FILE) -> None:
    results = load_or_init_results(results_file)
    if not results:
        print("[exp2] No results to analyze.")
        return

    unw_solvers = ["Greedy_MinDeg", "LocalSearch_1_2", "SimulatedAnnealing",
                   "Python_HILS", "Python_RedHILS"]
    unw_results = [r for r in results if r.solver in unw_solvers]

    print("\n=== Exp2: HILS Implementation Validation ===")
    print("\n--- Recall by n (unweighted) ---")
    print_comparison_table(unw_results, unw_solvers, group_field="n", metric="recall")
    print("\n--- Solution size by n (unweighted) ---")
    print_comparison_table(unw_results, unw_solvers, group_field="n", metric="solution_size")
    print("\n--- Runtime by n (unweighted) ---")
    print_comparison_table(unw_results, unw_solvers, group_field="n", metric="runtime")

    # HILS vs baselines: does HILS beat LocalSearch?
    import numpy as np
    from collections import defaultdict
    hils_better = 0; ls_better = 0; tie = 0
    by_id: Dict = defaultdict(dict)
    for r in unw_results:
        by_id[r.instance_id][r.solver] = r.solution_size
    for iid, sv in by_id.items():
        if "Python_HILS" in sv and "LocalSearch_1_2" in sv:
            if sv["Python_HILS"] > sv["LocalSearch_1_2"]:
                hils_better += 1
            elif sv["Python_HILS"] < sv["LocalSearch_1_2"]:
                ls_better += 1
            else:
                tie += 1
    print(f"\n--- HILS vs LocalSearch_1_2 ---")
    print(f"  HILS better: {hils_better},  LS better: {ls_better},  tie: {tie}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",       default=OUTPUT_FILE)
    parser.add_argument("--no_resume",    action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--n_max",        type=int, default=500)
    parser.add_argument("--time_limit",   type=float, default=TIME_LIMIT_S)
    args = parser.parse_args()

    if args.analyze_only:
        analyze_exp2(args.output)
    else:
        run_exp2(args.output,
                 resume=not args.no_resume,
                 n_max=args.n_max,
                 time_limit=args.time_limit)
        analyze_exp2(args.output)
