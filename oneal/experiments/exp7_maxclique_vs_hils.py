#!/usr/bin/env python3
"""
exp7_maxclique_vs_hils.py
=========================
Task 7: Compare MaxCLQ (max clique on complement) vs HILS+reduction
         and KaMIS solvers on dense graphs.

Dense graphs have high edge probability (p >= 0.2), making MIS hard for
local search. The complement of a dense graph is sparse, so MaxCLQ
may find exact/better solutions via branch-and-bound with MaxSAT pruning.

Solvers compared:
  - MaxCLQ_complement:  max_clique binary with --complement flag
  - Python_RedHILS:     Python Red+HILS with unit weights
  - KaMIS_redumis:      KaMIS evolutionary with full kernelization
  - KaMIS_online_mis:   ARW ILS with online reductions

Focuses on small-to-medium dense instances where exact B&B is feasible.

Results: oneal/benchmark_results/exp7_results.json
"""

import os, sys, time, argparse, tempfile, platform, subprocess
from itertools import chain

_ONEAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ONEAL)

from benchmark_suite import (
    TIME_LIMIT_S,
    iter_family_a,
    WSLRunner, ExperimentResult, make_result,
    save_results, load_or_init_results, get_completed_ids,
    print_comparison_table, write_metis, read_kamis_solution,
    verify_independent_set,
)
from hils import red_hils_weighted, HilsConfig

EXPERIMENT  = "exp7_maxclique_vs_hils"
OUTPUT_FILE = os.path.join(_ONEAL, "benchmark_results", "exp7_results.json")

# Dense graph focus: high p values, small-to-medium n (B&B feasible)
DENSE_N     = [100, 200, 500]
DENSE_P     = [0.2, 0.5]
DENSE_SEEDS = [42, 123, 456, 789, 1337]

_SOLVERS = ["MaxCLQ_complement", "Python_RedHILS", "KaMIS_redumis", "KaMIS_online_mis"]


class MaxCLQRunner:
    """Run the max_clique binary via WSL (or directly on Linux)."""

    def __init__(self, deploy_path: str):
        self.deploy_path = deploy_path
        self.is_windows = platform.system() == "Windows"

    def _to_wsl_path(self, win_path: str) -> str:
        win_path = os.path.abspath(win_path)
        drive = win_path[0].lower()
        rest  = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"

    def solve(self, graph_file: str, n: int, time_limit: float = 60.0):
        """Run max_clique --complement, return (solution_set, runtime)."""
        binary = os.path.join(self.deploy_path, "max_clique")
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "solution.txt")

            if self.is_windows:
                bin_wsl   = self._to_wsl_path(binary)
                graph_wsl = self._to_wsl_path(graph_file)
                out_wsl   = self._to_wsl_path(outfile)
                cmd = ["wsl", bin_wsl, graph_wsl, "--complement",
                       f"--time_limit={time_limit}",
                       f"--output", out_wsl]
            else:
                cmd = [binary, graph_file, "--complement",
                       f"--time_limit={time_limit}",
                       f"--output", outfile]

            env = os.environ.copy()
            env["MSYS_NO_PATHCONV"] = "1"

            t0 = time.time()
            try:
                r = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=time_limit + 10, env=env,
                )
                elapsed = time.time() - t0

                if os.path.isfile(outfile):
                    sol = read_kamis_solution(outfile, n)
                    return sol, elapsed
                else:
                    # Parse from stdout as fallback
                    return self._parse_stdout(r.stdout, r.stderr), elapsed
            except subprocess.TimeoutExpired:
                return set(), time_limit
            except Exception as e:
                print(f"    MaxCLQ error: {e}")
                return set(), -1.0

    def _parse_stdout(self, stdout, stderr):
        """Parse vertex list from stdout if output file wasn't written."""
        combined = stdout + "\n" + stderr
        for line in combined.split("\n"):
            if "Vertices (1-indexed):" in line:
                parts = line.split(":")[-1].strip().split()
                return {int(v) - 1 for v in parts if v.strip()}
        return set()


def run_exp7(
    kamis_path: str = None,
    output: str = OUTPUT_FILE,
    resume: bool = True,
    time_limit: float = TIME_LIMIT_S,
) -> None:
    runner  = WSLRunner(kamis_path)
    clq_runner = MaxCLQRunner(runner.deploy_path)
    results = load_or_init_results(output) if resume else []
    done    = {s: get_completed_ids(results, s) for s in _SOLVERS}

    cfg = HilsConfig(max_iter=2_000_000, time_limit=time_limit, seed=42)

    instances = list(iter_family_a(
        n_values=DENSE_N,
        p_values=DENSE_P,
        seeds=DENSE_SEEDS,
    ))
    print(f"[exp7] {len(instances)} dense instances × {len(_SOLVERS)} solvers")
    print(f"       n ∈ {DENSE_N}, p ∈ {DENSE_P}, time_limit={time_limit}s")

    for inst in instances:
        G = inst.graph
        n = inst.n
        p = inst.params.get("p", "?")
        tag = f"n={n:4d} p={p}"

        with tempfile.TemporaryDirectory() as tmpdir:
            gf = os.path.join(tmpdir, "graph.metis")
            write_metis(G, gf)

            # ── MaxCLQ (complement) ──────────────────────────────────────
            sname = "MaxCLQ_complement"
            if inst.instance_id not in done[sname]:
                sol, rt = clq_runner.solve(gf, n, time_limit)
                verified = verify_independent_set(G, sol) if sol else False
                note = "" if verified else "INVALID" if sol else "no_solution"
                r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit,
                                notes=note)
                results.append(r); done[sname].add(inst.instance_id)
                print(f"  {sname:22s} {tag} size={r.solution_size:3d} "
                      f"recall={r.recall:.3f} t={rt:6.1f}s {'✓' if verified else '✗'}")

            # ── KaMIS redumis ────────────────────────────────────────────
            sname = "KaMIS_redumis"
            if inst.instance_id not in done[sname]:
                if runner.is_available:
                    sol, rt = runner.solve_unweighted(inst, "redumis", time_limit)
                else:
                    sol, rt = set(), -1.0
                r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit)
                results.append(r); done[sname].add(inst.instance_id)
                print(f"  {sname:22s} {tag} size={r.solution_size:3d} "
                      f"recall={r.recall:.3f} t={rt:6.1f}s")

            # ── KaMIS online_mis (ARW) ───────────────────────────────────
            sname = "KaMIS_online_mis"
            if inst.instance_id not in done[sname]:
                if runner.is_available:
                    sol, rt = runner.solve_unweighted(inst, "online_mis", time_limit)
                else:
                    sol, rt = set(), -1.0
                r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit)
                results.append(r); done[sname].add(inst.instance_id)
                print(f"  {sname:22s} {tag} size={r.solution_size:3d} "
                      f"recall={r.recall:.3f} t={rt:6.1f}s")

        # ── Python Red+HILS (outside tmpdir so graph doesn't need to persist)
        sname = "Python_RedHILS"
        if inst.instance_id not in done[sname]:
            t0 = time.time()
            w1  = {v: 1.0 for v in G.nodes()}
            sol, _ = red_hils_weighted(G, w1, cfg)
            rt = time.time() - t0
            r = make_result(EXPERIMENT, inst, sname, sol, rt, time_limit)
            results.append(r); done[sname].add(inst.instance_id)
            print(f"  {sname:22s} {tag} size={r.solution_size:3d} "
                  f"recall={r.recall:.3f} t={rt:6.1f}s")

        save_results(results, output)

    print(f"\n[exp7] Done. Results: {output}")


def analyze_exp7(results_file: str = OUTPUT_FILE) -> None:
    results = load_or_init_results(results_file)
    if not results:
        print("[exp7] No results to analyze.")
        return

    import numpy as np
    from collections import defaultdict

    print("\n" + "=" * 70)
    print("  Exp7: MaxCLQ vs HILS+Reduction on Dense Graphs")
    print("=" * 70)

    print("\n--- Solution size by n ---")
    print_comparison_table(results, _SOLVERS, group_field="n",
                           metric="solution_size")

    print("\n--- Recall (overlap with planted IS / h) by n ---")
    print_comparison_table(results, _SOLVERS, group_field="n",
                           metric="recall")

    print("\n--- Runtime (seconds) by n ---")
    print_comparison_table(results, _SOLVERS, group_field="n",
                           metric="runtime")

    # Per (n, p) breakdown
    print("\n--- Solution size by (n, p) ---")
    by_np = defaultdict(lambda: defaultdict(list))
    for r in results:
        p = r.params.get("p")
        key = f"n={r.n},p={p}"
        by_np[key][r.solver].append(r.solution_size)

    header = f"{'':>16s}" + "".join(f"{s:>20s}" for s in _SOLVERS)
    print(header)
    print("-" * len(header))
    for key in sorted(by_np.keys()):
        row = f"{key:>16s}"
        for s in _SOLVERS:
            vals = by_np[key][s]
            if vals:
                import numpy as np
                row += f"{np.mean(vals):>16.1f}±{np.std(vals):.1f}"
            else:
                row += f"{'n/a':>20s}"
        print(row)

    # Win/tie/loss analysis
    print("\n--- Head-to-head: MaxCLQ vs each solver ---")
    by_id = defaultdict(dict)
    for r in results:
        if r.solution_size > 0:
            by_id[r.instance_id][r.solver] = r.solution_size

    for opponent in ["KaMIS_redumis", "KaMIS_online_mis", "Python_RedHILS"]:
        wins, ties, losses = 0, 0, 0
        for iid, sv in by_id.items():
            if "MaxCLQ_complement" in sv and opponent in sv:
                mc = sv["MaxCLQ_complement"]
                op = sv[opponent]
                if mc > op: wins += 1
                elif mc == op: ties += 1
                else: losses += 1
        total = wins + ties + losses
        if total > 0:
            print(f"  vs {opponent:22s}: "
                  f"MaxCLQ wins={wins} ties={ties} losses={losses} "
                  f"({100*wins/total:.0f}%/{100*ties/total:.0f}%/{100*losses/total:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kamis_path",   default=None)
    parser.add_argument("--output",       default=OUTPUT_FILE)
    parser.add_argument("--no_resume",    action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--time_limit",   type=float, default=TIME_LIMIT_S)
    args = parser.parse_args()

    if args.analyze_only:
        analyze_exp7(args.output)
    else:
        run_exp7(args.kamis_path, args.output,
                 resume=not args.no_resume,
                 time_limit=args.time_limit)
        analyze_exp7(args.output)
