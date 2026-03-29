#!/usr/bin/env python3
"""
benchmark_suite.py
==================
Canonical dataset definitions, shared result schema, and WSL-aware
KaMIS binary runner for all 6 experiments.

All experiments import from this module to ensure identical instances,
metrics, and output format.
"""

import os
import sys
import json
import math
import time
import platform
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional, Iterator

import numpy as np
import networkx as nx

# ── Re-export everything from mis_benchmark_combined ──────────────────────────
_ONEAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ONEAL_DIR)
from mis_benchmark_combined import (          # noqa: E402
    MISInstance,
    gen_erdos_renyi_planted,
    gen_multi_clique_core,
    write_metis,
    read_kamis_solution,
    verify_independent_set,
    greedy_min_degree,
    greedy_max_degree_removal,
    simulated_annealing_mis,
    local_search_1_2_swap,
    spectral_mis,
    exact_mis_small,
    KaMISRunner,
    SolverResult,
)

# ── Canonical dataset constants ───────────────────────────────────────────────

FAMILY_A_N      = [100, 200, 500, 1000, 2000]
FAMILY_A_P      = [0.001, 0.01, 0.05, 0.2, 0.5]
FAMILY_A_SEEDS  = [42, 123, 456, 789, 1337]

FAMILY_B_N      = [100, 200, 500]
FAMILY_B_Q      = 3
FAMILY_B_B      = 2
FAMILY_B_P_INTER = 0.3
FAMILY_B_SEEDS  = [42, 123, 456]

LARGE_N         = [5000, 10000, 50000]
LARGE_P         = [0.001, 0.01]
LARGE_SEEDS     = [42, 123]

WEIGHT_LOW      = 1
WEIGHT_HIGH     = 100
TIME_LIMIT_S    = 30.0


# ── Weighted instance wrapper ─────────────────────────────────────────────────

@dataclass
class WeightedMISInstance:
    base: MISInstance
    weights: Dict[int, int]   # node -> integer weight in [WEIGHT_LOW, WEIGHT_HIGH]
    weight_seed: int

    @property
    def instance_id(self) -> str:
        return self.base.instance_id + f"_w{self.weight_seed}"

    @property
    def graph(self) -> nx.Graph:
        return self.base.graph

    @property
    def planted_set(self) -> Set[int]:
        return self.base.planted_set

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def h(self) -> int:
        return self.base.h

    @property
    def family(self) -> str:
        return self.base.family

    @property
    def params(self) -> Dict:
        return self.base.params

    @property
    def seed(self) -> int:
        return self.base.seed

    def total_weight_planted(self) -> int:
        return sum(self.weights[v] for v in self.planted_set)


def add_weights(inst: MISInstance, weight_seed: int = None) -> WeightedMISInstance:
    """Assign integer weights uniform in [WEIGHT_LOW, WEIGHT_HIGH]."""
    if weight_seed is None:
        weight_seed = inst.seed
    rng = np.random.RandomState(weight_seed)
    weights = {v: int(rng.randint(WEIGHT_LOW, WEIGHT_HIGH + 1))
               for v in inst.graph.nodes()}
    return WeightedMISInstance(base=inst, weights=weights, weight_seed=weight_seed)


def write_metis_weighted(inst: WeightedMISInstance, filepath: str) -> None:
    """Write METIS format with node weights (format flag 10)."""
    G = inst.graph
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    assert nodes == list(range(n)), "Nodes must be 0..n-1"
    m = G.number_of_edges()
    with open(filepath, 'w') as f:
        f.write(f"{n} {m} 10\n")   # format flag 10 = node weights only
        for v in range(n):
            w = inst.weights[v]
            neighbors = sorted([u + 1 for u in G.neighbors(v)])
            nbr_str = " ".join(map(str, neighbors))
            f.write(f"{w} {nbr_str}\n" if neighbors else f"{w}\n")


# ── Dataset iterators ─────────────────────────────────────────────────────────

def _h(n: int) -> int:
    return max(1, int(math.floor(math.sqrt(n))))


def iter_family_a(
    n_values: List[int] = None,
    p_values: List[float] = None,
    seeds: List[int] = None,
) -> Iterator[MISInstance]:
    """Yield all Family A Erdos-Renyi planted instances, h = floor(sqrt(n))."""
    n_values = n_values if n_values is not None else FAMILY_A_N
    p_values = p_values if p_values is not None else FAMILY_A_P
    seeds    = seeds    if seeds    is not None else FAMILY_A_SEEDS
    for n in n_values:
        for p in p_values:
            for seed in seeds:
                h = _h(n)
                inst = gen_erdos_renyi_planted(n, h, p, seed)
                if inst.verify():
                    yield inst


def iter_family_b(
    n_values: List[int] = None,
    seeds: List[int] = None,
) -> Iterator[MISInstance]:
    """Yield all Family B multi-clique-core instances, h = floor(sqrt(n))."""
    n_values = n_values if n_values is not None else FAMILY_B_N
    seeds    = seeds    if seeds    is not None else FAMILY_B_SEEDS
    for n in n_values:
        for seed in seeds:
            h = _h(n)
            inst = gen_multi_clique_core(
                n, h,
                q=FAMILY_B_Q, b=FAMILY_B_B,
                p_inter=FAMILY_B_P_INTER, p_cam=0.2,
                seed=seed,
            )
            if inst.verify():
                yield inst


def _gen_large_erdos_renyi_planted(n: int, h: int, p: float,
                                   seed: int) -> MISInstance:
    """Fast O(n+m) planted-IS generator for large graphs.

    Uses nx.fast_gnp_random_graph then removes intra-S edges.
    The slow original (O(n²) loop) is only feasible for n ≤ ~2000.
    """
    rng = np.random.RandomState(seed)
    vertices = list(range(n))
    S = set(rng.choice(vertices, size=h, replace=False).tolist())
    # fast_gnp uses Batagelj-Brandes: O(n + m) time
    G = nx.fast_gnp_random_graph(n, p, seed=seed)
    # Remove edges that fall within S (preserving the planted IS)
    intra = [(u, v) for u, v in list(G.edges()) if u in S and v in S]
    G.remove_edges_from(intra)
    return MISInstance(G, S, n, h, 'erdos_renyi', {'p': p}, seed)


def iter_large_scale(
    n_values: List[int] = None,
    p_values: List[float] = None,
    seeds: List[int] = None,
) -> Iterator[MISInstance]:
    """Yield large-scale instances (for C++ solvers only)."""
    n_values = n_values if n_values is not None else LARGE_N
    p_values = p_values if p_values is not None else LARGE_P
    seeds    = seeds    if seeds    is not None else LARGE_SEEDS
    for n in n_values:
        for p in p_values:
            for seed in seeds:
                h = _h(n)
                inst = _gen_large_erdos_renyi_planted(n, h, p, seed)
                if inst.verify():
                    yield inst


def iter_weighted(instances) -> Iterator[WeightedMISInstance]:
    """Wrap any instance iterator to add integer weights."""
    for inst in instances:
        yield add_weights(inst)


# ── Shared result schema ──────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    experiment: str
    instance_id: str
    family: str
    n: int
    h: int
    params: Dict
    seed: int
    num_edges: int
    solver: str
    solution_size: int
    overlap_with_planted: int
    recall: float
    solution_weight: Optional[int]
    total_weight_planted: Optional[int]
    weight_recall: Optional[float]
    runtime: float
    time_limit: float
    verified: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def make_result(
    experiment: str,
    inst,                        # MISInstance or WeightedMISInstance
    solver: str,
    solution: Set[int],
    runtime: float,
    time_limit: float = TIME_LIMIT_S,
    weights: Dict[int, int] = None,
    notes: str = "",
) -> ExperimentResult:
    """Build an ExperimentResult from raw solver output."""
    G = inst.graph if hasattr(inst, 'graph') else inst.base.graph
    planted = inst.planted_set if hasattr(inst, 'planted_set') else inst.base.planted_set
    h = inst.h if hasattr(inst, 'h') else inst.base.h

    overlap = len(solution & planted)
    recall  = overlap / max(h, 1)
    verified = verify_independent_set(G, solution) if solution else False

    sol_weight  = None
    twp         = None
    w_recall    = None
    if weights:
        sol_weight = sum(weights[v] for v in solution)
        twp        = sum(weights[v] for v in planted)
        w_recall   = sol_weight / max(twp, 1)
    elif isinstance(inst, WeightedMISInstance):
        sol_weight = sum(inst.weights[v] for v in solution)
        twp        = inst.total_weight_planted()
        w_recall   = sol_weight / max(twp, 1)

    return ExperimentResult(
        experiment=experiment,
        instance_id=inst.instance_id,
        family=inst.family,
        n=inst.n,
        h=h,
        params=inst.params,
        seed=inst.seed,
        num_edges=G.number_of_edges(),
        solver=solver,
        solution_size=len(solution),
        overlap_with_planted=overlap,
        recall=recall,
        solution_weight=sol_weight,
        total_weight_planted=twp,
        weight_recall=w_recall,
        runtime=runtime,
        time_limit=time_limit,
        verified=verified,
        notes=notes,
    )


def save_results(results: List[ExperimentResult], filepath: str) -> None:
    """Write results list to JSON file (overwrites)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "schema_version": "1.0",
        "results": [r.to_dict() for r in results],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_or_init_results(filepath: str) -> List[ExperimentResult]:
    """Load existing results; return empty list if file absent."""
    if not os.path.isfile(filepath):
        return []
    with open(filepath) as f:
        data = json.load(f)
    raw = data.get("results", data) if isinstance(data, dict) else data
    results = []
    for d in raw:
        try:
            results.append(ExperimentResult.from_dict(d))
        except Exception:
            pass
    return results


def get_completed_ids(results: List[ExperimentResult], solver: str) -> Set[str]:
    """Return set of instance_ids already completed for a given solver."""
    return {r.instance_id for r in results if r.solver == solver}


# ── WSL-aware binary runner ───────────────────────────────────────────────────

class WSLRunner:
    """
    Runs KaMIS binaries via WSL on Windows, or directly on Linux/macOS.

    On Windows:
      - Converts paths like C:\\Users\\... to /mnt/c/Users/...
      - Prefixes commands with ["wsl"]
    Falls back gracefully (is_available=False) when WSL is not found.
    """

    def __init__(self, deploy_path: str = None):
        self.platform = platform.system()
        self.deploy_path = deploy_path or self._find_deploy()
        self._wsl_ok = self._check_wsl()
        self.is_available = bool(self.deploy_path) and (
            self.platform != "Windows" or self._wsl_ok
        )

    def _find_deploy(self) -> Optional[str]:
        root = os.path.dirname(_ONEAL_DIR)
        for rel in ["deploy", "../deploy", "../../deploy"]:
            p = os.path.normpath(os.path.join(_ONEAL_DIR, rel))
            if os.path.isdir(p):
                return p
        return None

    def _check_wsl(self) -> bool:
        if self.platform != "Windows":
            return False
        try:
            r = subprocess.run(
                ["wsl", "echo", "ok"],
                capture_output=True, text=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _to_wsl_path(self, win_path: str) -> str:
        """Convert absolute Windows path to WSL mount path."""
        win_path = os.path.abspath(win_path)
        drive = win_path[0].lower()
        rest  = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"

    def _build_cmd(
        self,
        binary_name: str,
        graph_file: str,
        output_file: str,
        extra_args: List[str],
        time_limit: float,
        seed: int,
    ) -> List[str]:
        if not self.deploy_path:
            raise RuntimeError("deploy_path not set")
        binary = os.path.join(self.deploy_path, binary_name)
        if self.platform == "Windows":
            binary   = self._to_wsl_path(binary)
            graph_f  = self._to_wsl_path(graph_file)
            output_f = self._to_wsl_path(output_file)
            cmd = ["wsl", binary, graph_f]
        else:
            graph_f  = graph_file
            output_f = output_file
            cmd = [binary, graph_f]
        cmd += [
            f"--time_limit={time_limit}",
            f"--seed={seed}",
            f"--output={output_f}",
            "--console_log",
        ]
        cmd += list(extra_args)
        return cmd

    def run_binary(
        self,
        binary_name: str,
        graph_file: str,
        output_file: str,
        extra_args: List[str] = (),
        time_limit: float = TIME_LIMIT_S,
        seed: int = 0,
    ) -> Tuple[bool, float, str]:
        """Run binary, return (success, elapsed, stderr_snippet)."""
        cmd = self._build_cmd(
            binary_name, graph_file, output_file, extra_args, time_limit, seed
        )
        t0 = time.time()
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=time_limit + 15,
            )
            elapsed = time.time() - t0
            success = os.path.isfile(output_file)
            return success, elapsed, r.stderr[:300]
        except subprocess.TimeoutExpired:
            return False, time_limit, "timeout"
        except Exception as e:
            return False, -1.0, str(e)

    def solve_unweighted(
        self,
        inst: MISInstance,
        binary_name: str,
        time_limit: float = TIME_LIMIT_S,
        extra_args: List[str] = (),
        seed: int = 0,
    ) -> Tuple[Set[int], float]:
        """Write METIS, run binary, read solution. Returns (mis, runtime)."""
        if not self.is_available:
            return set(), -1.0
        with tempfile.TemporaryDirectory() as tmpdir:
            gf = os.path.join(tmpdir, "graph.metis")
            of = os.path.join(tmpdir, "solution.txt")
            write_metis(inst.graph, gf)
            ok, elapsed, _ = self.run_binary(
                binary_name, gf, of, extra_args, time_limit, seed
            )
            if ok:
                return read_kamis_solution(of, inst.n), elapsed
            return set(), elapsed

    def solve_weighted(
        self,
        inst: WeightedMISInstance,
        binary_name: str,
        time_limit: float = TIME_LIMIT_S,
        extra_args: List[str] = (),
        seed: int = 0,
    ) -> Tuple[Set[int], float]:
        """Write weighted METIS, run binary, read solution."""
        if not self.is_available:
            return set(), -1.0
        with tempfile.TemporaryDirectory() as tmpdir:
            gf = os.path.join(tmpdir, "graph.metis")
            of = os.path.join(tmpdir, "solution.txt")
            write_metis_weighted(inst, gf)
            ok, elapsed, _ = self.run_binary(
                binary_name, gf, of, extra_args, time_limit, seed
            )
            if ok:
                return read_kamis_solution(of, inst.n), elapsed
            return set(), elapsed


# ── Utility: analysis helpers ─────────────────────────────────────────────────

def results_to_table(
    results: List[ExperimentResult],
    group_by: str = "solver",
    metric: str = "recall",
) -> Dict:
    """
    Aggregate results by group_by field, compute mean/std of metric.
    Returns dict: group_value -> {"mean": float, "std": float, "n": int}
    """
    from collections import defaultdict
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        key = getattr(r, group_by, "unknown")
        val = getattr(r, metric, None)
        if val is not None:
            groups[key].append(float(val))
    out = {}
    for k, vals in groups.items():
        arr = np.array(vals)
        out[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}
    return out


def print_comparison_table(
    results: List[ExperimentResult],
    solvers: List[str],
    group_field: str = "n",
    metric: str = "recall",
) -> None:
    """Print a text table: rows = group_field values, cols = solvers."""
    from collections import defaultdict
    # group -> solver -> [values]
    data: Dict = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.solver in solvers:
            gv = getattr(r, group_field, "?")
            mv = getattr(r, metric, None)
            if mv is not None:
                data[gv][r.solver].append(float(mv))

    header = f"{'':>6s}" + "".join(f"{s:>14s}" for s in solvers)
    print(header)
    print("-" * len(header))
    for gv in sorted(data.keys()):
        row = f"{str(gv):>6s}"
        for s in solvers:
            vals = data[gv][s]
            if vals:
                row += f"{np.mean(vals):>14.3f}"
            else:
                row += f"{'n/a':>14s}"
        print(row)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("benchmark_suite.py self-test")
    print(f"  FAMILY_A: {len(FAMILY_A_N)}n × {len(FAMILY_A_P)}p × {len(FAMILY_A_SEEDS)}seeds = "
          f"{len(FAMILY_A_N)*len(FAMILY_A_P)*len(FAMILY_A_SEEDS)} instances")
    print(f"  FAMILY_B: {len(FAMILY_B_N)}n × {len(FAMILY_B_SEEDS)}seeds = "
          f"{len(FAMILY_B_N)*len(FAMILY_B_SEEDS)} instances")
    print(f"  LARGE:    {len(LARGE_N)}n × {len(LARGE_P)}p × {len(LARGE_SEEDS)}seeds = "
          f"{len(LARGE_N)*len(LARGE_P)*len(LARGE_SEEDS)} instances")

    # Generate first 3 Family A instances and show them
    for i, inst in enumerate(iter_family_a(n_values=[100], p_values=[0.05], seeds=[42])):
        winst = add_weights(inst)
        print(f"  {inst.instance_id}: n={inst.n}, m={inst.graph.number_of_edges()}, "
              f"h={inst.h}, w_planted={winst.total_weight_planted()}")
        if i >= 2:
            break

    # Test WSLRunner availability
    runner = WSLRunner()
    print(f"  WSLRunner: available={runner.is_available}, deploy={runner.deploy_path}")

    print("  OK")
