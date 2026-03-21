#!/usr/bin/env python3
"""
Combined MIS Benchmarking Suite
================================
Merges configs from mis_benchmark.py and mis_benchmark2.py.
Runs all solvers across a wider range of n values and outputs
a summary to combined_run.txt.

Usage:
    python mis_benchmark_combined.py                    # quick test
    python mis_benchmark_combined.py --full             # full benchmark
    python mis_benchmark_combined.py --kamis_path PATH  # specify KaMIS binary location
"""

import numpy as np
import networkx as nx
import subprocess
import tempfile
import os
import time
import json
import argparse
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from math import comb


# ============================================================
# SECTION 1: Graph I/O (METIS format for KaMIS)
# ============================================================

def write_metis(G: nx.Graph, filepath: str):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    assert nodes == list(range(n)), "Nodes must be 0..n-1"
    m = G.number_of_edges()
    with open(filepath, 'w') as f:
        f.write(f"{n} {m}\n")
        for v in range(n):
            neighbors = sorted([u + 1 for u in G.neighbors(v)])
            f.write(" ".join(map(str, neighbors)) + "\n")


def read_kamis_solution(filepath: str, n: int) -> Set[int]:
    mis = set()
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if int(line.strip()) == 1:
                mis.add(i)
    return mis


# ============================================================
# SECTION 2: Instance Generation
# ============================================================

@dataclass
class MISInstance:
    graph: nx.Graph
    planted_set: Set[int]
    n: int
    h: int
    family: str
    params: Dict
    seed: int

    def verify(self) -> bool:
        for u in self.planted_set:
            for v in self.planted_set:
                if u != v and self.graph.has_edge(u, v):
                    return False
        return True

    @property
    def instance_id(self) -> str:
        if self.family == 'erdos_renyi':
            return f"{self.family}_n{self.n}_h{self.h}_p{self.params['p']}_s{self.seed}"
        elif self.family == 'multi_clique_core':
            return (
                f"{self.family}_n{self.n}_h{self.h}"
                f"_q{self.params['q']}_b{self.params['b']}_s{self.seed}"
            )
        return f"{self.family}_n{self.n}_h{self.h}_s{self.seed}"


def gen_erdos_renyi_planted(n: int, h: int, p: float = 0.5,
                            seed: int = 42) -> MISInstance:
    rng = np.random.RandomState(seed)
    vertices = list(range(n))
    S = set(rng.choice(vertices, size=h, replace=False).tolist())
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if i in S and j in S:
                continue
            if rng.random() < p:
                G.add_edge(i, j)
    return MISInstance(G, S, n, h, 'erdos_renyi', {'p': p}, seed)


def gen_multi_clique_core(n: int, h: int, q: int = 3, b: int = 2,
                          p_inter: float = 0.5, p_cam: float = 0.3,
                          seed: int = 42) -> MISInstance:
    rng = np.random.RandomState(seed)
    assert q * b <= h <= n
    S = set(range(h))
    R = list(range(h, n))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    block_size = max(1, len(R) // q)
    blocks = []
    for j in range(q):
        start = j * block_size
        end = start + block_size if j < q - 1 else len(R)
        block = R[start:end]
        blocks.append(block)
        for a in range(len(block)):
            for bb in range(a + 1, len(block)):
                G.add_edge(block[a], block[bb])
    for a in range(q):
        for bb in range(a + 1, q):
            for u in blocks[a]:
                for v in blocks[bb]:
                    if rng.random() < p_inter:
                        G.add_edge(u, v)
    S_list = sorted(S)
    for j in range(q):
        B_j = S_list[j * b: (j + 1) * b]
        for u in blocks[j]:
            for v in B_j:
                G.add_edge(u, v)
    for j in range(q):
        B_j = set(S_list[j * b: (j + 1) * b])
        S_minus_Bj = S - B_j
        for u in blocks[j]:
            for v in S_minus_Bj:
                if rng.random() < p_cam:
                    G.add_edge(u, v)
    perm = list(range(n))
    rng.shuffle(perm)
    mapping = {old: new for old, new in zip(range(n), perm)}
    G = nx.relabel_nodes(G, mapping)
    S = {mapping[v] for v in S}
    return MISInstance(G, S, n, h, 'multi_clique_core',
                       {'q': q, 'b': b, 'p_inter': p_inter, 'p_cam': p_cam},
                       seed)


# ============================================================
# SECTION 3: Solvers
# ============================================================

def greedy_min_degree(G: nx.Graph) -> Set[int]:
    H = G.copy()
    mis = set()
    while H.number_of_nodes() > 0:
        v = min(H.nodes(), key=lambda x: H.degree(x))
        mis.add(v)
        H.remove_nodes_from(set(H.neighbors(v)) | {v})
    return mis


def greedy_max_degree_removal(G: nx.Graph) -> Set[int]:
    """Repeatedly remove the highest-degree vertex (and its edges) until
    the remaining graph has no edges, then return all remaining vertices.
    This mimics the inexact reduction heuristic: high-degree vertices are
    unlikely to be in the MIS, so strip them first."""
    H = G.copy()
    while H.number_of_edges() > 0:
        v = max(H.nodes(), key=lambda x: H.degree(x))
        H.remove_node(v)
    return set(H.nodes())


def simulated_annealing_mis(G: nx.Graph, max_iter: int = 20000,
                            T0: float = 2.0, alpha: float = 0.9995,
                            seed: int = 42) -> Set[int]:
    """SA for MIS: start from a greedy solution, perturb by adding/removing
    vertices, accept worsening moves with Boltzmann probability."""
    rng = np.random.RandomState(seed)
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    nodes = list(G.nodes())

    # Start from greedy solution
    current = set(greedy_min_degree(G))
    best = set(current)
    T = T0

    for _ in range(max_iter):
        v = nodes[rng.randint(len(nodes))]

        if v in current:
            # Try removing v
            candidate = current - {v}
            delta = -1
        else:
            # Try adding v — must kick conflicting neighbors
            conflicts = adj[v] & current
            candidate = (current - conflicts) | {v}
            delta = 1 - len(conflicts)

        # Accept if improvement, or probabilistically if worsening
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current = candidate

        if len(current) > len(best):
            best = set(current)

        T *= alpha

    return best


def local_search_1_2_swap(G: nx.Graph, init_mis: Set[int],
                          max_iter: int = 5000) -> Set[int]:
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    current = set(init_mis)
    best = set(current)
    rng = np.random.RandomState(42)
    for _ in range(max_iter):
        improved = False
        nodes_list = list(current)
        rng.shuffle(nodes_list)
        for v_out in nodes_list:
            candidate = current - {v_out}
            freed = set()
            for u in adj[v_out]:
                if u not in candidate and not (adj[u] & candidate):
                    freed.add(u)
            added = set()
            for u in freed:
                if not (adj[u] & (candidate | added)):
                    added.add(u)
                    if len(added) >= 2:
                        break
            if len(added) >= 2:
                current = candidate | added
                improved = True
                break
        if len(current) > len(best):
            best = set(current)
        if not improved:
            break
    return best


def spectral_mis(G: nx.Graph) -> Set[int]:
    n = G.number_of_nodes()
    if n < 3 or G.number_of_edges() == 0:
        return set(G.nodes())
    try:
        from scipy.sparse.linalg import eigsh
        A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).astype(float)
        _, vecs = eigsh(A, k=1, which='LA')
        v1 = vecs[:, 0]
        threshold = np.median(v1)
        candidates = [i for i in range(n) if v1[i] < threshold]
        mis = set()
        for v in sorted(candidates, key=lambda x: G.degree(x)):
            if not (set(G.neighbors(v)) & mis):
                mis.add(v)
        return mis
    except Exception:
        return greedy_min_degree(G)


def exact_mis_small(G: nx.Graph) -> Set[int]:
    if G.number_of_nodes() > 80:
        return set()
    complement = nx.complement(G)
    cliques = list(nx.find_cliques(complement))
    if cliques:
        return set(max(cliques, key=len))
    return set()


# ============================================================
# SECTION 4: KaMIS Wrapper
# ============================================================

class KaMISRunner:
    def __init__(self, kamis_path: str = None):
        if kamis_path is None:
            for candidate in ['./KaMIS/deploy', '../KaMIS/deploy',
                              './KaMIS-master/deploy', '../KaMIS-master/deploy',
                              './deploy']:
                if os.path.isdir(candidate):
                    kamis_path = candidate
                    break
        self.kamis_path = kamis_path
        self.available = kamis_path is not None and os.path.isdir(kamis_path)
        if self.available:
            self.binaries = {}
            for name in ['redumis', 'online_mis', 'weighted_branch_reduce']:
                path = os.path.join(kamis_path, name)
                if os.path.isfile(path):
                    self.binaries[name] = path
            print(f"KaMIS found at {kamis_path}: {list(self.binaries.keys())}")
        else:
            self.binaries = {}
            print("KaMIS not found. Python-only solvers will be used.")

    def solve(self, G: nx.Graph, solver: str = 'online_mis',
              time_limit: float = 30.0, seed: int = 0) -> Tuple[Set[int], float]:
        if not self.available or solver not in self.binaries:
            return set(), -1.0
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_file = os.path.join(tmpdir, 'graph.metis')
            output_file = os.path.join(tmpdir, 'solution.txt')
            write_metis(G, graph_file)
            cmd = [
                self.binaries[solver], graph_file,
                f'--time_limit={time_limit}', f'--seed={seed}',
                f'--output={output_file}', '--console_log'
            ]
            start = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=time_limit + 10)
                elapsed = time.time() - start
                if os.path.isfile(output_file):
                    return read_kamis_solution(output_file, G.number_of_nodes()), elapsed
                else:
                    print(f"  KaMIS no output. stderr: {result.stderr[:200]}")
                    return set(), elapsed
            except subprocess.TimeoutExpired:
                return set(), time_limit
            except Exception as e:
                print(f"  KaMIS error: {e}")
                return set(), -1.0


# ============================================================
# SECTION 5: Benchmarking Pipeline
# ============================================================

@dataclass
class SolverResult:
    name: str
    mis_size: int
    overlap_with_planted: int
    recall: float
    runtime: float
    is_verified: bool


def verify_independent_set(G: nx.Graph, S: Set[int]) -> bool:
    for u in S:
        for v in S:
            if u != v and G.has_edge(u, v):
                return False
    return True


def run_all_solvers(inst: MISInstance, kamis: KaMISRunner) -> List[SolverResult]:
    G = inst.graph
    results = []
    solvers_py = {
        'Greedy_MinDeg': lambda: greedy_min_degree(G),
        'Greedy_MaxDegRemoval': lambda: greedy_max_degree_removal(G),
        'LocalSearch': lambda: local_search_1_2_swap(G, greedy_min_degree(G)),
        'SimulatedAnnealing': lambda: simulated_annealing_mis(G),
        'Spectral': lambda: spectral_mis(G),
    }
    if inst.n <= 80:
        solvers_py['Exact_NX'] = lambda: exact_mis_small(G)
    for name, solver_fn in solvers_py.items():
        t0 = time.time()
        try:
            mis = solver_fn()
        except Exception as e:
            print(f"    {name} failed: {e}")
            continue
        elapsed = time.time() - t0
        overlap = len(mis & inst.planted_set)
        results.append(SolverResult(
            name=name, mis_size=len(mis), overlap_with_planted=overlap,
            recall=overlap / max(inst.h, 1), runtime=elapsed,
            is_verified=verify_independent_set(G, mis)
        ))
    for solver_name in ['online_mis', 'redumis']:
        if solver_name in kamis.binaries:
            tl = 40.0 if inst.n <= 100 else (60.0 if inst.n <= 300 else 120.0)
            mis, elapsed = kamis.solve(G, solver_name, time_limit=tl)
            if mis:
                overlap = len(mis & inst.planted_set)
                results.append(SolverResult(
                    name=f'KaMIS_{solver_name}', mis_size=len(mis),
                    overlap_with_planted=overlap,
                    recall=overlap / max(inst.h, 1), runtime=elapsed,
                    is_verified=verify_independent_set(G, mis)
                ))
    return results


# ============================================================
# SECTION 6: Output
# ============================================================

def write_summary_txt(all_results: Dict, filepath: str):
    """Write a summary txt grouped by family, then by solver."""
    from collections import defaultdict

    # family -> solver -> list of (size, recall, runtime)
    data = defaultdict(lambda: defaultdict(list))

    for inst_id, bundle in all_results.items():
        inst = bundle['instance']
        for r in bundle['results']:
            data[inst.family][r.name].append((r.mis_size, r.recall, r.runtime))

    lines = []
    lines.append("=" * 70)
    lines.append("COMBINED BENCHMARK SUMMARY")
    lines.append("=" * 70)

    for family in sorted(data.keys()):
        lines.append(f"\n--- Family: {family} ---")
        for solver in sorted(data[family].keys()):
            entries = data[family][solver]
            avg_size = sum(e[0] for e in entries) / len(entries)
            avg_recall = sum(e[1] for e in entries) / len(entries)
            avg_time = sum(e[2] for e in entries) / len(entries)
            n_inst = len(entries)
            lines.append(
                f"  {solver:30s}: avg_size={avg_size:.1f}, "
                f"avg_recall={avg_recall:.3f}, avg_time={avg_time:.3f}s "
                f"({n_inst} instances)"
            )

    lines.append("\n" + "=" * 70)
    lines.append("PER-INSTANCE DETAIL")
    lines.append("=" * 70)
    lines.append(f"{'Instance':40s} {'Solver':22s} {'Size':>5s} {'Recall':>7s} {'Time(s)':>8s} {'Valid':>5s}")
    lines.append("-" * 90)

    for inst_id, bundle in sorted(all_results.items()):
        for r in bundle['results']:
            lines.append(
                f"{inst_id:40s} {r.name:22s} {r.mis_size:5d} "
                f"{r.recall:7.3f} {r.runtime:8.3f} "
                f"{'Y' if r.is_verified else 'N':>5s}"
            )
        lines.append("-" * 90)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSummary written to {filepath}")


# ============================================================
# SECTION 7: Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Combined MIS Benchmarking Suite')
    parser.add_argument('--full', action='store_true',
                        help='Run full benchmark (larger graphs)')
    parser.add_argument('--kamis_path', type=str, default=None)
    parser.add_argument('--output', type=str, default='combined_run.txt',
                        help='Output summary file (default: combined_run.txt)')
    parser.add_argument('--json', type=str, default='combined_results.json',
                        help='Output JSON file (default: combined_results.json)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Combined MIS Benchmarking Suite")
    print("=" * 60)

    kamis = KaMISRunner(args.kamis_path)

    # Combined configs: union of n values from both benchmark files
    if args.full:
        configs = [
            # From mis_benchmark.py full (low p, large n)
            ('erdos_renyi',  50,  7, {'p': 0.01}),
            ('erdos_renyi', 100, 10, {'p': 0.01}),
            ('erdos_renyi', 200, 14, {'p': 0.01}),
            ('erdos_renyi', 400, 20, {'p': 0.01}),
            ('erdos_renyi', 625, 25, {'p': 0.01}),
            ('erdos_renyi',  50,  7, {'p': 0.001}),
            ('erdos_renyi', 100, 10, {'p': 0.001}),
            ('erdos_renyi', 200, 14, {'p': 0.001}),
            ('erdos_renyi', 400, 20, {'p': 0.001}),
            ('erdos_renyi', 625, 25, {'p': 0.001}),
            # From mis_benchmark2.py full (p=0.5, wider n)
            ('erdos_renyi',  50,  7, {'p': 0.5}),
            ('erdos_renyi', 100, 10, {'p': 0.5}),
            ('erdos_renyi', 200, 14, {'p': 0.5}),
            # multi_clique_core from mis_benchmark2.py full
            ('multi_clique_core',  50, 10, {'q': 3, 'b': 2, 'p_inter': 0.5, 'p_cam': 0.3}),
            ('multi_clique_core', 100, 15, {'q': 3, 'b': 2, 'p_inter': 0.5, 'p_cam': 0.3}),
            ('multi_clique_core', 200, 20, {'q': 3, 'b': 2, 'p_inter': 0.5, 'p_cam': 0.3}),
        ]
    else:
        configs = [
            # Small graphs (original defaults)
            ('erdos_renyi',  50, 10, {'p': 0.5}),
            ('erdos_renyi',  50, 10, {'p': 0.8}),
            ('erdos_renyi', 100, 10, {'p': 0.2}),
            ('erdos_renyi',  30,  5, {'p': 0.5}),
            ('erdos_renyi',  50,  7, {'p': 0.5}),
            # Larger graphs: n=200..600 (sparse p=0.05, moderate p=0.2)
            ('erdos_renyi', 200, 15, {'p': 0.05}),
            ('erdos_renyi', 300, 18, {'p': 0.05}),
            ('erdos_renyi', 400, 20, {'p': 0.05}),
            ('erdos_renyi', 500, 22, {'p': 0.05}),
            ('erdos_renyi', 600, 25, {'p': 0.05}),
            ('erdos_renyi', 200, 12, {'p': 0.2}),
            ('erdos_renyi', 300, 14, {'p': 0.2}),
            ('erdos_renyi', 400, 16, {'p': 0.2}),
            ('erdos_renyi', 500, 18, {'p': 0.2}),
            ('erdos_renyi', 600, 20, {'p': 0.2}),
            # Multi-clique-core (original + larger)
            ('multi_clique_core',  20, 10, {'q': 3, 'b': 2, 'p_inter': 0.3, 'p_cam': 0.2}),
            ('multi_clique_core',  50, 10, {'q': 3, 'b': 2, 'p_inter': 0.6, 'p_cam': 0.5}),
            ('multi_clique_core',  30,  8, {'q': 3, 'b': 2, 'p_inter': 0.5, 'p_cam': 0.3}),
            ('multi_clique_core', 200, 20, {'q': 4, 'b': 3, 'p_inter': 0.5, 'p_cam': 0.3}),
            ('multi_clique_core', 300, 25, {'q': 4, 'b': 3, 'p_inter': 0.5, 'p_cam': 0.3}),
            ('multi_clique_core', 400, 30, {'q': 5, 'b': 3, 'p_inter': 0.5, 'p_cam': 0.3}),
        ]

    all_results = {}

    for family, n, h, params in configs:
        for seed in [42, 123, 456]:
            if family == 'erdos_renyi':
                inst = gen_erdos_renyi_planted(n, h, params.get('p', 0.5), seed)
            elif family == 'multi_clique_core':
                inst = gen_multi_clique_core(n, h, seed=seed, **params)
            else:
                continue

            if not inst.verify():
                print(f"  SKIP {inst.instance_id}: planted set not independent")
                continue

            print(f"\n  Running {inst.instance_id} "
                  f"(n={n}, h={h}, edges={inst.graph.number_of_edges()})...")

            results = run_all_solvers(inst, kamis)
            all_results[inst.instance_id] = {'instance': inst, 'results': results}

            for r in results:
                print(f"    {r.name:22s}  size={r.mis_size:4d}  "
                      f"recall={r.recall:.3f}  time={r.runtime:.3f}s  "
                      f"{'OK' if r.is_verified else 'INVALID'}")

    # Write summary txt
    write_summary_txt(all_results, args.output)

    # Write JSON
    json_results = {}
    for inst_id, bundle in all_results.items():
        inst = bundle['instance']
        json_results[inst_id] = {
            'family': inst.family, 'n': inst.n, 'h': inst.h,
            'params': inst.params, 'seed': inst.seed,
            'num_edges': inst.graph.number_of_edges(),
            'results': [
                {'solver': r.name, 'size': r.mis_size,
                 'overlap': r.overlap_with_planted, 'recall': r.recall,
                 'runtime': r.runtime, 'verified': r.is_verified}
                for r in bundle['results']
            ]
        }
    with open(args.json, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to {args.json}")


if __name__ == '__main__':
    main()
