#!/usr/bin/env python3
"""
MIS Benchmarking Suite with KaMIS Integration
=============================================
Generates planted MIS instances, runs Python solvers and KaMIS,
computes ET for small instances, and produces comparison tables.

Usage:
    python mis_benchmark.py                    # quick test
    python mis_benchmark.py --full             # full benchmark
    python mis_benchmark.py --kamis_path PATH  # specify KaMIS binary location
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
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from math import comb
from pathlib import Path

# ============================================================
# SECTION 1: Graph I/O (METIS format for KaMIS)
# ============================================================

def write_metis(G: nx.Graph, filepath: str):
    """Write a NetworkX graph to METIS format for KaMIS.
    KaMIS requires 1-indexed, sorted adjacency lists."""
    n = G.number_of_nodes()
    # Ensure nodes are 0..n-1
    nodes = sorted(G.nodes())
    assert nodes == list(range(n)), "Nodes must be 0..n-1"
    
    m = G.number_of_edges()
    with open(filepath, 'w') as f:
        f.write(f"{n} {m}\n")
        for v in range(n):
            neighbors = sorted([u + 1 for u in G.neighbors(v)])  # 1-indexed
            f.write(" ".join(map(str, neighbors)) + "\n")


def read_kamis_solution(filepath: str, n: int) -> Set[int]:
    """Read a KaMIS output file (one 0/1 per line) into a set of IS nodes."""
    mis = set()
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            val = int(line.strip())
            if val == 1:
                mis.add(i)
    return mis


# ============================================================
# SECTION 2: Instance Generation
# ============================================================

@dataclass
class MISInstance:
    """A planted MIS instance with metadata."""
    graph: nx.Graph
    planted_set: Set[int]
    n: int
    h: int
    family: str
    params: Dict
    seed: int
    
    def verify(self) -> bool:
        """Check the planted set is independent."""
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
                f"_q{self.params['q']}_b{self.params['b']}"
                f"_pi{self.params['p_inter']}_pc{self.params['p_cam']}"
                f"_s{self.seed}"
            )
        return f"{self.family}_n{self.n}_h{self.h}_s{self.seed}"


def gen_erdos_renyi_planted(n: int, h: int, p: float = 0.5,
                            seed: int = 42) -> MISInstance:
    """G(n,p) with a planted independent set of size h."""
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
    """Multi-clique core construction from the hard instance notes.
    V\\S is partitioned into q cliques, each blocking b vertices of S."""
    rng = np.random.RandomState(seed)
    assert q * b <= h <= n
    
    S = set(range(h))
    R = list(range(h, n))
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Partition R into q blocks, make each a clique
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
    
    # Inter-block edges
    for a in range(q):
        for bb in range(a + 1, q):
            for u in blocks[a]:
                for v in blocks[bb]:
                    if rng.random() < p_inter:
                        G.add_edge(u, v)
    
    # Blocking sets B_j ⊂ S
    S_list = sorted(S)
    for j in range(q):
        B_j = S_list[j * b: (j + 1) * b]
        for u in blocks[j]:
            for v in B_j:
                G.add_edge(u, v)
    
    # Camouflage edges
    all_blocked = set(S_list[:q * b])
    for j in range(q):
        B_j = set(S_list[j * b: (j + 1) * b])
        S_minus_Bj = S - B_j
        for u in blocks[j]:
            for v in S_minus_Bj:
                if rng.random() < p_cam:
                    G.add_edge(u, v)
    
    # Random relabeling
    perm = list(range(n))
    rng.shuffle(perm)
    mapping = {old: new for old, new in zip(range(n), perm)}
    G = nx.relabel_nodes(G, mapping)
    S = {mapping[v] for v in S}
    
    return MISInstance(G, S, n, h, 'multi_clique_core',
                       {'q': q, 'b': b, 'p_inter': p_inter, 'p_cam': p_cam},
                       seed)


# ============================================================
# SECTION 3: Python Solvers (no external dependencies)
# ============================================================

def greedy_min_degree(G: nx.Graph) -> Set[int]:
    """Greedy MIS: always pick the minimum-degree vertex."""
    H = G.copy()
    mis = set()
    while H.number_of_nodes() > 0:
        v = min(H.nodes(), key=lambda x: H.degree(x))
        mis.add(v)
        to_remove = set(H.neighbors(v)) | {v}
        H.remove_nodes_from(to_remove)
    return mis


def local_search_1_2_swap(G: nx.Graph, init_mis: Set[int],
                          max_iter: int = 5000) -> Set[int]:
    """Improve an MIS via (1,2)-swap local search."""
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
            # Vertices freed by removing v_out
            freed = set()
            for u in adj[v_out]:
                if u not in candidate:
                    blockers = adj[u] & candidate
                    if not blockers:
                        freed.add(u)
            
            # Try adding 2 from freed
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
    """Spectral method: use adjacency eigenvector to find planted IS."""
    n = G.number_of_nodes()
    if n < 3 or G.number_of_edges() == 0:
        return set(G.nodes())
    
    try:
        from scipy.sparse.linalg import eigsh
        A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).astype(float)
        _, vecs = eigsh(A, k=1, which='LA')
        v1 = vecs[:, 0]
        
        # Planted vertices tend to have smaller eigenvector entries
        threshold = np.median(v1)
        candidates = [i for i in range(n) if v1[i] < threshold]
        
        # Greedily clean up
        mis = set()
        for v in sorted(candidates, key=lambda x: G.degree(x)):
            if not (set(G.neighbors(v)) & mis):
                mis.add(v)
        return mis
    except Exception:
        return greedy_min_degree(G)


def exact_mis_small(G: nx.Graph) -> Set[int]:
    """Exact MIS via complement clique (only for n ≤ 60)."""
    if G.number_of_nodes() > 60:
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
    """Python wrapper around KaMIS binaries."""
    
    def __init__(self, kamis_path: str = None):
        """
        Args:
            kamis_path: path to KaMIS deploy/ directory.
                        If None, tries ./KaMIS/deploy/ and ../KaMIS/deploy/
        """
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
        """
        Run a KaMIS solver on the graph.
        
        Returns:
            (independent_set, runtime_seconds)
        """
        if not self.available or solver not in self.binaries:
            return set(), -1.0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_file = os.path.join(tmpdir, 'graph.metis')
            output_file = os.path.join(tmpdir, 'solution.txt')
            
            write_metis(G, graph_file)
            
            cmd = [
                self.binaries[solver],
                graph_file,
                f'--time_limit={time_limit}',
                f'--seed={seed}',
                f'--output={output_file}',
                '--console_log'
            ]
            
            start = time.time()
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=time_limit + 10
                )
                elapsed = time.time() - start
                
                if os.path.isfile(output_file):
                    mis = read_kamis_solution(output_file, G.number_of_nodes())
                    return mis, elapsed
                else:
                    print(f"  KaMIS produced no output file. stderr: {result.stderr[:200]}")
                    return set(), elapsed
                    
            except subprocess.TimeoutExpired:
                return set(), time_limit
            except Exception as e:
                print(f"  KaMIS error: {e}")
                return set(), -1.0


# ============================================================
# SECTION 5: ET Computation
# ============================================================

def compute_ET_toy(n: int) -> Dict:
    """Compute ET for the toy polynomial f = x1 + ... + xn - n.
    Closed-form from Li's paper."""
    ET = 2.0 * n * n
    for i in range(1, n):
        j = n - i
        ET += j * j + j * j / (n - j + 1)
    
    return {
        'ET': ET,
        'n': n,
        'ratio_to_n3_over_3': ET / (n**3 / 3),
        'z_norm_sq': n,
    }


def count_independent_sets_by_size(G: nx.Graph, S: Set[int],
                                   max_size: int = None) -> Dict:
    """Count independent sets by size, decomposed by overlap with S.
    Only feasible for small graphs (n ≤ 20)."""
    n = G.number_of_nodes()
    h = len(S)
    if max_size is None:
        max_size = h
    if n > 20:
        return {'error': 'too_large'}
    
    nodes = sorted(G.nodes())
    edges = set()
    for u, v in G.edges():
        edges.add((min(u, v), max(u, v)))
    
    def is_independent(subset):
        s = list(subset)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if (min(s[i], s[j]), max(s[i], s[j])) in edges:
                    return False
        return True
    
    counts = {}  # (size, t) -> count where t = |T \ S|
    total_by_size = {}
    
    for size in range(1, max_size + 1):
        total = 0
        for subset in combinations(nodes, size):
            if is_independent(set(subset)):
                t = len(set(subset) - S)
                key = (size, t)
                counts[key] = counts.get(key, 0) + 1
                total += 1
        total_by_size[size] = total
    
    # Check the counting condition from the paper
    counting_ok = True
    for i in range(1, h + 1):
        if i in total_by_size:
            bound = 10 * n * comb(h, i)
            if total_by_size[i] > bound:
                counting_ok = False
    
    return {
        'counts_by_size_and_overlap': {str(k): v for k, v in counts.items()},
        'total_by_size': total_by_size,
        'counting_condition_ok': counting_ok,
        'n': n, 'h': h,
    }


# ============================================================
# SECTION 6: Benchmarking Pipeline
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
    """Check that S is indeed an independent set in G."""
    for u in S:
        for v in S:
            if u != v and G.has_edge(u, v):
                return False
    return True


def run_all_solvers(inst: MISInstance,
                    kamis: KaMISRunner) -> List[SolverResult]:
    """Run all available solvers on one instance."""
    G = inst.graph
    results = []
    
    # --- Python solvers ---
    solvers_py = {
        'Greedy_MinDeg': lambda: greedy_min_degree(G),
        'LocalSearch': lambda: local_search_1_2_swap(G, greedy_min_degree(G)),
        'Spectral': lambda: spectral_mis(G),
    }
    
    if inst.n <= 60:
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
        recall = overlap / max(inst.h, 1)
        verified = verify_independent_set(G, mis)
        
        results.append(SolverResult(
            name=name, mis_size=len(mis), overlap_with_planted=overlap,
            recall=recall, runtime=elapsed, is_verified=verified
        ))
    
    # --- KaMIS solvers ---
    for solver_name in ['online_mis', 'redumis']:
        if solver_name in kamis.binaries:
            tl = 40.0 if inst.n <= 100 else 100.0
            mis, elapsed = kamis.solve(G, solver_name, time_limit=tl)
            if mis:
                overlap = len(mis & inst.planted_set)
                recall = overlap / max(inst.h, 1)
                verified = verify_independent_set(G, mis)
                results.append(SolverResult(
                    name=f'KaMIS_{solver_name}', mis_size=len(mis),
                    overlap_with_planted=overlap, recall=recall,
                    runtime=elapsed, is_verified=verified
                ))
    
    return results


def print_results_table(all_results):
    print("\n" + "=" * 90)
    print(f"{'Instance':30s} {'Solver':22s} {'Size':>5s} {'Recall':>7s} "
          f"{'Time(s)':>8s} {'Valid':>5s}")
    print("-" * 90)

    for inst_id, bundle in all_results.items():
        results = bundle['results']
        for r in results:
            print(f"{inst_id:30s} {r.name:22s} {r.mis_size:5d} "
                  f"{r.recall:7.3f} {r.runtime:8.3f} "
                  f"{'✓' if r.is_verified else '✗':>5s}")
        print("-" * 90)


# ============================================================
# SECTION 7: Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='MIS Benchmarking Suite')
    parser.add_argument('--full', action='store_true',
                        help='Run full benchmark (larger graphs)')
    parser.add_argument('--kamis_path', type=str, default=None,
                        help='Path to KaMIS deploy/ directory')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output file for results')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  MIS Classical Benchmarking Suite")
    print("=" * 60)
    
    # Initialize KaMIS
    kamis = KaMISRunner(args.kamis_path)
    
    # === Part 1: Toy ET Validation ===
    print("\n--- Toy ET Validation (f = x1 + ... + xn - n) ---")
    print(f"{'n':>5s}  {'ET':>12s}  {'n^3/3':>12s}  {'Ratio':>8s}")
    for n_toy in [5, 8, 10, 15, 20, 30, 50]:
        et = compute_ET_toy(n_toy)
        print(f"{n_toy:5d}  {et['ET']:12.1f}  {n_toy**3/3:12.1f}  "
              f"{et['ratio_to_n3_over_3']:8.4f}")
    
    # === Part 2: Small Instance IS Counting ===
    print("\n--- Independent Set Counting (small instances) ---")
    for n_small, h_small in [(10, 4), (12, 4), (14, 5), (16, 5)]:
        inst = gen_erdos_renyi_planted(n_small, h_small, p=0.5, seed=100)
        if inst.verify():
            counts = count_independent_sets_by_size(inst.graph, inst.planted_set)
            print(f"  n={n_small}, h={h_small}: counting_ok={counts.get('counting_condition_ok')}")
            for sz, cnt in sorted(counts.get('total_by_size', {}).items()):
                bnd = comb(h_small, sz)
                ratio = cnt / max(bnd, 1)
                print(f"    I_{sz} = {cnt:6d},  C(h,{sz}) = {bnd:6d},  "
                      f"ratio = {ratio:.2f}")
    
    # === Part 3: Solver Comparison ===
    print("\n--- Solver Comparison ---")
    
    if args.full:
        configs = [
            ('erdos_renyi', 625, 25, {'p': 0.01}),
            ('erdos_renyi', 400, 20, {'p': 0.01}),
            ('erdos_renyi', 50, 7, {'p': 0.01}),
            ('erdos_renyi', 100, 10, {'p': 0.01}),
            ('erdos_renyi', 200, 14, {'p': 0.01}),

            ('erdos_renyi', 625, 25, {'p': 0.001}),
            ('erdos_renyi', 400, 20, {'p': 0.001}),
            ('erdos_renyi', 50, 7, {'p': 0.001}),
            ('erdos_renyi', 100, 10, {'p': 0.001}),
            ('erdos_renyi', 200, 14, {'p': 0.001}),
            
            ('erdos_renyi', 625, 25, {'p': 0.005}),
            ('erdos_renyi', 400, 20, {'p': 0.005}),
            ('erdos_renyi', 50, 7, {'p': 0.005}),
            ('erdos_renyi', 100, 10, {'p': 0.005}),
            ('erdos_renyi', 200, 14, {'p': 0.005}),
           
        ]
    else:
        configs = [
            ('erdos_renyi', 100, 10, {'p': 0.2}),
            ('erdos_renyi', 50, 10, {'p': 0.5}),
            ('erdos_renyi', 50, 10, {'p': 0.8}),
            ('multi_clique_core', 20, 10, {'q': 3, 'b': 2, 'p_inter': 0.3, 'p_cam': 0.2}),
            ('multi_clique_core', 50, 10, {'q': 3, 'b': 2, 'p_inter': 0.6, 'p_cam': 0.5})
       
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
                status = "✓" if r.is_verified else "✗"
                print(f"    {r.name:22s}  size={r.mis_size:4d}  "
                    f"recall={r.recall:.3f}  time={r.runtime:.3f}s  {status}")
    
    # Print summary table
    print_results_table(all_results)
    
    # Save results
    json_results = {}
    for inst_id, bundle in all_results.items():
        inst = bundle['instance']
        results = bundle['results']
        json_results[inst_id] = {
            'family': inst.family,
            'n': inst.n,
            'h': inst.h,
            'params': inst.params,
            'seed': inst.seed,
            'num_edges': inst.graph.number_of_edges(),
            'results': [
                {
                    'solver': r.name,
                    'size': r.mis_size,
                    'overlap': r.overlap_with_planted,
                    'recall': r.recall,
                    'runtime': r.runtime,
                    'verified': r.is_verified
                }
                for r in results
            ]
        }
    with open(args.output, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()