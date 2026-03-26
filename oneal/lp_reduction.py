#!/usr/bin/env python3
"""
lp_reduction.py
===============
LP relaxation as preprocessing for MIS/MWIS solvers.

The LP relaxation of MIS:
    max  Σ w_i x_i
    s.t. x_i + x_j ≤ 1   for all (i,j) ∈ E
         0 ≤ x_i ≤ 1

Vertices with LP value = 1 can be fixed in the IS (fixed_in).
Vertices with LP value = 0 can be excluded (fixed_out).
Fractional vertices are passed to the downstream heuristic on the residual graph.

This mirrors the implicit LP-based reduction in KaMIS's branch_and_reduce_algorithm
(via bipartite matching / Dinic's algorithm), but uses scipy.optimize.linprog
for explicit LP solving.

Scalability note:
  - Dense graphs (m > 100_000) are skipped (LP returns all-0.5, no reduction).
  - Effective range: sparse instances (p ≤ 0.05, n ≤ 1000).
"""

from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple, Optional, Callable

import numpy as np
import networkx as nx

# scipy is optional; LP preprocessing is silently disabled if not available
try:
    from scipy.optimize import linprog
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Maximum number of edge constraints for LP (above this, skip LP)
_LP_EDGE_LIMIT = 100_000
_LP_FRAC_EPSILON = 1e-6    # threshold for fixed_in / fixed_out classification


# ── LP solver ─────────────────────────────────────────────────────────────────

def solve_lp_relaxation(
    G: nx.Graph,
    weights: Dict[int, float] = None,
    epsilon: float = _LP_FRAC_EPSILON,
) -> Dict[int, float]:
    """
    Solve the LP relaxation of MIS/MWIS on graph G.
    Returns dict node -> LP value in [0, 1].

    If scipy is unavailable or the graph is too large (m > _LP_EDGE_LIMIT),
    returns {v: 0.5 for all v} (all fractional = no reduction).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return {}

    all_half = {v: 0.5 for v in G.nodes()}

    if not _SCIPY_AVAILABLE:
        return all_half

    if m > _LP_EDGE_LIMIT:
        return all_half

    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}

    # Objective: minimize -w^T x  (maximise w^T x)
    if weights:
        c = np.array([-weights.get(v, 1.0) for v in nodes])
    else:
        c = np.full(n, -1.0)

    # Inequality constraints: x_i + x_j <= 1 for each edge
    edges = list(G.edges())
    n_eq = len(edges)
    if n_eq == 0:
        # No edges: all vertices can be included
        return {v: 1.0 for v in nodes}

    A_ub = np.zeros((n_eq, n), dtype=np.float64)
    b_ub = np.ones(n_eq, dtype=np.float64)
    for k, (u, v) in enumerate(edges):
        A_ub[k, idx[u]] = 1.0
        A_ub[k, idx[v]] = 1.0

    # Bounds: 0 <= x_i <= 1
    bounds = [(0.0, 1.0)] * n

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            return {nodes[i]: float(res.x[i]) for i in range(n)}
        else:
            return all_half
    except Exception:
        return all_half


# ── Classification ────────────────────────────────────────────────────────────

def classify_lp_solution(
    lp_values: Dict[int, float],
    epsilon: float = _LP_FRAC_EPSILON,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Partition LP solution into:
      fixed_in:  lp_value >= 1 - epsilon  (include in IS)
      fixed_out: lp_value <= epsilon       (exclude from IS)
      fractional: remaining              (pass to heuristic)
    """
    fixed_in   = set()
    fixed_out  = set()
    fractional = set()
    for v, x in lp_values.items():
        if x >= 1.0 - epsilon:
            fixed_in.add(v)
        elif x <= epsilon:
            fixed_out.add(v)
        else:
            fractional.add(v)
    return fixed_in, fixed_out, fractional


# ── Residual graph extraction ─────────────────────────────────────────────────

def extract_residual_graph(
    G: nx.Graph,
    fixed_in: Set[int],
    fixed_out: Set[int],
) -> Tuple[nx.Graph, Set[int]]:
    """
    Build residual graph over fractional nodes:
      - Remove fixed_in and their neighborhoods (they're in IS, block neighbors).
      - Remove fixed_out (excluded).
      - The residual contains only fractional nodes minus N(fixed_in).

    Returns (residual_G, removed_nodes).
    """
    # Nodes blocked by fixed_in: their neighbors can't be in IS
    blocked = set(fixed_in)
    for v in fixed_in:
        blocked.update(G.neighbors(v))
    removed = fixed_in | fixed_out | blocked

    # Subgraph on fractional nodes that are not blocked
    residual_nodes = set(G.nodes()) - removed
    residual_G = G.subgraph(residual_nodes).copy()
    return residual_G, removed


# ── LP-preprocessed solver ────────────────────────────────────────────────────

def lp_preprocess_then_solve(
    G: nx.Graph,
    solver_fn: Callable[[nx.Graph], Set[int]],
    weights: Dict[int, float] = None,
    epsilon: float = _LP_FRAC_EPSILON,
) -> Tuple[Set[int], float, Dict]:
    """
    Full pipeline:
      1. Solve LP relaxation.
      2. Classify fixed_in / fixed_out / fractional.
      3. Build residual graph.
      4. Run solver_fn on residual graph.
      5. Combine: fixed_in + residual solution.

    Returns (solution, total_runtime, metadata).
    metadata keys:
      lp_available, lp_skipped, lp_fixed_in, lp_fixed_out, lp_fractional,
      lp_reduction_ratio, lp_time, solver_time
    """
    n = G.number_of_nodes()
    t_start = time.time()

    # Step 1: LP
    t_lp = time.time()
    lp_values = solve_lp_relaxation(G, weights, epsilon)
    lp_time = time.time() - t_lp

    # Step 2: Classify
    fixed_in, fixed_out, fractional = classify_lp_solution(lp_values, epsilon)

    lp_skipped = all(abs(v - 0.5) < 0.01 for v in lp_values.values())
    lp_reduction_ratio = 1.0 - len(fractional) / max(n, 1)

    # Step 3: Residual graph
    residual_G, _ = extract_residual_graph(G, fixed_in, fixed_out)

    # Step 4: Solve residual
    t_solver = time.time()
    residual_sol: Set[int] = set()
    if residual_G.number_of_nodes() > 0:
        try:
            residual_sol = solver_fn(residual_G)
        except Exception:
            residual_sol = set()
    solver_time = time.time() - t_solver

    # Step 5: Combine
    solution = fixed_in | residual_sol
    total_time = time.time() - t_start

    metadata = {
        "lp_available": _SCIPY_AVAILABLE,
        "lp_skipped": lp_skipped,
        "lp_fixed_in": len(fixed_in),
        "lp_fixed_out": len(fixed_out),
        "lp_fractional": len(fractional),
        "lp_reduction_ratio": lp_reduction_ratio,
        "lp_time": lp_time,
        "solver_time": solver_time,
        "total_time": total_time,
    }
    return solution, total_time, metadata


def lp_solver_wrapper(
    base_solver_fn: Callable[[nx.Graph], Set[int]],
    weights: Dict[int, float] = None,
) -> Callable[[nx.Graph], Set[int]]:
    """
    Returns a solver function that applies LP preprocessing then base_solver_fn.
    Use this to create 'LP+Greedy', 'LP+LocalSearch' etc.
    """
    def wrapped(G: nx.Graph) -> Set[int]:
        sol, _, _ = lp_preprocess_then_solve(G, base_solver_fn, weights)
        return sol
    return wrapped


def lp_stats(
    G: nx.Graph,
    weights: Dict[int, float] = None,
    epsilon: float = _LP_FRAC_EPSILON,
) -> Dict:
    """Return statistics about LP reduction quality (for analysis)."""
    lp_values = solve_lp_relaxation(G, weights, epsilon)
    fixed_in, fixed_out, fractional = classify_lp_solution(lp_values, epsilon)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    lp_bound = sum(lp_values.values())
    return {
        "n": n, "m": m,
        "fixed_in": len(fixed_in),
        "fixed_out": len(fixed_out),
        "fractional": len(fractional),
        "reduction_ratio": 1.0 - len(fractional) / max(n, 1),
        "lp_bound": lp_bound,
        "lp_available": _SCIPY_AVAILABLE,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("lp_reduction.py self-test")
    print(f"  scipy available: {_SCIPY_AVAILABLE}")

    if _SCIPY_AVAILABLE:
        # K4: LP solution should be all 0.5 (fractional)
        G = nx.complete_graph(4)
        lp = solve_lp_relaxation(G)
        print(f"  K4 LP values: {lp}")
        fi, fo, frac = classify_lp_solution(lp)
        print(f"  K4 classified: fixed_in={fi}, fixed_out={fo}, fractional={frac}")

        # Path P4 (0-1-2-3): optimal IS = {{0,2}, {1,3}} size=2
        # LP should give 0.5 for all (fractional) OR fix 0 and 3
        G2 = nx.path_graph(4)
        lp2 = solve_lp_relaxation(G2)
        print(f"  P4 LP values: {lp2}")

        # Star K_{1,3}: center has LP value 0.5, leaves 0.5 typically
        G3 = nx.star_graph(3)   # center=0, leaves=1,2,3
        weights3 = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        lp3 = solve_lp_relaxation(G3, weights3)
        print(f"  Star K_1,3 LP values: {lp3}")

        # Weighted star: center weight = 10
        weights3w = {0: 10.0, 1: 1.0, 2: 1.0, 3: 1.0}
        lp3w = solve_lp_relaxation(G3, weights3w)
        fi3w, fo3w, frac3w = classify_lp_solution(lp3w)
        print(f"  Weighted star LP: fixed_in={fi3w}, fixed_out={fo3w}, frac={frac3w}")

        # Full pipeline test
        from mis_benchmark_combined import greedy_min_degree
        sol, t, meta = lp_preprocess_then_solve(G3, greedy_min_degree, weights3w)
        print(f"  LP+Greedy on weighted star: sol={sol}, meta={meta}")
        assert 0 in sol or len(sol) >= 1, "Should find IS"

    print("  OK")
