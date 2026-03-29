#!/usr/bin/env python3
"""
hils.py
=======
Pure Python implementation of HILS (Hybrid Iterated Local Search) and
Red+HILS for Maximum Weight Independent Set.

Based on:
  [1] Nogueira et al. "A hybrid iterated local search heuristic for the MWIS"
  [2] Lamm et al. "Exactly Solving the MWIS Problem on Large Real-World Graphs"

Algorithm correspondence with KaMIS C++:
  vnd_omega_1_swap  <->  local_search::omegaImprovement  (wmis/lib/mis/ils/local_search.h)
  vnd_1_2_swap      <->  local_search::direct_improvement (ARW paper, kindly provided by Werneck)
  perturbation      <->  ils::force / perform_ils           (wmis/lib/mis/ils/ils.h)

Notes:
  - In unweighted mode (all weights=1), vnd_omega_1_swap never improves
    (swapping v in and removing ω≥1 neighbors has gain = 1-ω ≤ 0).
    Only N2 (1,2-swap) is active, consistent with the ARW algorithm.
  - Python HILS targets n ≤ ~2000 with 30s time limit.
    For larger graphs use the C++ mmwis binary.
"""

from __future__ import annotations

import time
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

import networkx as nx

# Import reductions for Red+HILS
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reductions import ReducibleGraph, reduce_graph, lift_solution


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class HilsConfig:
    max_iter: int    = 2_000_000
    time_limit: float = 30.0
    seed: int        = 42
    # Acceptance / perturbation constants (Nogueira et al.)
    c1: int = 1    # gain threshold for N1: w(v) > c1 * sum(removed S-neighbors)
    c2: int = 3    # patience divisor: accept within |S|/c2 steps without improvement
    c3: int = 4    # big reward multiplier on new global best
    c4: int = 2    # stronger perturbation strength on restart
    # ILS forcing
    force_candidates: int = 4  # mirrors MISConfig::force_cand=4 in KaMIS


# ── State ─────────────────────────────────────────────────────────────────────

class HilsState:
    """
    Maintains current IS with O(degree) incremental updates via tight_count.
    tight_count[v] = |N(v) ∩ S| for v ∉ S.
    free[v] iff v ∉ S and tight_count[v] == 0.
    """

    def __init__(
        self,
        adj: Dict[int, Set[int]],
        weights: Dict[int, float],
        nodes: List[int],
    ) -> None:
        self.adj = adj
        self.weights = weights
        self.nodes = nodes
        self.solution: Set[int] = set()
        self.solution_weight: float = 0.0
        self.best_solution: Set[int] = set()
        self.best_weight: float = 0.0
        self.tight_count: Dict[int, int] = {v: 0 for v in nodes}
        self.last_forced: Dict[int, int] = {v: -1 for v in nodes}
        self.iteration: int = 0
        self.start_time: float = time.time()

    def is_free(self, v: int) -> bool:
        return v not in self.solution and self.tight_count[v] == 0

    def one_tight_nodes(self) -> List[int]:
        """Non-solution nodes with exactly one solution-neighbor."""
        return [v for v in self.nodes
                if v not in self.solution and self.tight_count[v] == 1]

    def add_to_solution(self, v: int) -> None:
        assert v not in self.solution
        assert self.tight_count[v] == 0, f"Can't add {v}: tight_count={self.tight_count[v]}"
        self.solution.add(v)
        self.solution_weight += self.weights[v]
        for u in self.adj[v]:
            if u in self.nodes:
                self.tight_count[u] += 1

    def remove_from_solution(self, v: int) -> None:
        assert v in self.solution
        self.solution.discard(v)
        self.solution_weight -= self.weights[v]
        for u in self.adj[v]:
            if u in self.nodes:
                self.tight_count[u] -= 1

    def force_node(self, v: int) -> List[int]:
        """
        Force v into S: remove all conflicting S-neighbors, then add v.
        Returns list of removed nodes.
        """
        removed = [u for u in self.adj[v] if u in self.solution]
        for u in removed:
            self.remove_from_solution(u)
        self.add_to_solution(v)
        return removed

    def snapshot(self) -> Tuple[Set[int], float]:
        return set(self.solution), self.solution_weight

    def restore(self, snap: Tuple[Set[int], float]) -> None:
        sol, wt = snap
        # Clear current solution
        for v in list(self.solution):
            self.remove_from_solution(v)
        # Restore snapshot
        for v in sol:
            self.add_to_solution(v)

    def update_best(self) -> bool:
        if self.solution_weight > self.best_weight:
            self.best_weight = self.solution_weight
            self.best_solution = set(self.solution)
            return True
        return False

    def elapsed(self) -> float:
        return time.time() - self.start_time


# ── Initialisation ────────────────────────────────────────────────────────────

def _build_state(
    adj: Dict[int, Set[int]],
    weights: Dict[int, float],
    rng: random.Random,
    init_solution: Optional[Set[int]] = None,
) -> HilsState:
    nodes = list(adj.keys())
    state = HilsState(adj, weights, nodes)

    if init_solution is not None:
        for v in init_solution:
            if state.is_free(v):
                state.add_to_solution(v)
    else:
        # Greedy weighted init: decreasing weight order
        for v in sorted(nodes, key=lambda x: -weights[x]):
            if state.is_free(v):
                state.add_to_solution(v)

    state.update_best()
    return state


def _make_maximal(state: HilsState, rng: random.Random) -> None:
    """Insert all free nodes (shuffled) to make solution maximal."""
    free = [v for v in state.nodes if state.is_free(v)]
    rng.shuffle(free)
    for v in free:
        if state.is_free(v):   # re-check after previous insertions changed tight_counts
            state.add_to_solution(v)


# ── Neighbourhood N1: (ω,1)-swap ─────────────────────────────────────────────

def _vnd_omega_1_swap(state: HilsState, config: HilsConfig) -> bool:
    """
    For each v ∉ S: if w(v) > c1 * sum(w(S-neighbors of v)), perform swap.
    Removes all S-neighbors of v and inserts v.
    Returns True if any improving swap found.
    """
    improved = False
    for v in state.nodes:
        if state.elapsed() >= config.time_limit:
            break
        if v in state.solution:
            continue
        s_nbrs = [u for u in state.adj[v] if u in state.solution]
        nbr_weight = sum(state.weights[u] for u in s_nbrs)
        if state.weights[v] > config.c1 * nbr_weight:
            for u in s_nbrs:
                state.remove_from_solution(u)
            state.add_to_solution(v)
            improved = True
    return improved


# ── Neighbourhood N2: (1,2)-swap  ────────────────────────────────────────────

def _vnd_1_2_swap(state: HilsState, config: HilsConfig = None) -> bool:
    """
    For each one-tight non-solution node u (has exactly one S-neighbor v):
      Remove v, look for two free nodes x, y (non-adjacent) with w(x)+w(y) > w(v).
      If found: swap {v} -> {x, y}.
    Follows direct_improvement() from KaMIS local_search.h.
    Returns True if any improving swap found.
    """
    improved = False
    one_tight = state.one_tight_nodes()
    for u in one_tight:
        if config is not None and state.elapsed() >= config.time_limit:
            break
        if u in state.solution:
            continue
        if state.tight_count.get(u, 0) != 1:
            continue
        # Find the single S-neighbor of u
        s_nbrs = [w for w in state.adj[u] if w in state.solution]
        if len(s_nbrs) != 1:
            continue
        v = s_nbrs[0]
        wv = state.weights[v]

        # Tentatively remove v: nodes that become free
        state.remove_from_solution(v)

        # Collect free candidates (tight_count == 0 now, not in S)
        free_cands = [x for x in state.nodes
                      if x not in state.solution and state.tight_count.get(x, 0) == 0]

        # Sort by decreasing weight for greedy pair selection
        free_cands.sort(key=lambda x: -state.weights[x])

        # Find best non-adjacent pair (x, y) with w(x)+w(y) > w(v)
        found = False
        for i, x in enumerate(free_cands):
            if state.weights[x] * 2 <= wv:
                break   # even best pair can't beat w(v)
            for y in free_cands[i+1:]:
                if state.weights[x] + state.weights[y] > wv:
                    if y not in state.adj[x]:  # non-adjacent
                        # Perform swap: insert x and y
                        state.add_to_solution(x)
                        if state.is_free(y):
                            state.add_to_solution(y)
                            found = True
                            improved = True
                            break
                        else:
                            state.remove_from_solution(x)
                if found:
                    break
            if found:
                break

        if not found:
            # Revert: put v back
            state.add_to_solution(v)

    return improved


# ── VND Local Search ──────────────────────────────────────────────────────────

def _local_search(
    state: HilsState,
    config: HilsConfig,
    rng: random.Random,
) -> None:
    """
    Variable Neighbourhood Descent (VND) over N1 and N2.
    Restart from N1 whenever any improvement is found.
    Respects time_limit to prevent runaway on dense graphs.
    """
    k = 1
    while k <= 2:
        if state.elapsed() >= config.time_limit:
            break
        if k == 1:
            improved = _vnd_omega_1_swap(state, config)
        else:
            improved = _vnd_1_2_swap(state, config)

        if improved:
            k = 1
            _make_maximal(state, rng)
        else:
            k += 1
    _make_maximal(state, rng)


# ── Perturbation ──────────────────────────────────────────────────────────────

def _perturb(
    state: HilsState,
    strength: int,
    config: HilsConfig,
    rng: random.Random,
) -> None:
    """
    Force `strength` non-solution nodes into S (least-recently-forced first).
    Mirrors ils::force() in KaMIS.
    """
    non_sol = [v for v in state.nodes if v not in state.solution]
    if not non_sol:
        return
    # Sort by last_forced ascending (least recently forced = best candidate)
    non_sol.sort(key=lambda v: state.last_forced[v])
    # Pick first `strength` candidates (add small shuffle for ties)
    pool = non_sol[:max(strength * 2, config.force_candidates)]
    rng.shuffle(pool)
    chosen = pool[:strength]

    for v in chosen:
        if v in state.solution:
            continue
        state.force_node(v)
        state.last_forced[v] = state.iteration

    _make_maximal(state, rng)


# ── Acceptance criterion (simplified from ils.cpp) ───────────────────────────

def _accept(
    state: HilsState,
    snap_before: Tuple[Set[int], float],
    acc_i: List[int],       # mutable counter [i]
    config: HilsConfig,
    rng: random.Random,
) -> None:
    """
    Compare current solution to snapshot taken before perturbation.
    - If improved: update best, reward counter.
    - Elif within patience: accept, increment counter.
    - Else: revert to best, restart with stronger perturbation.
    Modifies state in-place.
    """
    s_size = len(state.solution)
    patience = max(1, s_size // config.c2)

    if state.solution_weight > snap_before[1]:
        # Improved over pre-perturbation baseline
        state.update_best()
        acc_i[0] = max(1, acc_i[0] - s_size // config.c2)
        if state.solution_weight >= state.best_weight:
            acc_i[0] = max(1, acc_i[0] - s_size * config.c3)
    elif acc_i[0] <= patience:
        acc_i[0] += 1
    else:
        # Patience exceeded: revert to best and re-perturb harder
        state.restore((state.best_solution, state.best_weight))
        _perturb(state, config.c4, config, rng)
        _local_search(state, config, rng)
        state.update_best()
        acc_i[0] = 1


# ── Main HILS loop ────────────────────────────────────────────────────────────

def hils_weighted(
    G: nx.Graph,
    weights: Dict[int, float],
    config: HilsConfig = None,
    init_solution: Optional[Set[int]] = None,
) -> Tuple[Set[int], float]:
    """
    HILS for Maximum Weight Independent Set.
    Returns (best_solution, best_weight).
    """
    if config is None:
        config = HilsConfig()
    rng = random.Random(config.seed)

    if G.number_of_nodes() == 0:
        return set(), 0.0

    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    # ensure all nodes present in adj (isolated nodes may be missing)
    for v in G.nodes():
        adj.setdefault(v, set())

    state = _build_state(adj, weights, rng, init_solution)

    # Initial local search
    _local_search(state, config, rng)
    state.update_best()

    acc_i = [1]  # mutable acceptance counter

    for it in range(1, config.max_iter + 1):
        state.iteration = it
        if state.elapsed() >= config.time_limit:
            break

        snap = state.snapshot()
        _perturb(state, config.c1, config, rng)
        _local_search(state, config, rng)
        state.update_best()
        _accept(state, snap, acc_i, config, rng)

    return state.best_solution, state.best_weight


def hils_unweighted(
    G: nx.Graph,
    config: HilsConfig = None,
    init_solution: Optional[Set[int]] = None,
) -> Tuple[Set[int], float]:
    """
    HILS with all weights = 1 (Maximum Independent Set).
    Returns (best_solution, best_size_as_float).
    Note: in unweighted mode N1 is inactive; only N2 (1,2-swap) fires.
    """
    weights = {v: 1.0 for v in G.nodes()}
    return hils_weighted(G, weights, config, init_solution)


# ── Red+HILS ──────────────────────────────────────────────────────────────────

def red_hils_weighted(
    G: nx.Graph,
    weights: Dict[int, float],
    config: HilsConfig = None,
) -> Tuple[Set[int], float]:
    """
    Red+HILS pipeline:
      1. Apply weighted reduction rules to shrink G to a kernel.
      2. Run hils_weighted on the kernel.
      3. Lift the kernel solution back to the original graph.
    Returns (full_solution, full_weight).
    """
    if config is None:
        config = HilsConfig()

    if G.number_of_nodes() == 0:
        return set(), 0.0

    # Step 1: Build reducible graph and apply reductions
    g_red = ReducibleGraph.from_graph(G, weights)
    stack = []
    reduce_graph(g_red, stack)

    # Step 2: Extract compact kernel
    G_kernel, old_to_new, new_to_old = g_red.to_nx_graph()
    kernel_weights = {
        new_to_old.get(v, v): g_red.weights[new_to_old.get(v, v)]
        for v in G_kernel.nodes()
    }
    # Remap weights to kernel IDs
    kernel_weights_remapped = {
        v: g_red.weights[new_to_old[v]]
        for v in G_kernel.nodes()
    }

    # Step 3: Run HILS on kernel (adjust time limit for reduction time)
    kernel_sol: Set[int] = set()
    if G_kernel.number_of_nodes() > 0:
        kernel_sol_orig, _ = hils_weighted(G_kernel, kernel_weights_remapped, config)
        kernel_sol = kernel_sol_orig

    # Step 4: Lift solution back to original graph
    full_sol = lift_solution(kernel_sol, stack, new_to_old, g_red)

    full_weight = sum(weights.get(v, 0.0) for v in full_sol)
    return full_sol, full_weight


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_hils_solver(
    inst,                      # MISInstance or WeightedMISInstance
    mode: str,                 # "unweighted" | "weighted" | "red_weighted"
    config: HilsConfig = None,
    time_limit: float = 30.0,
) -> Tuple[Set[int], float]:
    """
    Unified entry point for all experiment scripts.
    Returns (solution_set, runtime_seconds).
    """
    if config is None:
        config = HilsConfig(time_limit=time_limit)
    else:
        config.time_limit = time_limit

    G = inst.graph if hasattr(inst, 'graph') else inst.base.graph

    t0 = time.time()
    if mode == "unweighted":
        sol, _ = hils_unweighted(G, config)
    elif mode == "weighted":
        weights = inst.weights if hasattr(inst, 'weights') else {v: 1.0 for v in G.nodes()}
        sol, _ = hils_weighted(G, weights, config)
    elif mode == "red_weighted":
        weights = inst.weights if hasattr(inst, 'weights') else {v: 1.0 for v in G.nodes()}
        sol, _ = red_hils_weighted(G, weights, config)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    elapsed = time.time() - t0
    return sol, elapsed


# ── Correctness checks ────────────────────────────────────────────────────────

def _verify_is(G: nx.Graph, sol: Set[int]) -> bool:
    for u in sol:
        for v in sol:
            if u != v and G.has_edge(u, v):
                return False
    return True


def correctness_check() -> bool:
    """Quick sanity checks on trivial graphs."""
    import networkx as nx

    # K_n: only one node can be in IS
    for n in [3, 5, 10]:
        G = nx.complete_graph(n)
        sol, _ = hils_unweighted(G, HilsConfig(max_iter=1000, time_limit=5.0))
        assert len(sol) == 1, f"K_{n}: expected IS size 1, got {len(sol)}"
        assert _verify_is(G, sol)

    # Empty graph: all nodes in IS
    G = nx.empty_graph(10)
    sol, _ = hils_unweighted(G, HilsConfig(max_iter=1000, time_limit=5.0))
    assert len(sol) == 10, f"Empty graph n=10: expected IS size 10, got {len(sol)}"
    assert _verify_is(G, sol)

    # Path graph P_n: IS size = ceil(n/2)
    for n in [4, 6, 10]:
        G = nx.path_graph(n)
        sol, _ = hils_unweighted(G, HilsConfig(max_iter=10_000, time_limit=10.0))
        expected = (n + 1) // 2
        assert len(sol) >= expected, f"P_{n}: expected IS >= {expected}, got {len(sol)}"
        assert _verify_is(G, sol), f"P_{n}: invalid IS {sol}"

    # Red+HILS on weighted K3
    G = nx.complete_graph(3)
    weights = {0: 10.0, 1: 1.0, 2: 1.0}
    sol, w = red_hils_weighted(G, weights, HilsConfig(max_iter=1000, time_limit=5.0))
    assert 0 in sol and len(sol) == 1, f"Weighted K3: expected {{0}}, got {sol}"
    assert w == 10.0, f"Weighted K3 weight: expected 10.0, got {w}"

    return True


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("hils.py self-test")
    ok = correctness_check()
    print(f"  correctness_check: {'PASS' if ok else 'FAIL'}")

    # Quick benchmark on a small random graph
    import networkx as nx
    G = nx.erdos_renyi_graph(50, 0.3, seed=42)
    weights = {v: float(v % 10 + 1) for v in G.nodes()}

    t0 = time.time()
    sol_u, w_u = hils_unweighted(G, HilsConfig(max_iter=50_000, time_limit=5.0, seed=42))
    print(f"  HILS unweighted  n=50: IS size={len(sol_u)}, valid={_verify_is(G,sol_u)}, "
          f"t={time.time()-t0:.2f}s")

    t0 = time.time()
    sol_w, w_w = hils_weighted(G, weights, HilsConfig(max_iter=50_000, time_limit=5.0, seed=42))
    print(f"  HILS weighted    n=50: weight={w_w:.1f}, valid={_verify_is(G,sol_w)}, "
          f"t={time.time()-t0:.2f}s")

    t0 = time.time()
    sol_r, w_r = red_hils_weighted(G, weights, HilsConfig(max_iter=50_000, time_limit=5.0, seed=42))
    print(f"  Red+HILS weighted n=50: weight={w_r:.1f}, valid={_verify_is(G,sol_r)}, "
          f"t={time.time()-t0:.2f}s")
