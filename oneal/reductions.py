#!/usr/bin/env python3
"""
reductions.py
=============
Weighted graph reduction rules with an undo stack for Red+HILS.

Implements 5 core rules from Lamm et al., Section 5:
  1. Neighborhood Removal (Reduction 4)
  2. Weighted Domination  (Reduction 9)
  3. Vertex Folding       (Reduction 7, deg-2)
  4. Weighted Twin        (Reduction 8)
  5. Isolated Vertex Removal (Reduction 5, deg <= max_degree)

Each rule returns True if it fired at least once.
reduce_graph() applies rules exhaustively in priority order.
lift_solution() unwinds the undo stack to recover the full IS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import networkx as nx


# ── Mutable working graph ─────────────────────────────────────────────────────

@dataclass
class ReducibleGraph:
    adj: Dict[int, Set[int]]             # active adjacency (dict-of-sets)
    weights: Dict[int, float]            # node -> weight
    active: Set[int]                     # currently live nodes
    is_status: Dict[int, Optional[bool]] # True=IS, False=excluded, None=unknown
    n_original: int
    next_node_id: int = field(default=0) # counter for fold meta-nodes

    @classmethod
    def from_graph(
        cls,
        G: nx.Graph,
        weights: Dict[int, float],
    ) -> "ReducibleGraph":
        adj = {v: set(G.neighbors(v)) for v in G.nodes()}
        w   = {v: float(weights.get(v, 1.0)) for v in G.nodes()}
        active = set(G.nodes())
        is_status = {v: None for v in G.nodes()}
        n = G.number_of_nodes()
        # meta-node IDs start after original IDs
        next_id = max(active) + 1 if active else 0
        return cls(adj=adj, weights=w, active=active,
                   is_status=is_status, n_original=n, next_node_id=next_id)

    def degree(self, v: int) -> int:
        # adj is kept clean by remove_node, no need to intersect with active
        return len(self.adj.get(v, set()))

    def neighbors(self, v: int) -> Set[int]:
        # adj is kept clean by remove_node, no need to intersect with active
        return self.adj.get(v, set())

    def remove_node(self, v: int) -> None:
        """Remove v from active set and from all neighbor adjacency lists."""
        self.active.discard(v)
        for u in list(self.adj.get(v, set())):
            self.adj[u].discard(v)

    def include_in_is(self, v: int) -> None:
        """Mark v as in IS, then exclude all active neighbors."""
        self.is_status[v] = True
        for u in list(self.neighbors(v)):
            self.exclude_from_is(u)
        self.remove_node(v)

    def exclude_from_is(self, v: int) -> None:
        """Mark v as excluded from IS and remove from graph."""
        self.is_status[v] = False
        self.remove_node(v)

    def add_meta_node(
        self,
        node_id: int,
        weight: float,
        neighbors: Set[int],
    ) -> None:
        """Add a new meta-node (used by folding reductions)."""
        self.active.add(node_id)
        self.weights[node_id] = weight
        self.is_status[node_id] = None
        self.adj[node_id] = neighbors & self.active
        for u in self.adj[node_id]:
            self.adj[u].add(node_id)

    def to_nx_graph(self) -> Tuple[nx.Graph, Dict[int, int], Dict[int, int]]:
        """
        Build a compact nx.Graph from active nodes.
        Returns (G_kernel, old_to_new, new_to_old) mappings.
        """
        active_sorted = sorted(self.active)
        old_to_new = {v: i for i, v in enumerate(active_sorted)}
        new_to_old = {i: v for i, v in enumerate(active_sorted)}
        G = nx.Graph()
        G.add_nodes_from(range(len(active_sorted)))
        for v in active_sorted:
            nv = old_to_new[v]
            for u in self.neighbors(v):
                nu = old_to_new[u]
                if nv < nu:
                    G.add_edge(nv, nu)
        return G, old_to_new, new_to_old


# ── Undo stack entries ────────────────────────────────────────────────────────

@dataclass
class UndoNeighborhoodRemoval:
    """v was included in IS; its neighbors were excluded."""
    primary: int             # v
    excluded_neighbors: List[int]

@dataclass
class UndoWeightedDomination:
    """dominated was removed (not in IS); dominator stays."""
    dominated: int
    dominator: int

@dataclass
class UndoVertexFolding:
    """
    deg-2 fold: {v, u, w} -> meta-node m (stored at ID meta_id).
    w(m) = w(u) + w(w) - w(v).
    If m in IS  -> {u, w} in IS, v not in IS.
    If m not IS -> {v} in IS, u and w not in IS.
    """
    v: int
    u: int
    w: int
    meta_id: int
    orig_weight_v: float
    orig_weight_u: float
    orig_weight_w: float
    meta_neighbors: List[int]  # N(m) at time of folding

@dataclass
class UndoWeightedTwin:
    """
    twin1 and twin2 non-adjacent with identical neighborhoods.
    Folded into meta-node meta_id.
    w(meta) = w(twin1) + w(twin2).
    N(meta) = N(twin1) = N(twin2).
    If meta in IS  -> {twin1, twin2} in IS.
    If meta not IS -> nothing forced (covered by neighborhood reduction later).
    """
    twin1: int
    twin2: int
    meta_id: int
    common_neighbors: List[int]
    orig_weight_t1: float
    orig_weight_t2: float

@dataclass
class UndoIsolatedRemoval:
    """
    v was included in IS (it dominated its small neighborhood).
    Its neighbors were excluded.
    """
    primary: int
    excluded_neighbors: List[int]


UndoEntry = (
    UndoNeighborhoodRemoval
    | UndoWeightedDomination
    | UndoVertexFolding
    | UndoWeightedTwin
    | UndoIsolatedRemoval
)


# ── Reduction rule 1: Neighborhood Removal ───────────────────────────────────

def apply_neighborhood_removal(
    g: ReducibleGraph,
    stack: List,
) -> bool:
    """
    If w(v) >= sum(w(N(v))), include v in IS and exclude all neighbors.
    Fires on isolated vertices (N(v) = {} -> 0 <= w(v) always).
    """
    fired = False
    candidates = list(g.active)
    for v in candidates:
        if v not in g.active:
            continue
        nbrs = g.neighbors(v)
        nbr_weight = sum(g.weights[u] for u in nbrs)
        if g.weights[v] >= nbr_weight:
            excl = list(nbrs)
            stack.append(UndoNeighborhoodRemoval(primary=v, excluded_neighbors=excl))
            g.include_in_is(v)
            fired = True
    return fired


# ── Reduction rule 2: Weighted Domination ────────────────────────────────────

def apply_weighted_domination(
    g: ReducibleGraph,
    stack: List,
) -> bool:
    """
    If N[u] ⊆ N[v] and w(u) <= w(v), remove u (dominated by v).
    Only checks pairs (u, v) where v ∈ N(u) and deg(v) >= deg(u).
    """
    fired = False
    candidates = list(g.active)
    for u in candidates:
        if u not in g.active:
            continue
        nbrs_u = g.neighbors(u)
        deg_u = len(nbrs_u)
        Nu_closed = nbrs_u | {u}
        # Check each neighbor v of u as potential dominator
        for v in list(nbrs_u):
            if v not in g.active:
                continue
            # Degree pruning: N[u] ⊆ N[v] requires deg(v) >= deg(u)
            if g.degree(v) < deg_u:
                continue
            # Weight pruning: need w(u) <= w(v)
            if g.weights[u] > g.weights[v]:
                continue
            Nv_closed = g.neighbors(v) | {v}
            # v dominates u iff N[u] ⊆ N[v] and w(u) <= w(v)
            if Nu_closed <= Nv_closed:
                stack.append(UndoWeightedDomination(dominated=u, dominator=v))
                g.exclude_from_is(u)
                fired = True
                break
    return fired


# ── Reduction rule 3: Vertex Folding (deg-2) ─────────────────────────────────

def apply_vertex_folding(
    g: ReducibleGraph,
    stack: List,
) -> bool:
    """
    If deg(v) == 2, neighbors u and w are non-adjacent, and
    max(w(u), w(w)) <= w(v) < w(u)+w(w):
      fold {v, u, w} into meta-node m with w(m) = w(u)+w(w)-w(v).
    """
    fired = False
    candidates = list(g.active)
    for v in candidates:
        if v not in g.active:
            continue
        nbrs = g.neighbors(v)
        if len(nbrs) != 2:
            continue
        u, w = list(nbrs)
        if u not in g.active or w not in g.active:
            continue
        # u and w must be non-adjacent
        if w in g.neighbors(u):
            continue
        wv, wu, ww = g.weights[v], g.weights[u], g.weights[w]
        if wv >= max(wu, ww) and wv < wu + ww:
            # Fold: meta-node replaces {v, u, w}
            meta_id = g.next_node_id
            g.next_node_id += 1
            meta_weight = wu + ww - wv
            # N(meta) = (N(u) ∪ N(w)) \ {v, u, w}
            meta_nbrs = (g.neighbors(u) | g.neighbors(w)) - {v, u, w}
            record = UndoVertexFolding(
                v=v, u=u, w=w, meta_id=meta_id,
                orig_weight_v=wv, orig_weight_u=wu, orig_weight_w=ww,
                meta_neighbors=list(meta_nbrs),
            )
            stack.append(record)
            # Remove original nodes
            g.remove_node(v)
            g.remove_node(u)
            g.remove_node(w)
            # Add meta-node
            g.add_meta_node(meta_id, meta_weight, meta_nbrs)
            fired = True
    return fired


# ── Reduction rule 4: Weighted Twin ──────────────────────────────────────────

def apply_weighted_twin(
    g: ReducibleGraph,
    stack: List,
) -> bool:
    """
    If twin1 and twin2 are non-adjacent and share identical neighborhoods,
    fold them into a single meta-node with w(meta) = w(t1)+w(t2).
    Picking either twin (or neither) is equivalent for the remaining graph.
    """
    fired = False
    candidates = list(g.active)
    # Build neighborhood signature -> list of nodes
    from collections import defaultdict
    nbr_sig: Dict = defaultdict(list)
    for v in candidates:
        if v not in g.active:
            continue
        sig = frozenset(g.neighbors(v))
        nbr_sig[sig].append(v)

    for sig, nodes in nbr_sig.items():
        if len(nodes) < 2:
            continue
        # Check all pairs in this group
        processed = set()
        for i in range(len(nodes)):
            t1 = nodes[i]
            if t1 not in g.active or t1 in processed:
                continue
            for j in range(i + 1, len(nodes)):
                t2 = nodes[j]
                if t2 not in g.active or t2 in processed:
                    continue
                # Must be non-adjacent
                if t2 in g.neighbors(t1):
                    continue
                # Fold t1, t2 into meta
                meta_id = g.next_node_id
                g.next_node_id += 1
                meta_weight = g.weights[t1] + g.weights[t2]
                common_nbrs = list(sig & g.active)
                record = UndoWeightedTwin(
                    twin1=t1, twin2=t2, meta_id=meta_id,
                    common_neighbors=common_nbrs,
                    orig_weight_t1=g.weights[t1],
                    orig_weight_t2=g.weights[t2],
                )
                stack.append(record)
                g.remove_node(t1)
                g.remove_node(t2)
                g.add_meta_node(meta_id, meta_weight, sig)
                processed.add(t1)
                fired = True
                break   # only fold one pair per signature per pass
    return fired


# ── Reduction rule 5: Isolated / Low-degree vertex removal ───────────────────

def apply_isolated_vertex_removal(
    g: ReducibleGraph,
    stack: List,
    max_degree: int = 3,
) -> bool:
    """
    If deg(v) <= max_degree, N(v) is a clique, and w(v) >= max(w(N(v))):
      include v in IS, exclude N(v).
    Covers isolated vertices (deg=0) which always satisfy the condition.
    """
    fired = False
    candidates = list(g.active)
    for v in candidates:
        if v not in g.active:
            continue
        nbrs = g.neighbors(v)
        d = len(nbrs)
        if d > max_degree:
            continue
        # Check if N(v) is a clique
        nbr_list = list(nbrs)
        is_clique = True
        for a in range(len(nbr_list)):
            for b in range(a + 1, len(nbr_list)):
                if nbr_list[b] not in g.neighbors(nbr_list[a]):
                    is_clique = False
                    break
            if not is_clique:
                break
        if not is_clique:
            continue
        # Check weight dominance
        max_nbr_w = max((g.weights[u] for u in nbrs), default=0.0)
        if g.weights[v] >= max_nbr_w:
            excl = list(nbrs)
            stack.append(UndoIsolatedRemoval(primary=v, excluded_neighbors=excl))
            g.include_in_is(v)
            fired = True
    return fired


# ── Reduction driver ──────────────────────────────────────────────────────────

_RULES = [
    apply_neighborhood_removal,
    apply_isolated_vertex_removal,
    apply_weighted_domination,
    apply_vertex_folding,
    apply_weighted_twin,
]


def reduce_graph(
    g: ReducibleGraph,
    stack: List,
    rules=None,
) -> int:
    """
    Apply reduction rules exhaustively (restart from top when any fires).
    Returns total number of reductions applied.
    """
    rules = rules or _RULES
    total = 0
    changed = True
    while changed and g.active:
        changed = False
        for rule_fn in rules:
            if rule_fn(g, stack):
                changed = True
                total += 1
                break   # restart from first rule
    return total


# ── Solution lifting ──────────────────────────────────────────────────────────

def lift_solution(
    reduced_solution: Set[int],   # IS in (possibly remapped) kernel IDs
    stack: List,
    new_to_old: Dict[int, int],   # kernel id -> original graph id
    g: ReducibleGraph,            # needed for is_status of non-meta nodes
) -> Set[int]:
    """
    Unwind the undo stack (reverse order) to extend kernel IS to full IS.
    Returns set of original node IDs.
    """
    # Start from the reduced solution, mapped back to original IDs
    in_is: Dict[int, bool] = {}
    for kid, oid in new_to_old.items():
        in_is[oid] = (kid in reduced_solution)

    # Also incorporate nodes already decided by reductions (is_status)
    for v, status in g.is_status.items():
        if status is True:
            in_is[v] = True
        elif status is False:
            in_is[v] = False

    # Unwind in reverse
    for entry in reversed(stack):
        if isinstance(entry, UndoNeighborhoodRemoval):
            # primary was included; neighbors were excluded
            in_is[entry.primary] = True
            for u in entry.excluded_neighbors:
                in_is[u] = False

        elif isinstance(entry, UndoIsolatedRemoval):
            in_is[entry.primary] = True
            for u in entry.excluded_neighbors:
                in_is[u] = False

        elif isinstance(entry, UndoWeightedDomination):
            # dominated was removed; it is NOT in IS
            in_is[entry.dominated] = False

        elif isinstance(entry, UndoVertexFolding):
            # meta_id result -> decide v, u, w
            meta_in = in_is.get(entry.meta_id, False)
            if meta_in:
                # meta in IS -> {u, w} in IS, v not
                in_is[entry.u] = True
                in_is[entry.w] = True
                in_is[entry.v] = False
            else:
                # meta not in IS -> v in IS, u and w not
                in_is[entry.v] = True
                in_is[entry.u] = False
                in_is[entry.w] = False
            # Remove meta_id entry
            in_is.pop(entry.meta_id, None)

        elif isinstance(entry, UndoWeightedTwin):
            meta_in = in_is.get(entry.meta_id, False)
            if meta_in:
                # Both twins in IS
                in_is[entry.twin1] = True
                in_is[entry.twin2] = True
            else:
                in_is[entry.twin1] = False
                in_is[entry.twin2] = False
            in_is.pop(entry.meta_id, None)

    return {v for v, status in in_is.items() if status is True}


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("reductions.py self-test")

    # Test 1: Neighborhood removal on a star graph
    # Center (0) connects to 1,2,3. w(0)=10, w(1)=w(2)=w(3)=1
    # Sum of neighbor weights = 3 < 10, so 0 should be included
    G = nx.Graph()
    G.add_edges_from([(0,1),(0,2),(0,3)])
    weights = {0: 10.0, 1: 1.0, 2: 1.0, 3: 1.0}
    g = ReducibleGraph.from_graph(G, weights)
    stack = []
    fired = apply_neighborhood_removal(g, stack)
    assert fired, "Neighborhood removal should fire on star"
    sol = lift_solution(set(), stack, {}, g)
    assert 0 in sol, f"Node 0 should be in IS, got {sol}"
    assert not any(v in sol for v in [1,2,3]), f"Neighbors should be excluded, got {sol}"
    print("  Test 1 PASS: neighborhood removal on star")

    # Test 2: Vertex folding on P3 path
    # 1 - 0 - 2, all weights = 1
    # deg(0) = 2, N(0) = {1,2}, w(0) = 1 >= max(1,1) = 1 and < 1+1=2
    G2 = nx.path_graph(3)  # 0-1-2
    weights2 = {0: 1.0, 1: 1.0, 2: 1.0}
    g2 = ReducibleGraph.from_graph(G2, weights2)
    stack2 = []
    n_red = reduce_graph(g2, stack2)
    assert n_red > 0, "Should reduce P3"
    print(f"  Test 2: P3 reduced in {n_red} steps, active={g2.active}")

    # Test 3: Weighted domination
    # Triangle: 0-1, 1-2, 0-2. w = {0:3, 1:1, 2:1}
    # N[1] = {0,1,2} = N[2] since it's a triangle; w(1) = w(2) = 1
    # Neighborhood removal fires first: 0 dominates with w(0)=3 >= w(1)+w(2)=2
    G3 = nx.complete_graph(3)
    weights3 = {0: 3.0, 1: 1.0, 2: 1.0}
    g3 = ReducibleGraph.from_graph(G3, weights3)
    stack3 = []
    n_red3 = reduce_graph(g3, stack3)
    sol3 = lift_solution(set(), stack3, {}, g3)
    assert 0 in sol3, f"Node 0 should be in IS for K3 with weights {weights3}, got {sol3}"
    print(f"  Test 3 PASS: K3 weighted -> IS = {sol3}")

    print("  All tests passed.")
