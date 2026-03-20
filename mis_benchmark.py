#!/usr/bin/env python3
"""
MIS Classical Benchmarking Suite for Quantum-Classical Comparison
================================================================
Complete implementation for generating planted MIS instances,
running classical solvers, computing the ET parameter, and
analyzing results.

Dependencies: numpy, networkx, scipy, matplotlib
Optional: python-sat (pysat) for SAT-based solving

Install: pip install numpy networkx scipy matplotlib python-sat
"""

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh
from itertools import combinations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from abc import ABC, abstractmethod
import time
import warnings
import json
import os

# ============================================================
# SECTION 1: INSTANCE GENERATION
# ============================================================

@dataclass
class MISInstance:
    """A planted MIS instance with metadata."""
    graph: nx.Graph
    planted_set: Set[int]
    n: int
    h: int  # planted set size
    family: str
    params: Dict
    seed: int
    instance_id: str = ""

    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = f"{self.family}_n{self.n}_h{self.h}_s{self.seed}"

    def verify(self) -> bool:
        """Verify the planted set is indeed independent."""
        G = self.graph
        S = self.planted_set
        for u in S:
            for v in S:
                if u != v and G.has_edge(u, v):
                    return False
        return True


class InstanceGenerator:
    """
    Generates planted MIS instances across multiple families.

    Families:
      - erdos_renyi_planted: G(n,p) with planted IS of size h
      - regular_planted: random d-regular with planted IS
      - multi_clique_core: construction from hard-instance notes
      - camouflaged_clique: multi-clique with degree equalization
      - geometric_planted: random geometric graph with planted IS
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.base_seed = seed

    def generate(self, family: str, n: int, h: int,
                 params: Optional[Dict] = None, seed: Optional[int] = None) -> MISInstance:
        """Generate a single instance."""
        if seed is None:
            seed = self.rng.randint(0, 2**31)
        if params is None:
            params = {}

        generators = {
            'erdos_renyi_planted': self._erdos_renyi_planted,
            'regular_planted': self._regular_planted,
            'multi_clique_core': self._multi_clique_core,
            'camouflaged_clique': self._camouflaged_clique,
            'geometric_planted': self._geometric_planted,
        }

        if family not in generators:
            raise ValueError(f"Unknown family: {family}. Options: {list(generators.keys())}")

        return generators[family](n, h, params, seed)

    def _erdos_renyi_planted(self, n: int, h: int, params: Dict, seed: int) -> MISInstance:
        """
        Erdos-Renyi graph with planted independent set.
        Edges within S are removed; edges elsewhere with probability p.

        params:
          - p: edge probability (default: 0.5)
        """
        rng = np.random.RandomState(seed)
        p = params.get('p', 0.5)

        vertices = list(range(n))
        S = set(rng.choice(vertices, size=h, replace=False))

        G = nx.Graph()
        G.add_nodes_from(vertices)

        for i in range(n):
            for j in range(i + 1, n):
                if i in S and j in S:
                    continue  # no edges within planted set
                if rng.random() < p:
                    G.add_edge(i, j)

        return MISInstance(
            graph=G, planted_set=S, n=n, h=h,
            family='erdos_renyi_planted',
            params={'p': p}, seed=seed
        )

    def _regular_planted(self, n: int, h: int, params: Dict, seed: int) -> MISInstance:
        """
        Approximate d-regular graph with planted independent set.
        Start with random d-regular on V\S, then connect S vertices
        to achieve near-uniform degree.

        params:
          - d: target degree (default: n//2)
        """
        rng = np.random.RandomState(seed)
        d = params.get('d', max(3, n // 4))

        vertices = list(range(n))
        S = set(rng.choice(vertices, size=h, replace=False))
        R = [v for v in vertices if v not in S]

        G = nx.Graph()
        G.add_nodes_from(vertices)

        # Connect R densely (near-complete to avoid large IS in R)
        p_R = min(1.0, d / max(len(R) - 1, 1))
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                if rng.random() < p_R:
                    G.add_edge(R[i], R[j])

        # Connect S to R to achieve approximate degree d
        for v in S:
            # Connect to approximately d vertices in R
            n_connections = min(d, len(R))
            targets = rng.choice(R, size=n_connections, replace=False)
            for u in targets:
                G.add_edge(v, u)

        return MISInstance(
            graph=G, planted_set=S, n=n, h=h,
            family='regular_planted',
            params={'d': d}, seed=seed
        )

    def _multi_clique_core(self, n: int, h: int, params: Dict, seed: int) -> MISInstance:
        """
        Multi-Clique Core construction from hard instance notes.

        V\S is partitioned into q cliques. Each clique C_j is
        completely connected to a block B_j ⊂ S of size b.
        Additional camouflage edges connect C_j to S\B_j.

        params:
          - q: number of clique blocks (default: 3)
          - b: blocking size per clique (default: 2)
          - p_inter: inter-block edge probability (default: 0.5)
          - p_cam: camouflage edge probability (default: 0.3)
        """
        rng = np.random.RandomState(seed)
        q = params.get('q', 3)
        b = params.get('b', 2)
        p_inter = params.get('p_inter', 0.5)
        p_cam = params.get('p_cam', 0.3)

        assert q * b <= h, f"Need q*b <= h, got {q}*{b} > {h}"
        assert h <= n, f"Need h <= n"

        vertices = list(range(n))
        S = set(range(h))  # planted set = first h vertices
        R = list(range(h, n))  # remaining vertices

        G = nx.Graph()
        G.add_nodes_from(vertices)

        # Partition R into q blocks (as evenly as possible)
        block_size = len(R) // q
        blocks = []
        for j in range(q):
            start = j * block_size
            end = start + block_size if j < q - 1 else len(R)
            blocks.append(R[start:end])

        # Make each block a clique
        for block in blocks:
            for i in range(len(block)):
                for j in range(i + 1, len(block)):
                    G.add_edge(block[i], block[j])

        # Inter-block edges
        for a in range(q):
            for bb in range(a + 1, q):
                for u in blocks[a]:
                    for v in blocks[bb]:
                        if rng.random() < p_inter:
                            G.add_edge(u, v)

        # Blocking: assign disjoint B_j ⊂ S to each block
        S_list = sorted(S)
        B_sets = []
        for j in range(q):
            B_j = set(S_list[j * b: (j + 1) * b])
            B_sets.append(B_j)

        # Connect each clique to its blocking set completely
        for j in range(q):
            for u in blocks[j]:
                for v in B_sets[j]:
                    G.add_edge(u, v)

        # Camouflage: connect clique vertices to S \ B_j
        for j in range(q):
            S_minus_Bj = S - B_sets[j]
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

        return MISInstance(
            graph=G, planted_set=S, n=n, h=h,
            family='multi_clique_core',
            params={'q': q, 'b': b, 'p_inter': p_inter, 'p_cam': p_cam},
            seed=seed
        )

    def _camouflaged_clique(self, n: int, h: int, params: Dict, seed: int) -> MISInstance:
        """
        Enhanced multi-clique core with degree equalization.
        After base construction, add/remove edges to make degree
        distribution of S and V\S overlap, defeating degree-based attacks.

        params: same as multi_clique_core plus
          - target_degree: target average degree for S vertices
        """
        # Start with multi-clique core
        inst = self._multi_clique_core(n, h, params, seed)
        G = inst.graph
        S = inst.planted_set
        rng = np.random.RandomState(seed + 1000)

        # Compute current degree statistics
        S_degrees = [G.degree(v) for v in S]
        R_degrees = [G.degree(v) for v in G.nodes() if v not in S]

        if not R_degrees:
            return inst

        target_deg = params.get('target_degree', int(np.median(R_degrees)))

        # For S vertices with degree < target, add edges to R
        R_list = [v for v in G.nodes() if v not in S]
        for v in S:
            current_deg = G.degree(v)
            deficit = target_deg - current_deg
            if deficit > 0:
                # Add edges to random R vertices (not already neighbors)
                non_neighbors = [u for u in R_list if not G.has_edge(v, u)]
                n_add = min(deficit, len(non_neighbors))
                if n_add > 0:
                    targets = rng.choice(non_neighbors, size=n_add, replace=False)
                    for u in targets:
                        G.add_edge(v, u)

        inst.family = 'camouflaged_clique'
        return inst

    def _geometric_planted(self, n: int, h: int, params: Dict, seed: int) -> MISInstance:
        """
        Random geometric graph with planted independent set.
        Vertices are points in [0,1]^dim; edges connect points within
        distance r. Planted set vertices are placed far apart.

        params:
          - dim: dimension (default: 2)
          - r: connection radius (default: auto)
        """
        rng = np.random.RandomState(seed)
        dim = params.get('dim', 2)
        r = params.get('r', None)

        if r is None:
            # Choose r to get moderate density
            r = 1.5 * (np.log(n) / n) ** (1.0 / dim)

        # Place planted set vertices on a grid-like structure (well-separated)
        S_points = np.zeros((h, dim))
        grid_side = int(np.ceil(h ** (1.0 / dim)))
        idx = 0
        for coords in np.ndindex(*([grid_side] * dim)):
            if idx >= h:
                break
            S_points[idx] = np.array(coords) / (grid_side + 1) + 0.5 / (grid_side + 1)
            idx += 1
        # Ensure planted points are > r apart (scale if needed)
        min_dist = np.inf
        for i in range(h):
            for j in range(i + 1, h):
                d = np.linalg.norm(S_points[i] - S_points[j])
                min_dist = min(min_dist, d)
        if min_dist < r * 1.1:
            S_points *= (r * 1.2) / min_dist
            S_points = S_points % 1.0  # wrap to [0,1]^dim

        # Place remaining vertices randomly
        R_points = rng.random((n - h, dim))
        all_points = np.vstack([S_points, R_points])

        # Build graph: connect points within distance r
        G = nx.Graph()
        G.add_nodes_from(range(n))
        S = set(range(h))

        for i in range(n):
            for j in range(i + 1, n):
                if i in S and j in S:
                    continue
                dist = np.linalg.norm(all_points[i] - all_points[j])
                # Use toroidal distance for wrap-around
                diff = np.abs(all_points[i] - all_points[j])
                diff = np.minimum(diff, 1.0 - diff)
                dist = np.linalg.norm(diff)
                if dist < r:
                    G.add_edge(i, j)

        # Random relabeling
        perm = list(range(n))
        rng.shuffle(perm)
        mapping = {old: new for old, new in zip(range(n), perm)}
        G = nx.relabel_nodes(G, mapping)
        S = {mapping[v] for v in S}

        return MISInstance(
            graph=G, planted_set=S, n=n, h=h,
            family='geometric_planted',
            params={'dim': dim, 'r': r}, seed=seed
        )

    def generate_suite(self, families: List[str], sizes: List[int],
                       h_ratios: List[float], seeds_per: int = 5) -> List[MISInstance]:
        """
        Generate a complete benchmark suite.

        Args:
            families: list of family names
            sizes: list of n values
            h_ratios: list of h/sqrt(n) ratios (e.g., [0.5, 1.0, 2.0])
            seeds_per: number of random seeds per configuration

        Returns:
            List of MISInstance objects
        """
        instances = []
        for family in families:
            for n in sizes:
                for ratio in h_ratios:
                    h = max(2, int(ratio * np.sqrt(n)))
                    h = min(h, n - 2)  # leave room for non-S vertices
                    for s in range(seeds_per):
                        seed = self.base_seed + hash((family, n, h, s)) % (2**31)
                        try:
                            inst = self.generate(family, n, h, seed=seed)
                            if inst.verify():
                                instances.append(inst)
                            else:
                                warnings.warn(f"Verification failed: {inst.instance_id}")
                        except Exception as e:
                            warnings.warn(f"Generation failed for {family}, n={n}, h={h}: {e}")
        return instances


# ============================================================
# SECTION 2: DIFFICULTY METRICS
# ============================================================

class DifficultyAnalyzer:
    """Compute multi-dimensional difficulty metrics for MIS instances."""

    @staticmethod
    def compute_metrics(inst: MISInstance) -> Dict:
        """Compute structural, spectral, and algorithmic difficulty metrics."""
        G = inst.graph
        n = inst.n
        metrics = {}

        # --- Structural metrics ---
        degrees = [d for _, d in G.degree()]
        metrics['n_vertices'] = n
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = 2 * G.number_of_edges() / max(n * (n - 1), 1)
        metrics['avg_degree'] = np.mean(degrees)
        metrics['max_degree'] = max(degrees) if degrees else 0
        metrics['min_degree'] = min(degrees) if degrees else 0
        metrics['degree_std'] = np.std(degrees)

        # Degeneracy (k-core number)
        metrics['degeneracy'] = max(nx.core_number(G).values()) if n > 0 else 0

        # Clustering coefficient
        metrics['avg_clustering'] = nx.average_clustering(G)

        # Planted set degree statistics
        S = inst.planted_set
        S_degrees = [G.degree(v) for v in S]
        R_degrees = [G.degree(v) for v in G.nodes() if v not in S]
        metrics['planted_avg_degree'] = np.mean(S_degrees) if S_degrees else 0
        metrics['non_planted_avg_degree'] = np.mean(R_degrees) if R_degrees else 0
        metrics['degree_gap'] = abs(metrics['planted_avg_degree'] - metrics['non_planted_avg_degree'])

        # --- Spectral metrics ---
        if n > 2 and G.number_of_edges() > 0:
            try:
                L = nx.laplacian_matrix(G).astype(float)
                k = min(5, n - 2)
                eigenvalues = eigsh(L, k=k, which='SM', return_eigenvectors=False)
                eigenvalues = np.sort(eigenvalues)
                # Filter out near-zero eigenvalues
                nonzero_eigs = eigenvalues[eigenvalues > 1e-10]
                metrics['algebraic_connectivity'] = float(nonzero_eigs[0]) if len(nonzero_eigs) > 0 else 0.0

                top_eigs = eigsh(L, k=min(3, n - 1), which='LM', return_eigenvectors=False)
                metrics['spectral_radius_laplacian'] = float(max(top_eigs))

                if metrics['algebraic_connectivity'] > 1e-10:
                    metrics['spectral_gap_ratio'] = (
                        metrics['spectral_radius_laplacian'] / metrics['algebraic_connectivity']
                    )
                else:
                    metrics['spectral_gap_ratio'] = float('inf')
            except Exception:
                metrics['algebraic_connectivity'] = None
                metrics['spectral_radius_laplacian'] = None
                metrics['spectral_gap_ratio'] = None
        else:
            metrics['algebraic_connectivity'] = 0.0
            metrics['spectral_radius_laplacian'] = 0.0
            metrics['spectral_gap_ratio'] = float('inf')

        # --- Independent set count estimate (for small graphs) ---
        if n <= 25:
            metrics['exact_alpha'] = len(max(nx.find_cliques(nx.complement(G)), key=len, default=[]))
        else:
            metrics['exact_alpha'] = None

        metrics['planted_set_size'] = inst.h
        metrics['h_over_sqrt_n'] = inst.h / np.sqrt(n)

        return metrics


# ============================================================
# SECTION 3: CLASSICAL SOLVERS
# ============================================================

@dataclass
class SolverResult:
    """Result from a classical MIS solver."""
    independent_set: Set[int]
    size: int
    runtime: float
    solver_name: str
    is_optimal: bool = False
    extra: Dict = field(default_factory=dict)


class MISSolver(ABC):
    """Abstract base class for MIS solvers."""

    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout

    @abstractmethod
    def solve(self, G: nx.Graph) -> SolverResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class GreedyMinDegreeSolver(MISSolver):
    """Greedy algorithm: iteratively select minimum-degree vertex."""

    @property
    def name(self) -> str:
        return "Greedy_MinDeg"

    def solve(self, G: nx.Graph) -> SolverResult:
        start = time.time()
        H = G.copy()
        independent_set = set()

        while H.number_of_nodes() > 0:
            # Select vertex with minimum degree
            v = min(H.nodes(), key=lambda x: H.degree(x))
            independent_set.add(v)
            # Remove v and its neighbors
            to_remove = set(H.neighbors(v)) | {v}
            H.remove_nodes_from(to_remove)

        runtime = time.time() - start
        return SolverResult(
            independent_set=independent_set,
            size=len(independent_set),
            runtime=runtime,
            solver_name=self.name
        )


class GreedyMaxDegreeSolver(MISSolver):
    """Greedy: remove max-degree vertex iteratively, then collect isolated."""

    @property
    def name(self) -> str:
        return "Greedy_MaxDegRemoval"

    def solve(self, G: nx.Graph) -> SolverResult:
        start = time.time()
        H = G.copy()

        while H.number_of_edges() > 0:
            v = max(H.nodes(), key=lambda x: H.degree(x))
            H.remove_node(v)

        independent_set = set(H.nodes())
        runtime = time.time() - start
        return SolverResult(
            independent_set=independent_set,
            size=len(independent_set),
            runtime=runtime,
            solver_name=self.name
        )


class LocalSearchSolver(MISSolver):
    """
    Iterated local search with (1,2)-swap neighborhood.
    Start from greedy solution, try swapping one vertex out for up to two in.
    """

    def __init__(self, timeout: float = 300.0, max_iter: int = 10000,
                 n_restarts: int = 5):
        super().__init__(timeout)
        self.max_iter = max_iter
        self.n_restarts = n_restarts

    @property
    def name(self) -> str:
        return "LocalSearch_12swap"

    def _is_independent(self, G: nx.Graph, S: Set[int]) -> bool:
        for u in S:
            for v in S:
                if u < v and G.has_edge(u, v):
                    return False
        return True

    def _greedy_init(self, G: nx.Graph, rng: np.random.RandomState) -> Set[int]:
        """Randomized greedy initialization."""
        H = G.copy()
        IS = set()
        nodes = list(H.nodes())
        rng.shuffle(nodes)

        for v in nodes:
            if v in H:
                neighbors_in_IS = set(H.neighbors(v)) & IS
                if not neighbors_in_IS:
                    IS.add(v)
        return IS

    def solve(self, G: nx.Graph) -> SolverResult:
        start = time.time()
        best_IS = set()
        rng = np.random.RandomState(42)

        for restart in range(self.n_restarts):
            if time.time() - start > self.timeout:
                break

            current_IS = self._greedy_init(G, rng)

            for iteration in range(self.max_iter):
                if time.time() - start > self.timeout:
                    break

                improved = False
                IS_list = list(current_IS)
                rng.shuffle(IS_list)

                for v_out in IS_list:
                    # Try removing v_out and adding neighbors' non-neighbors
                    candidate = current_IS - {v_out}

                    # Find vertices that could be added
                    blocked_by_v = set()
                    for u in G.neighbors(v_out):
                        if u not in current_IS:
                            # u is blocked only by vertices in IS that are its neighbors
                            blockers = set(G.neighbors(u)) & candidate
                            if not blockers:
                                blocked_by_v.add(u)

                    # Try adding up to 2 from blocked_by_v
                    added = set()
                    for u in blocked_by_v:
                        if not (set(G.neighbors(u)) & (candidate | added)):
                            added.add(u)
                            if len(added) >= 2:
                                break

                    if len(added) >= 2:
                        current_IS = candidate | added
                        improved = True
                        break

                if not improved:
                    break

            if len(current_IS) > len(best_IS):
                best_IS = current_IS.copy()

        runtime = time.time() - start
        return SolverResult(
            independent_set=best_IS,
            size=len(best_IS),
            runtime=runtime,
            solver_name=self.name
        )


class SimulatedAnnealingSolver(MISSolver):
    """Simulated annealing for MIS."""

    def __init__(self, timeout: float = 300.0, n_steps: int = 50000,
                 T_init: float = 2.0, T_final: float = 0.01):
        super().__init__(timeout)
        self.n_steps = n_steps
        self.T_init = T_init
        self.T_final = T_final

    @property
    def name(self) -> str:
        return "SimulatedAnnealing"

    def solve(self, G: nx.Graph) -> SolverResult:
        start = time.time()
        rng = np.random.RandomState(42)
        nodes = list(G.nodes())
        n = len(nodes)

        # Initialize with greedy
        solver = GreedyMinDegreeSolver()
        init_result = solver.solve(G)
        current_IS = init_result.independent_set.copy()
        best_IS = current_IS.copy()

        adj = {v: set(G.neighbors(v)) for v in nodes}

        alpha = (self.T_final / self.T_init) ** (1.0 / self.n_steps)
        T = self.T_init

        for step in range(self.n_steps):
            if time.time() - start > self.timeout:
                break

            T *= alpha
            v = nodes[rng.randint(n)]

            if v in current_IS:
                # Try removing v
                delta = -1
                if rng.random() < np.exp(delta / max(T, 1e-10)):
                    current_IS.remove(v)
            else:
                # Try adding v (must remove conflicting neighbors)
                conflicts = adj[v] & current_IS
                delta = 1 - len(conflicts)
                if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    current_IS -= conflicts
                    current_IS.add(v)

            if len(current_IS) > len(best_IS):
                best_IS = current_IS.copy()

        runtime = time.time() - start
        return SolverResult(
            independent_set=best_IS,
            size=len(best_IS),
            runtime=runtime,
            solver_name=self.name
        )


class SpectralPlantedSolver(MISSolver):
    """
    Spectral method for planted independent set detection.
    Uses the top eigenvector of the adjacency matrix.
    """

    @property
    def name(self) -> str:
        return "Spectral_Planted"

    def solve(self, G: nx.Graph) -> SolverResult:
        start = time.time()
        n = G.number_of_nodes()
        nodes = sorted(G.nodes())
        node_to_idx = {v: i for i, v in enumerate(nodes)}

        if n <= 2 or G.number_of_edges() == 0:
            return SolverResult(
                independent_set=set(nodes),
                size=n,
                runtime=time.time() - start,
                solver_name=self.name
            )

        # Build adjacency matrix
        A = nx.adjacency_matrix(G, nodelist=nodes).astype(float)

        try:
            # Top eigenvector of adjacency matrix
            _, vecs = eigsh(A, k=1, which='LA')
            v1 = vecs[:, 0]

            # For planted IS, the top eigenvector tends to have
            # smaller (more negative) values on planted vertices
            # since they have fewer connections
            # Threshold: take vertices with smallest eigenvector values
            threshold = np.median(v1)
            candidate = set()
            for i, node in enumerate(nodes):
                if v1[i] < threshold:
                    candidate.add(node)

            # Clean up: greedily remove conflicts
            IS = set()
            for v in sorted(candidate, key=lambda x: G.degree(x)):
                if not (set(G.neighbors(v)) & IS):
                    IS.add(v)

        except Exception:
            IS = GreedyMinDegreeSolver().solve(G).independent_set

        runtime = time.time() - start
        return SolverResult(
            independent_set=IS,
            size=len(IS),
            runtime=runtime,
            solver_name=self.name
        )


class NetworkXExactSolver(MISSolver):
    """
    Exact MIS via complement graph max clique (NetworkX).
    Only feasible for small graphs (n <= ~60).
    """

    @property
    def name(self) -> str:
        return "Exact_NX_Complement"

    def solve(self, G: nx.Graph) -> SolverResult:
        start = time.time()
        n = G.number_of_nodes()

        if n > 80:
            return SolverResult(
                independent_set=set(),
                size=0,
                runtime=time.time() - start,
                solver_name=self.name,
                extra={'status': 'skipped_too_large'}
            )

        try:
            complement = nx.complement(G)
            cliques = list(nx.find_cliques(complement))
            if cliques:
                max_clique = max(cliques, key=len)
                IS = set(max_clique)
            else:
                IS = set()
        except Exception as e:
            IS = set()

        runtime = time.time() - start
        return SolverResult(
            independent_set=IS,
            size=len(IS),
            runtime=runtime,
            solver_name=self.name,
            is_optimal=True
        )


class SATSolver(MISSolver):
    """
    MIS via incremental SAT solving (requires python-sat).
    Encodes as maximum satisfiability problem.
    """

    @property
    def name(self) -> str:
        return "SAT_MaxIS"

    def solve(self, G: nx.Graph) -> SolverResult:
        start = time.time()
        n = G.number_of_nodes()
        nodes = sorted(G.nodes())
        node_to_var = {v: i + 1 for i, v in enumerate(nodes)}

        try:
            from pysat.solvers import Glucose3
            from pysat.card import CardEnc, EncType
        except ImportError:
            return SolverResult(
                independent_set=set(), size=0,
                runtime=time.time() - start,
                solver_name=self.name,
                extra={'status': 'pysat_not_available'}
            )

        best_IS = set()

        # Binary search on IS size
        lo, hi = 1, n
        while lo <= hi:
            if time.time() - start > self.timeout:
                break

            mid = (lo + hi) // 2

            solver = Glucose3()

            # Edge constraints: for each edge (u,v), not both in IS
            for u, v in G.edges():
                solver.add_clause([-node_to_var[u], -node_to_var[v]])

            # Cardinality: at least mid vertices selected
            card_clauses = CardEnc.atleast(
                lits=list(node_to_var.values()),
                bound=mid,
                encoding=EncType.seqcounter
            )
            for clause in card_clauses:
                solver.add_clause(clause)

            if solver.solve():
                model = solver.get_model()
                IS = {nodes[abs(lit) - 1] for lit in model
                      if lit > 0 and abs(lit) <= n}
                if len(IS) >= len(best_IS):
                    best_IS = IS
                lo = mid + 1
            else:
                hi = mid - 1

            solver.delete()

        runtime = time.time() - start
        return SolverResult(
            independent_set=best_IS,
            size=len(best_IS),
            runtime=runtime,
            solver_name=self.name,
            is_optimal=(lo > hi)
        )


# ============================================================
# SECTION 4: ET COMPUTATION (for small instances)
# ============================================================

class ETComputer:
    """
    Compute the ET parameter for the planted MIS Macaulay system.
    Only feasible for small n (n <= 16 or so) due to 2^n dimension.
    """

    @staticmethod
    def compute_ET_toy(n: int) -> Dict:
        """
        Compute ET for the toy polynomial f = x1 + ... + xn - n.
        Returns exact ET and components.
        """
        # Use the closed-form solution
        ET = 2 * n**2  # contribution from empty row
        for i in range(1, n):
            j = n - i
            term = j**2 + j**2 / (n - j + 1)
            ET += term

        # Also compute p_k values
        from math import comb
        p_values = {k: 1.0 / comb(n, k) for k in range(n + 1)}

        return {
            'ET': ET,
            'ET_theoretical': n * (n - 1) * (2*n - 1) / 6 + 2 * n**2,  # approximate
            'p_values': p_values,
            'n': n,
            'z_norm_sq': n
        }

    @staticmethod
    def compute_ET_mis(G: nx.Graph, S: Set[int], h: int) -> Optional[Dict]:
        """
        Compute ET for the planted MIS Macaulay system.

        WARNING: This constructs the explicit Macaulay matrix.
        Only feasible for very small n (n <= 14).

        Args:
            G: the graph
            S: the planted independent set
            h: size of planted set

        Returns:
            dict with ET, p vector norms, etc., or None if too large
        """
        n = G.number_of_nodes()
        if n > 14:
            warnings.warn(f"n={n} too large for explicit ET computation (limit 14)")
            return None

        from math import comb
        from itertools import combinations as combs

        nodes = sorted(G.nodes())
        edges = set(G.edges())

        # Enumerate all multilinear monomials (subsets of [n])
        # Each nonempty subset T ⊆ [n] corresponds to a column
        # We only keep "valid" monomials (those not divisible by x_i x_j for (i,j) ∈ E)
        def is_independent_set(T):
            T_list = list(T)
            for i in range(len(T_list)):
                for j in range(i+1, len(T_list)):
                    if (T_list[i], T_list[j]) in edges or (T_list[j], T_list[i]) in edges:
                        return False
            return True

        # Count independent sets by size
        IS_counts = {}
        for size in range(1, n + 1):
            count = sum(1 for T in combs(nodes, size) if is_independent_set(set(T)))
            IS_counts[size] = count

        # The counting condition from the paper
        counting_satisfied = True
        for i in range(1, h + 1):
            bound = 10 * n * comb(h, i)  # poly(n) * C(h,i) with small constant
            if IS_counts.get(i, 0) > bound:
                counting_satisfied = False
                break

        # For the full ET computation, we would need to construct the
        # weighted Macaulay matrix AD and solve for p. This is 2^n x 2^n.
        # For n <= 14, 2^14 = 16384 which is feasible.

        # Simplified: use the layer-by-layer structure from the paper
        # ET ≈ Σ_{i=1}^{h} Σ_{t=0}^{i} I_{i,t} / C(h,i)  (approximate)

        # Count I_{i,t}: independent sets of size i with t vertices outside S
        S_set = set(S)
        I_by_layer = {}
        for size in range(1, h + 1):
            for t in range(size + 1):
                count = 0
                # Need exactly (size - t) from S and t from V\S
                for T_S in combs(sorted(S_set), size - t):
                    for T_R in combs(sorted(set(nodes) - S_set), t):
                        T = set(T_S) | set(T_R)
                        if is_independent_set(T):
                            count += 1
                I_by_layer[(size, t)] = count

        # Estimate ET using the approximate formula
        ET_estimate = 1.0  # p_0^2 * d_0 contribution
        for i in range(1, h + 1):
            for t in range(i + 1):
                I_it = I_by_layer.get((i, t), 0)
                # p_{m_i,t}^2 * d_{m_i} ≈ 1/C(h,i) (approximate)
                if comb(h, i) > 0:
                    ET_estimate += I_it / comb(h, i)

        return {
            'ET_estimate': ET_estimate,
            'IS_counts': IS_counts,
            'I_by_layer': {str(k): v for k, v in I_by_layer.items()},
            'counting_condition_satisfied': counting_satisfied,
            'n': n,
            'h': h,
            'h_over_sqrt_n': h / np.sqrt(n),
        }


# ============================================================
# SECTION 5: BENCHMARKING PIPELINE
# ============================================================

class BenchmarkRunner:
    """Run all solvers on all instances and collect results."""

    def __init__(self, solvers: List[MISSolver], compute_difficulty: bool = True):
        self.solvers = solvers
        self.compute_difficulty = compute_difficulty

    def run(self, instances: List[MISInstance], verbose: bool = True) -> List[Dict]:
        """
        Run benchmark suite.

        Returns:
            List of result dictionaries, one per (instance, solver) pair.
        """
        results = []
        total = len(instances) * len(self.solvers)
        count = 0

        for inst in instances:
            # Compute difficulty metrics once per instance
            if self.compute_difficulty:
                try:
                    difficulty = DifficultyAnalyzer.compute_metrics(inst)
                except Exception as e:
                    difficulty = {'error': str(e)}
            else:
                difficulty = {}

            for solver in self.solvers:
                count += 1
                if verbose:
                    print(f"  [{count}/{total}] {solver.name} on {inst.instance_id}...",
                          end='', flush=True)

                try:
                    result = solver.solve(inst.graph)

                    # Compute overlap with planted set
                    overlap = len(result.independent_set & inst.planted_set)
                    recall = overlap / max(inst.h, 1)
                    precision = overlap / max(result.size, 1)

                    entry = {
                        'instance_id': inst.instance_id,
                        'family': inst.family,
                        'n': inst.n,
                        'h': inst.h,
                        'h_over_sqrt_n': inst.h / np.sqrt(inst.n),
                        'seed': inst.seed,
                        'solver': solver.name,
                        'solution_size': result.size,
                        'runtime': result.runtime,
                        'is_optimal': result.is_optimal,
                        'planted_overlap': overlap,
                        'planted_recall': recall,
                        'planted_precision': precision,
                        **{f'diff_{k}': v for k, v in difficulty.items()
                           if isinstance(v, (int, float)) and v is not None},
                    }
                    results.append(entry)

                    if verbose:
                        print(f" size={result.size}, recall={recall:.2f}, "
                              f"time={result.runtime:.3f}s")

                except Exception as e:
                    if verbose:
                        print(f" ERROR: {e}")
                    results.append({
                        'instance_id': inst.instance_id,
                        'family': inst.family,
                        'n': inst.n,
                        'h': inst.h,
                        'solver': solver.name,
                        'error': str(e),
                    })

        return results

    @staticmethod
    def summarize(results: List[Dict]) -> str:
        """Generate text summary of results."""
        lines = []
        lines.append("=" * 70)
        lines.append("BENCHMARK SUMMARY")
        lines.append("=" * 70)

        # Group by family and solver
        from collections import defaultdict
        by_family_solver = defaultdict(list)
        for r in results:
            if 'error' not in r:
                key = (r['family'], r['solver'])
                by_family_solver[key].append(r)

        families = sorted(set(r.get('family', '?') for r in results))
        solvers = sorted(set(r.get('solver', '?') for r in results))

        for family in families:
            lines.append(f"\n--- Family: {family} ---")
            for solver in solvers:
                key = (family, solver)
                data = by_family_solver.get(key, [])
                if not data:
                    continue
                sizes = [d['solution_size'] for d in data]
                recalls = [d['planted_recall'] for d in data]
                times = [d['runtime'] for d in data]
                lines.append(
                    f"  {solver:30s}: "
                    f"avg_size={np.mean(sizes):.1f}, "
                    f"avg_recall={np.mean(recalls):.3f}, "
                    f"avg_time={np.mean(times):.3f}s "
                    f"({len(data)} instances)"
                )

        return '\n'.join(lines)


# ============================================================
# SECTION 6: MAIN ENTRY POINT
# ============================================================

def run_full_benchmark(
    sizes: List[int] = None,
    h_ratios: List[float] = None,
    families: List[str] = None,
    seeds_per: int = 3,
    output_dir: str = "benchmark_results"
):
    """
    Run the complete classical benchmarking suite.

    This is the main entry point for the benchmarking campaign.
    """
    if sizes is None:
        sizes = [30, 50, 80, 120]
    if h_ratios is None:
        h_ratios = [0.5, 1.0, 1.5, 2.0]  # multiples of sqrt(n)
    if families is None:
        families = [
            'erdos_renyi_planted',
            'multi_clique_core',
            'camouflaged_clique',
            'geometric_planted',
        ]

    print("=" * 70)
    print("MIS CLASSICAL BENCHMARKING SUITE")
    print("=" * 70)

    # Generate instances
    print("\n[1/4] Generating instances...")
    gen = InstanceGenerator(seed=2025)
    instances = gen.generate_suite(families, sizes, h_ratios, seeds_per)
    print(f"  Generated {len(instances)} instances")

    # Verify all instances
    verified = sum(1 for inst in instances if inst.verify())
    print(f"  Verified: {verified}/{len(instances)}")

    # Initialize solvers
    print("\n[2/4] Initializing solvers...")
    solvers = [
        GreedyMinDegreeSolver(),
        GreedyMaxDegreeSolver(),
        LocalSearchSolver(timeout=30, max_iter=5000, n_restarts=3),
        SimulatedAnnealingSolver(timeout=30, n_steps=20000),
        SpectralPlantedSolver(),
        NetworkXExactSolver(),
    ]

    # Try to add SAT solver
    try:
        import pysat
        solvers.append(SATSolver(timeout=60))
        print(f"  Loaded {len(solvers)} solvers (including SAT)")
    except ImportError:
        print(f"  Loaded {len(solvers)} solvers (pysat not available)")

    # Run benchmarks
    print("\n[3/4] Running benchmarks...")
    runner = BenchmarkRunner(solvers, compute_difficulty=True)
    results = runner.run(instances, verbose=True)

    # Summarize
    print("\n[4/4] Analysis...")
    summary = BenchmarkRunner.summarize(results)
    print(summary)

    # Compute ET for small instances
    print("\n--- ET Computation (small instances) ---")
    et_results = []
    for inst in instances:
        if inst.n <= 14:
            et = ETComputer.compute_ET_mis(inst.graph, inst.planted_set, inst.h)
            if et is not None:
                et_results.append({
                    'instance_id': inst.instance_id,
                    'family': inst.family,
                    **et
                })
                print(f"  {inst.instance_id}: ET≈{et['ET_estimate']:.1f}, "
                      f"counting_ok={et['counting_condition_satisfied']}")

    # Toy ET validation
    print("\n--- Toy ET Validation ---")
    for n_toy in [5, 8, 10, 12, 15, 20]:
        toy = ETComputer.compute_ET_toy(n_toy)
        print(f"  n={n_toy:3d}: ET={toy['ET']:.1f}, "
              f"n^3/3={n_toy**3/3:.1f}, "
              f"ratio={toy['ET']/(n_toy**3/3):.3f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Filter results for JSON serialization
    json_results = []
    for r in results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool)):
                jr[k] = v
            elif isinstance(v, np.floating):
                jr[k] = float(v)
            elif isinstance(v, np.integer):
                jr[k] = int(v)
        json_results.append(jr)

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(summary)

    print(f"\nResults saved to {output_dir}/")
    return results, et_results


# ============================================================
# Run if executed directly
# ============================================================
if __name__ == "__main__":
    results, et_results = run_full_benchmark(
        sizes=[20, 30, 50],
        h_ratios=[0.5, 1.0, 2.0],
        seeds_per=2
    )