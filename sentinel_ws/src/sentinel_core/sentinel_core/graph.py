"""Communication-graph utilities: Laplacian, algebraic connectivity and LMST.

The local minimum spanning tree (LMST) of Li, Hou and Sha (INFOCOM 2003) is
used to select, for every UAV, the subset of neighbours whose links must be
preserved. Each UAV builds an MST over its one-hop neighbourhood using only
locally available information; the symmetric LMST topology contains the
global MST of the communication graph (for unique edge weights), hence it
spans every connected component (Lemma 2 in the paper).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

import numpy as np


def pairwise_dist(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def laplacian(adj: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(adj, dtype=float)
    if weights is not None:
        a = a * weights
    np.fill_diagonal(a, 0.0)
    return np.diag(a.sum(1)) - a


def algebraic_connectivity(adj: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Second-smallest Laplacian eigenvalue (Fiedler value), 0 if N < 2."""
    n = adj.shape[0]
    if n < 2:
        return 0.0
    ev = np.linalg.eigvalsh(laplacian(adj, weights))
    return float(max(ev[1], 0.0))


def components(adj: np.ndarray, alive: Iterable[int] | None = None) -> List[List[int]]:
    """Connected components restricted to the ``alive`` vertex set."""
    n = adj.shape[0]
    nodes = set(range(n)) if alive is None else set(alive)
    seen: Set[int] = set()
    comps = []
    for s in sorted(nodes):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.flatnonzero(adj[u]):
                v = int(v)
                if v in nodes and v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def is_connected(adj: np.ndarray, alive: Iterable[int] | None = None) -> bool:
    return len(components(adj, alive)) <= 1


def _edge_key(dist: np.ndarray, u: int, v: int) -> Tuple[float, int, int]:
    """Unique edge weight: distance, ties broken by node ids (LMST paper)."""
    return (float(dist[u, v]), max(u, v), min(u, v))


def _prim(nodes: List[int], adj: np.ndarray, dist: np.ndarray) -> Set[Tuple[int, int]]:
    """MST (forest) edges over the subgraph induced by ``nodes``."""
    remaining = set(nodes)
    edges: Set[Tuple[int, int]] = set()
    while remaining:
        root = min(remaining)
        remaining.discard(root)
        in_tree = {root}
        while True:
            best = None
            for u in in_tree:
                for v in remaining:
                    if adj[u, v]:
                        k = _edge_key(dist, u, v)
                        if best is None or k < best[0]:
                            best = (k, u, v)
            if best is None:
                break
            _, u, v = best
            edges.add((min(u, v), max(u, v)))
            in_tree.add(v)
            remaining.discard(v)
    return edges


def _mst_edges_scipy(nodes: List[int], adj: np.ndarray, dist: np.ndarray):
    """MST (forest) of the induced subgraph with the LMST tie-breaking order.

    Weights are distance plus a tiny id-dependent offset, which reproduces the
    lexicographic order (d, max id, min id) of ``_edge_key`` for distinct
    distances and makes every weight unique.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    idx = np.asarray(nodes)
    a = adj[np.ix_(idx, idx)]
    d = dist[np.ix_(idx, idx)]
    n_tot = adj.shape[0] + 1
    hi = np.maximum(idx[:, None], idx[None, :]).astype(float)
    lo = np.minimum(idx[:, None], idx[None, :]).astype(float)
    w = np.where(a, d + 1e-7 * (hi * n_tot + lo) / (n_tot * n_tot) + 1e-12, 0.0)
    np.fill_diagonal(w, 0.0)
    t = minimum_spanning_tree(w).tocoo()
    return {(min(idx[r], idx[c]), max(idx[r], idx[c])) for r, c in zip(t.row, t.col)}


def lmst_selected(i: int, adj: np.ndarray, dist: np.ndarray,
                  alive: Iterable[int] | None = None) -> Set[int]:
    """Neighbours that UAV ``i`` selects in its local MST (1-hop information)."""
    alive_set = set(range(adj.shape[0])) if alive is None else set(alive)
    local = [i] + [int(v) for v in np.flatnonzero(adj[i]) if int(v) in alive_set and v != i]
    if len(local) == 1:
        return set()
    sel = set()
    for u, v in _mst_edges_scipy(local, adj, dist):
        if u == i:
            sel.add(int(v))
        elif v == i:
            sel.add(int(u))
    return sel


def lmst_topology(adj: np.ndarray, dist: np.ndarray,
                  alive: Iterable[int] | None = None,
                  symmetric: bool = True) -> Dict[int, Set[int]]:
    """LMST neighbour sets for all alive nodes.

    ``symmetric=True`` keeps an edge only if both endpoints select it (G0^-),
    the sparser variant that still preserves connectivity.
    """
    alive_list = sorted(range(adj.shape[0]) if alive is None else alive)
    sel = {i: lmst_selected(i, adj, dist, alive_list) for i in alive_list}
    if not symmetric:
        out = {i: set(s) for i, s in sel.items()}
        for i, s in sel.items():
            for j in s:
                out[j].add(i)
        return out
    return {i: {j for j in s if i in sel[j]} for i, s in sel.items()}


def topology_adjacency(nbrs: Dict[int, Set[int]], n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=bool)
    for i, s in nbrs.items():
        for j in s:
            a[i, j] = a[j, i] = True
    return a


def nearest_neighbor(i: int, dist: np.ndarray, alive: Iterable[int]) -> int | None:
    cands = [j for j in alive if j != i]
    return min(cands, key=lambda j: dist[i, j]) if cands else None
