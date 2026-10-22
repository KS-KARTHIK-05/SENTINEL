import numpy as np
import pytest

from sentinel_core import graph


def _path(n):
    a = np.zeros((n, n), bool)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = True
    return a


def test_lambda2_known_graphs():
    n = 7
    assert graph.algebraic_connectivity(_path(n)) == pytest.approx(2 * (1 - np.cos(np.pi / n)))
    star = np.zeros((n, n), bool)
    star[0, 1:] = star[1:, 0] = True
    assert graph.algebraic_connectivity(star) == pytest.approx(1.0)
    full = ~np.eye(n, dtype=bool)
    assert graph.algebraic_connectivity(full) == pytest.approx(n)
    two = np.zeros((4, 4), bool)
    two[0, 1] = two[1, 0] = two[2, 3] = two[3, 2] = True
    assert graph.algebraic_connectivity(two) == pytest.approx(0.0, abs=1e-12)


def _mst_edges(adj, dist):
    return graph._prim(list(range(len(adj))), adj, dist)


@pytest.mark.parametrize('seed', range(40))
def test_lmst_contains_global_mst_and_preserves_connectivity(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 30))
    pos = rng.uniform(0, 150, (n, 2))
    dist = graph.pairwise_dist(pos)
    adj = dist <= 50.0
    np.fill_diagonal(adj, False)
    topo = graph.topology_adjacency(graph.lmst_topology(adj, dist), n)
    assert len(graph.components(topo)) == len(graph.components(adj))
    for u, v in _mst_edges(adj, dist):
        assert topo[u, v]
    assert np.all(adj[topo])          # only uses existing links


def test_nearest_neighbour_rule_does_not_imply_connectivity():
    # Lemma 1: every UAV within R of its nearest neighbour, graph disconnected.
    pos = np.array([[0, 0], [10, 0], [200, 0], [210, 0]], float)
    dist = graph.pairwise_dist(pos)
    nn_ok = all(dist[i, graph.nearest_neighbor(i, dist, range(4))] <= 50 for i in range(4))
    adj = dist <= 50
    np.fill_diagonal(adj, False)
    assert nn_ok and not graph.is_connected(adj)


@pytest.mark.parametrize('seed', range(30))
def test_scipy_lmst_matches_reference_prim(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(3, 40))
    pos = rng.uniform(0, 150, (n, 2))
    dist = graph.pairwise_dist(pos)
    adj = dist <= 50.0
    np.fill_diagonal(adj, False)
    for i in range(n):
        local = [i] + [int(v) for v in np.flatnonzero(adj[i])]
        ref = {v for u, v in graph._prim(local, adj, dist) if u == i} | \
              {u for u, v in graph._prim(local, adj, dist) if v == i}
        assert graph.lmst_selected(i, adj, dist) == ref
