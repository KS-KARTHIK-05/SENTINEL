import numpy as np
import pytest

from sentinel_core import graph
from sentinel_core.safety import midpoint_balls, project_step


def _connected_positions(rng, n, R, side):
    while True:
        pos = rng.uniform(0, side, (n, 2))
        d = graph.pairwise_dist(pos)
        a = d <= R
        np.fill_diagonal(a, False)
        if graph.is_connected(a):
            return pos


@pytest.mark.parametrize('seed', range(300))
def test_projected_motion_preserves_connectivity(seed):
    """Proposition 1: LMST-constrained projected steps keep G(t) connected."""
    rng = np.random.default_rng(seed)
    n, R = int(rng.integers(2, 25)), 48.0
    pos = _connected_positions(rng, n, R, 30.0 * np.sqrt(n))
    for _ in range(20):
        d = graph.pairwise_dist(pos)
        adj = d <= R
        np.fill_diagonal(adj, False)
        sel = {i: graph.lmst_selected(i, adj, d) for i in range(n)}
        steps = rng.normal(0, 15.0, (n, 2))        # adversarial-ish desired motion
        new = pos.copy()
        for i in range(n):
            balls = midpoint_balls(pos[i], [pos[j] for j in sel[i]], R)
            new[i] = pos[i] + project_step(pos[i], steps[i], balls)
        for i in range(n):                          # every preserved edge survives
            for j in sel[i]:
                if i in sel[j]:
                    assert np.linalg.norm(new[i] - new[j]) <= R + 1e-6
        d2 = graph.pairwise_dist(new)
        a2 = d2 <= R + 1e-6
        np.fill_diagonal(a2, False)
        assert graph.is_connected(a2)
        pos = new
