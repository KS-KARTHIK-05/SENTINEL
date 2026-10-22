"""Monte-Carlo verification of Proposition 1 (connectivity preservation).

Random connected swarms take adversarial desired steps; every agent projects
its step onto the midpoint balls of its LMST-selected neighbours. The check is
that the communication graph stays connected after every step.
Usage: python verify_prop1.py [trials] [seed]
"""
import sys

import numpy as np

from sentinel_core import graph
from sentinel_core.safety import midpoint_balls, project_step


def trial(rng, R=48.0, steps=25):
    n = int(rng.integers(2, 51))
    while True:
        pos = rng.uniform(0, 25.0 * np.sqrt(n), (n, 2))
        d = graph.pairwise_dist(pos)
        a = d <= R
        np.fill_diagonal(a, False)
        if graph.is_connected(a):
            break
    for _ in range(steps):
        d = graph.pairwise_dist(pos)
        adj = d <= R
        np.fill_diagonal(adj, False)
        sel = {i: graph.lmst_selected(i, adj, d) for i in range(n)}
        desired = rng.normal(0, rng.uniform(1, 40), (n, 2))
        new = pos.copy()
        for i in range(n):
            new[i] = pos[i] + project_step(pos[i], desired[i],
                                           midpoint_balls(pos[i], [pos[j] for j in sel[i]], R))
        d2 = graph.pairwise_dist(new)
        a2 = d2 <= R
        np.fill_diagonal(a2, False)
        if not graph.is_connected(a2):
            return False, n
        pos = new
    return True, n


if __name__ == '__main__':
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    fails = [n for ok, n in (trial(rng) for _ in range(trials)) if not ok]
    print(f'trials={trials} steps/trial=25 failures={len(fails)} sizes={fails[:10]}')
    sys.exit(1 if fails else 0)
