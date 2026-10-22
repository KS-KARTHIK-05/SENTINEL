"""Decentralised connectivity-preserving step projection.

For every preserved neighbour j of UAV i with current distance
``d_ij <= R``, both UAVs restrict their next position to the ball of radius
``R/2`` centred at the common midpoint ``m_ij = (x_i + x_j) / 2``. Because the
two balls share the same centre, the triangle inequality gives
``||x_i' - x_j'|| <= R`` regardless of how the neighbour moves, so the link
survives the step without any synchronisation (Proposition 1). The current
position always lies in the ball (``||x_i - m_ij|| = d_ij / 2``), so scaling
the proposed displacement back along its direction is always feasible.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

Ball = Tuple[np.ndarray, float]


def midpoint_balls(x_i: np.ndarray, neighbours: Iterable[np.ndarray], radius: float,
                   margin: float = 0.0) -> list:
    """Constraint balls for preserving links of length at most ``radius``."""
    # relative slack keeps preserved links strictly below ``radius`` despite
    # floating-point error in the step scaling
    r = 0.5 * radius * (1.0 - 1e-7) - margin
    return [(0.5 * (np.asarray(x_i, float) + np.asarray(x_j, float)), r) for x_j in neighbours]


def max_feasible_fraction(x: np.ndarray, delta: np.ndarray, balls: Sequence[Ball]) -> float:
    """Largest alpha in [0, 1] with x + alpha * delta inside every ball."""
    x = np.asarray(x, float)
    delta = np.asarray(delta, float)
    dd = float(delta @ delta)
    if dd == 0.0:
        return 1.0
    alpha = 1.0
    for c, r in balls:
        a = x - c
        aa = float(a @ a)
        ad = float(a @ delta)
        if aa > r * r:
            # Already outside (stale data or a topology change): only allow
            # displacements that do not increase the distance to the centre.
            if ad >= 0.0:
                return 0.0
            alpha = min(alpha, -2.0 * ad / dd)
            continue
        disc = ad * ad - dd * (aa - r * r)
        root = (-ad + np.sqrt(max(disc, 0.0))) / dd
        alpha = min(alpha, max(root, 0.0))
    return alpha


def project_step(x: np.ndarray, delta: np.ndarray, balls: Sequence[Ball]) -> np.ndarray:
    """Feasible displacement: ``delta`` scaled back to stay inside all balls."""
    return max_feasible_fraction(x, delta, balls) * np.asarray(delta, float)


def recovery_step(x: np.ndarray, balls: Sequence[Ball], v_max_dt: float) -> np.ndarray:
    """Displacement toward the most violated ball centre (re-attachment)."""
    x = np.asarray(x, float)
    worst, worst_v = None, 0.0
    for c, r in balls:
        v = np.linalg.norm(x - c) - r
        if v > worst_v:
            worst, worst_v = c, v
    if worst is None:
        return np.zeros_like(x)
    d = worst - x
    n = np.linalg.norm(d)
    return d / n * min(v_max_dt, worst_v + 1e-6) if n > 0 else np.zeros_like(x)
