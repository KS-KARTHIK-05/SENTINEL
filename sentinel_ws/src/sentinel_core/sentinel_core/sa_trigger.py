"""Collaborative synthetic-aperture (SA) flanking.

A tentative entry (0.40 <= fused confidence < 0.70) requests a second view
from a lateral offset of ``offset_m`` perpendicular to the observer-target
direction (the perpendicular construction of SENTINEL v1). The wingman is the
nearest available neighbour whose flank point is admissible, i.e. reachable
without violating its connectivity constraints (``admissible`` callback).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


def flank_points(observer_xy, target_xy, offset_m: float = 15.0):
    """Two candidate view points at +-offset perpendicular to observer->target."""
    o = np.asarray(observer_xy[:2], float)
    t = np.asarray(target_xy[:2], float)
    d = t - o
    n = np.linalg.norm(d)
    if n < 1e-6:            # observer overhead: any horizontal axis works
        d, n = np.array([1.0, 0.0]), 1.0
    u = d / n
    perp = np.array([-u[1], u[0]])
    return t + perp * offset_m, t - perp * offset_m


def plan_flank(observer: int, observer_xy, target_xy, candidates: Dict[int, np.ndarray],
               admissible: Callable[[int, np.ndarray], bool],
               offset_m: float = 15.0) -> Optional[Tuple[int, np.ndarray]]:
    """Choose (wingman, flank point) or None if no admissible assignment exists."""
    pts = flank_points(observer_xy, target_xy, offset_m)
    order = sorted((j for j in candidates if j != observer),
                   key=lambda j: np.linalg.norm(candidates[j][:2] - np.asarray(target_xy[:2])))
    for j in order:
        for p in sorted(pts, key=lambda q: np.linalg.norm(q - candidates[j][:2])):
            if admissible(j, p):
                return j, p
    return None
