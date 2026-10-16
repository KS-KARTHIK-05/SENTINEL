"""Per-agent constriction PSO over the reachable waypoint disk.

Each UAV optimises its next waypoint with a Clerc-Kennedy constriction PSO
(chi = 0.7298, c1 = c2 = 2.05), which keeps particle trajectories bounded.
Particles are seeded with the previous best waypoint (temporal memory) and
with the best waypoints broadcast by mesh neighbours (social information over
the communication graph), so the swarm-level search is a local-best PSO whose
neighbourhood is the communication topology. Cost per agent per planning step
is O(P * I * (k + 1)) for P particles, I iterations and k spring neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PSOConfig:
    particles: int = 24
    iterations: int = 12
    chi: float = 0.7298
    c1: float = 2.05
    c2: float = 2.05


def _clip_disk(p: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    v = p - center
    n = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.where(n > radius, radius / np.maximum(n, 1e-12), 1.0)
    return center + v * scale


def optimise(f: Callable[[np.ndarray], np.ndarray], center: np.ndarray, radius: float,
             bounds: Tuple[np.ndarray, np.ndarray], rng: np.random.Generator,
             cfg: PSOConfig = PSOConfig(),
             seeds: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, float]:
    """Maximise ``f`` over the disk(center, radius) intersected with ``bounds``."""
    lo, hi = bounds
    center = np.asarray(center, float)

    def feasible(p):
        return np.clip(_clip_disk(p, center, radius), lo, hi)

    r = radius * np.sqrt(rng.random(cfg.particles))
    th = 2 * np.pi * rng.random(cfg.particles)
    x = center + np.stack([r * np.cos(th), r * np.sin(th)], 1)
    x[0] = center                           # hovering is always a candidate
    if seeds:
        s = np.asarray(list(seeds), float)[: cfg.particles - 1]
        x[1:1 + len(s)] = s
    x = feasible(x)
    v = np.zeros_like(x)
    fx = f(x)
    pbest, pf = x.copy(), fx.copy()
    g = int(np.argmax(pf))
    for _ in range(cfg.iterations):
        r1, r2 = rng.random((2,) + x.shape)
        v = cfg.chi * (v + cfg.c1 * r1 * (pbest - x) + cfg.c2 * r2 * (pbest[g] - x))
        v = np.clip(v, -radius, radius)
        x = feasible(x + v)
        fx = f(x)
        better = fx > pf
        pbest[better], pf[better] = x[better], fx[better]
        g = int(np.argmax(pf))
    return pbest[g].copy(), float(pf[g])
