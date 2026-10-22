"""EM-PSO waypoint fitness.

    F_i(x) = w_cov * G_cov(x) + w_det * G_det(x) - w_tr * ||x - x_i|| / h - P_spring(x)

* ``G_cov``: normalised expected detections of a look at x (search map).
* ``G_det``: attraction to tentative targets awaiting confirmation.
* ``P_spring`` variants:
    - ``none``         : standard PSO search (baseline B0)
    - ``distance_nn``  : SENTINEL v1, lam * max(0, d_nn - R_max)^2 (baseline B1)
    - ``outage``       : lam_i * sum_{j in S_i} ((P_out(x, x_j) - eps_i)_+ / eps_i)^2
                         with information-aware tension (contribution C1)
                         lam_i = lam0 (1 + beta U_i), eps_i = eps0 / (1 + beta U_i)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from .channel import ChannelProfile, outage_table

ObsFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SpringConfig:
    mode: str = 'outage'            # none | distance_nn | outage
    lam0: float = 1.0
    eps0: float = 0.05
    beta: float = 0.5               # information-aware gain (0 disables C1)
    r_max: float = 50.0             # distance_nn only


@dataclass
class FitnessContext:
    x_i: np.ndarray
    horizon: float
    gain: Callable[[np.ndarray], np.ndarray]           # P x 2 -> P, already normalised
    neighbours: List[np.ndarray] = field(default_factory=list)  # spring set S_i positions
    tentative: List[tuple] = field(default_factory=list)        # [(pos, conf)]
    unreplicated: int = 0
    profile: Optional[ChannelProfile] = None
    obstructions: Optional[ObsFn] = None
    w_cov: float = 1.0
    w_det: float = 1.0
    w_tr: float = 0.05
    det_sigma: float = 8.0


def spring_penalty(pts: np.ndarray, ctx: FitnessContext, cfg: SpringConfig) -> np.ndarray:
    pts = np.atleast_2d(pts)
    out = np.zeros(len(pts))
    if cfg.mode == 'none' or not ctx.neighbours:
        return out
    if cfg.mode == 'distance_nn':
        d = np.min([np.linalg.norm(pts - n, axis=1) for n in ctx.neighbours], axis=0)
        return cfg.lam0 * np.maximum(0.0, d - cfg.r_max) ** 2
    if cfg.mode != 'outage':
        raise ValueError(cfg.mode)
    scale = 1.0 + cfg.beta * ctx.unreplicated
    lam, eps = cfg.lam0 * scale, cfg.eps0 / scale
    for n in ctx.neighbours:
        d = np.linalg.norm(pts - n, axis=1)
        obs = (ctx.obstructions(pts, n) if ctx.obstructions is not None
               else np.zeros(len(pts), dtype=int))
        p = outage_table(ctx.profile)(d, obs)
        out += lam * (np.maximum(0.0, p - eps) / eps) ** 2
    return out


def fitness(pts: np.ndarray, ctx: FitnessContext, cfg: SpringConfig) -> np.ndarray:
    pts = np.atleast_2d(pts)
    f = ctx.w_cov * ctx.gain(pts)
    if ctx.tentative:
        s2 = 2.0 * ctx.det_sigma ** 2
        for pos, conf in ctx.tentative:
            f += ctx.w_det * conf * np.exp(-np.sum((pts - pos) ** 2, axis=1) / s2)
    f -= ctx.w_tr * np.linalg.norm(pts - ctx.x_i, axis=1) / max(ctx.horizon, 1e-6)
    return f - spring_penalty(pts, ctx, cfg)
