"""B7: hard distance wall (the first version's 50 m radius as a hard constraint).

Standard constriction PSO search (B0) in which a candidate waypoint is infeasible if it lies
farther than ``r_wall`` from the UAV's nearest neighbour (last beacon positions of all fresh
neighbours, the responder included). Infeasible candidates are rejected; if every candidate is
infeasible, the UAV holds its position. The point toward the nearest neighbour (clipped to the
search radius) is added to the particle seeds, so a UAV that is beyond the wall can find a
feasible candidate if one is reachable. With no fresh neighbour the constraint is vacuous.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..agent import SentinelAgent


def nearest_distance(pts: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Distance from each point in ``pts`` (P x 2) to the nearest of ``others`` (K x 2)."""
    pts = np.atleast_2d(pts)
    return np.min(np.linalg.norm(pts[:, None, :] - others[None, :, :], axis=-1), axis=1)


class HardWallAgent(SentinelAgent):
    r_wall: float = 50.0

    def _others(self, nbrs: Dict) -> np.ndarray:
        return np.stack([s.pos for s in nbrs.values()])

    def _search_hook(self, t, f, seeds, nbrs):
        if not nbrs:
            return f, seeds
        others, r = self._others(nbrs), self.r_wall

        def walled(p, f0=f):
            p = np.atleast_2d(p)
            return np.where(nearest_distance(p, others) <= r, f0(p), -np.inf)

        q = others[int(np.argmin(np.linalg.norm(others - self.pos, axis=1)))]
        d0 = float(np.linalg.norm(q - self.pos))
        horizon = self.cfg.v_max * self.cfg.horizon_s
        if d0 > 1e-9:
            seeds = list(seeds) + [self.pos + (q - self.pos) / d0 * min(horizon, d0)]
        return walled, seeds

    def _plan(self, t, nbrs):
        super()._plan(t, nbrs)
        if nbrs and nearest_distance(self.waypoint, self._others(nbrs))[0] > self.r_wall:
            self.waypoint = self.pos.copy()          # every candidate infeasible: hold
