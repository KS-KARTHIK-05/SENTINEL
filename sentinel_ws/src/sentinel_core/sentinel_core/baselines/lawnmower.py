"""B6: boustrophedon (lawnmower) sweep of static vertical strips, one per UAV.

Classical SAR coverage practice; no communication awareness and no
reassignment of the strips of failed UAVs.
"""
from __future__ import annotations

import numpy as np

from ..agent import SentinelAgent


class LawnmowerAgent(SentinelAgent):
    n_strips: int = 1
    strip: int = 0
    lane_factor: float = 1.8      # lane spacing / footprint radius

    def _route(self):
        lo, hi = self.bounds
        w = (hi[0] - lo[0]) / self.n_strips
        x0, x1 = lo[0] + self.strip * w, lo[0] + (self.strip + 1) * w
        lane = self.lane_factor * self.cfg.footprint_r
        xs = np.arange(x0 + lane / 2, x1, lane)
        if len(xs) == 0:
            xs = np.array([(x0 + x1) / 2])
        y_lo, y_hi = lo[1] + lane / 2, hi[1] - lane / 2
        pts = []
        for k, x in enumerate(xs):
            ys = (y_lo, y_hi) if k % 2 == 0 else (y_hi, y_lo)
            pts += [np.array([x, ys[0]]), np.array([x, ys[1]])]
        if self.no_fly is not None and pts:
            keep = ~self.no_fly(np.stack(pts))
            pts = [p for p, k in zip(pts, keep) if k]
        return pts

    def _desired_step(self, t, dt, nbrs):
        if not hasattr(self, '_wps'):
            self._wps, self._k = self._route(), 0
            self._best_d, self._t_progress = np.inf, t
        if self._k >= len(self._wps):
            self._wps, self._k = self._wps[::-1], 0      # repeat sweep
        wp = self._wps[self._k]
        d = np.linalg.norm(wp - self.pos)
        if d < self._best_d - 0.5:
            self._best_d, self._t_progress = d, t
        if d < 1.0 or t - self._t_progress > 15.0:      # reached, or blocked: next point
            self._k += 1
            self._best_d, self._t_progress = np.inf, t
        self.waypoint = wp
        d = wp - self.pos
        n = np.linalg.norm(d)
        return d if n <= self.cfg.v_max * dt else d / n * self.cfg.v_max * dt
