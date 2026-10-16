"""B4: edge-tension potential with hysteresis (Ji & Egerstedt, IEEE T-RO 2007).

An edge (i, j) enters the maintained set when d_ij <= R - hyst and is then
kept; its tension V_ij = d^2 / (R - d) yields the control
-grad V = -(2R - d)(x_i - x_j) / (R - d)^2, added to the B0 task velocity.
"""
from __future__ import annotations

from typing import Set

import numpy as np

from ..agent import SentinelAgent


class PotentialAgent(SentinelAgent):
    gain: float = 2.0
    hyst: float = 5.0

    def _desired_step(self, t, dt, nbrs):
        task = super()._desired_step(t, dt, nbrs)
        if not hasattr(self, '_edges'):
            self._edges: Set[int] = set()
        R = self.r_hard
        u = np.zeros(2)
        for j in list(self._edges):
            if j not in nbrs:
                self._edges.discard(j)
        for j, s in nbrs.items():
            d = float(np.linalg.norm(self.pos - s.pos))
            if j not in self._edges and d <= R - self.hyst:
                self._edges.add(j)
            if j in self._edges:
                if d >= R:
                    self._edges.discard(j)
                    continue
                u -= self.gain * (2 * R - d) * (self.pos - s.pos) / (R - d) ** 2
        return task + u * dt
