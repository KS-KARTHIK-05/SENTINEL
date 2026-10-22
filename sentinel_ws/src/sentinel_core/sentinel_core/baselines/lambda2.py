"""B3: algebraic-connectivity gradient control (Sabattini et al., IJRR 2013).

Edge weights a_ij = exp(-d^2 / (2 s^2)) for d <= R (0 otherwise), with s chosen
so that a(R) = delta. The connectivity control is the gradient of the
potential V(lambda2) = coth(lambda2 - lam_min):

    u_i = -dV/dlambda2 * sum_j da_ij/dx_i (v_i - v_j)^2

added to the task velocity of the standard PSO planner (B0). ``oracle`` is set
by the simulator and returns the exact lambda2 and Fiedler vector of the
current weighted graph, an idealisation of the decentralised power-iteration
estimator that favours this baseline.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..agent import SentinelAgent


class Lambda2Agent(SentinelAgent):
    gain: float = 40.0
    lam_min: float = 0.02
    delta: float = 0.01
    oracle: Optional[Callable[[], Tuple[float, Dict[int, float], Dict[int, np.ndarray]]]] = None

    def _desired_step(self, t, dt, nbrs):
        task = super()._desired_step(t, dt, nbrs)
        if self.oracle is None:
            return task
        lam2, v2, pos = self.oracle()
        if self.id not in v2:
            return task
        R = self.r_hard
        s2 = -R * R / (2.0 * np.log(self.delta))
        grad = np.zeros(2)
        for j, xj in pos.items():
            if j == self.id:
                continue
            diff = self.pos - xj
            d2 = float(diff @ diff)
            if d2 > R * R:
                continue
            a = np.exp(-d2 / (2 * s2))
            grad += -a * diff / s2 * (v2[self.id] - v2[j]) ** 2
        gap = max(lam2 - self.lam_min, 1e-3)
        u = self.gain * grad / np.sinh(gap) ** 2       # -dV/dlambda2 = 1/sinh^2
        return task + u * dt
