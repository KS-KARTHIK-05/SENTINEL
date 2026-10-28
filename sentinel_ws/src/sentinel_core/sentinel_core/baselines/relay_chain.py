"""B8: relay chain from the responder toward the searching UAVs.

A fraction ``rho`` of the swarm (rounded, at least one relay when N >= 3, at most N - 1) does
not search: the relays form a chain on the ray from the responder to the centroid of the alive
searching UAVs, evenly spaced at ``spacing * R_0.05(1)`` (the reliable range through one
obstruction of the active channel profile), or closer when the centroid is nearer than the
chain is long. The chain is re-planned every planning period and re-formed from the remaining
relays when a relay fails. The other UAVs search with the standard PSO planner (B0) and their
reports reach the responder through the chain by the shared gossip.

Relays are the highest UAV ids, so the searchers keep a contiguous sector assignment. The chain
geometry uses the true positions of the alive UAVs (``bind_sim``), an idealisation that favours
this baseline, as the exact lambda2 does for B3.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..agent import SentinelAgent
from ..channel import reliable_range


def n_relays(n_uav: int, rho: float) -> int:
    """Number of relays: round(rho * N), at least 1 when N >= 3, never all UAVs."""
    if n_uav < 2:
        return 0
    m = int(np.floor(rho * n_uav + 0.5))
    if n_uav >= 3:
        m = max(m, 1)
    return min(m, n_uav - 1)


def chain_targets(base: np.ndarray, centroid: np.ndarray, m: int, gap: float) -> np.ndarray:
    """Positions of m relays on the ray base -> centroid, spaced ``gap`` (or D / m if closer)."""
    if m <= 0:
        return np.zeros((0, 2))
    v = np.asarray(centroid, float) - np.asarray(base, float)
    dist = float(np.linalg.norm(v))
    if dist < 1e-9:
        return np.repeat(np.asarray(base, float)[None], m, 0)
    step = min(gap, dist / m)
    k = np.arange(1, m + 1)[:, None]
    return np.asarray(base, float) + v / dist * step * k


class RelayChainAgent(SentinelAgent):
    rho: float = 0.25
    spacing: float = 0.8
    is_relay: bool = False

    def bind_sim(self, sim) -> None:
        self._sim = sim
        n = sim.N
        m = n_relays(n, self.rho)
        self.is_relay = self.id >= n - m
        if not self.is_relay:                      # sectors over the searchers only
            self.cfg = replace(self.cfg, n_agents=max(1, n - m))
        self._gap = self.spacing * reliable_range(self.profile, 0.05, 1)

    def _chain_target(self) -> np.ndarray:
        sim = self._sim
        uavs = [a for a in sim.agents[:sim.N] if a.alive]
        relays = sorted(a.id for a in uavs if getattr(a, 'is_relay', False))
        searchers = [a.pos for a in uavs if not getattr(a, 'is_relay', False)]
        base = (np.asarray(self.cfg.base_xy, float) if self.cfg.base_xy is not None
                else (self.bounds[0] + self.bounds[1]) / 2)
        centroid = np.mean(searchers, axis=0) if searchers else base
        return chain_targets(base, centroid, len(relays), self._gap)[relays.index(self.id)]

    def _desired_step(self, t, dt, nbrs):
        if not self.is_relay or getattr(self, '_sim', None) is None:
            return super()._desired_step(t, dt, nbrs)
        if t - self._last_plan >= self.cfg.plan_period_s:
            self.waypoint = self._chain_target()
            self._last_plan = t
        d = self.waypoint - self.pos
        n = np.linalg.norm(d)
        return d if n <= self.cfg.v_max * dt else d / n * self.cfg.v_max * dt
