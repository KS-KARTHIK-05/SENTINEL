"""B5: MAPPO policy agent (Yu et al., NeurIPS 2022 datasets & benchmarks).

Decentralised execution with a parameter-shared actor over local observations;
the centralised critic exists only during training (sentinel_sim.marl).
The actor is a small MLP evaluated with NumPy so simulation workers do not
need torch. Sensing, gossip and the ledger are the shared SENTINEL stack, so
only the motion decisions are learned.

Actions (every planning period): 0-7 fly toward one of 8 headings, 8 hover,
9 return toward the responder. The observation includes the direction to the same
frontier goal the PSO-based methods use, so the learner is not information-starved.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..agent import SentinelAgent

N_ACTIONS = 10
OBS_DIM = 28


def mlp_forward(params: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    """tanh MLP: params = [W0, b0, W1, b1, ..., Wk, bk]."""
    h = x
    for i in range(0, len(params) - 2, 2):
        h = np.tanh(h @ params[i] + params[i + 1])
    return h @ params[-2] + params[-1]


class MappoAgent(SentinelAgent):
    policy: Optional[List[np.ndarray]] = None   # actor parameters
    stochastic: bool = False                    # sample (training) or argmax (evaluation)
    record: Optional[list] = None               # training buffer of (t, obs, action, logp)

    def observe_local(self, t: float, nbrs) -> np.ndarray:
        lo, hi = self.bounds
        span = float(np.max(hi - lo))
        o = np.zeros(OBS_DIM)
        o[0:2] = (self.pos - lo) / span
        base = np.asarray(self.cfg.base_xy if self.cfg.base_xy is not None else lo, float)
        o[2:4] = (base - self.pos) / span
        # residual detection mass in 8 sectors (r < 60 m) around the UAV + total nearby
        res = self.map.residual()
        dx, dy = self.map.cx - self.pos[0], self.map.cy - self.pos[1]
        r = np.hypot(dx, dy)
        near = r < 60.0
        ang = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        tot = res[near].sum() + 1e-9
        for k in range(8):
            o[4 + k] = res[near & (ang == k)].sum() / tot
        o[12] = res.mean() / max(self.map.p0, 1e-9)
        # neighbours
        o[13] = min(len(nbrs), 8) / 8.0
        if nbrs:
            j = min(nbrs, key=lambda q: np.linalg.norm(nbrs[q].pos - self.pos))
            o[14:16] = (nbrs[j].pos - self.pos) / span
        o[16] = 1.0 if self.connected_to_root(t) else 0.0
        o[17] = min(t - self.root_t, 300.0) / 300.0 if np.isfinite(self.root_t) else 1.0
        uns = self.unsecured(t)
        o[18] = min(len(uns), 5) / 5.0
        o[19] = (min(max(t - e.t_first for e in uns), 300.0) / 300.0) if uns else 0.0
        o[20] = len(self.ledger.tentative()) / 5.0
        o[21] = np.linalg.norm(base - self.pos) / span
        o[22] = min(t, 1200.0) / 1200.0                # elapsed mission time
        o[23] = float((self.map.looks.sum(0) > 0).mean())  # own-map coverage fraction
        o[24] = float(self.no_fly(self.pos[None] + np.array([[5.0, 0.0]]))[0]) if self.no_fly else 0
        o[25] = float(self.no_fly(self.pos[None] + np.array([[0.0, 5.0]]))[0]) if self.no_fly else 0
        # direction to the frontier goal (same exploration information as the PSO methods)
        field_ = self.map.gain_field(self.cfg.footprint_r, [s.waypoint for s in nbrs.values()])
        g = self._frontier_goal(field_, nbrs)
        if g is not None:
            dg = g - self.pos
            o[26:28] = dg / max(np.linalg.norm(dg), 1e-9)
        return o

    def _plan(self, t: float, nbrs) -> None:
        if self.policy is None:
            return super()._plan(t, nbrs)
        # policies trained on the shorter v1 observation use its leading features
        obs = self.observe_local(t, nbrs)[: self.policy[0].shape[0]]
        logits = mlp_forward(self.policy, obs)
        z = logits - logits.max()
        p = np.exp(z) / np.exp(z).sum()
        a = int(self.rng.choice(N_ACTIONS, p=p)) if self.stochastic else int(np.argmax(p))
        if self.record is not None:
            self.record.append((t, self.id, obs, a, float(np.log(p[a] + 1e-12))))
        horizon = self.cfg.v_max * self.cfg.horizon_s
        if a < 8:
            th = a * np.pi / 4
            self.waypoint = np.clip(self.pos + horizon * np.array([np.cos(th), np.sin(th)]),
                                    self.bounds[0], self.bounds[1])
        elif a == 8:
            self.waypoint = self.pos.copy()
        else:
            base = self.cfg.base_xy if self.cfg.base_xy is not None else self.bounds[0]
            self.waypoint = np.asarray(base, float)
