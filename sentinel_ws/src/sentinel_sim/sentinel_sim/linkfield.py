"""Vectorised per-tick realisation of all broadcast links (see sentinel_core.channel).

Signal power on a directed link t->r combines mean path loss (with NLOS
penalty), reciprocal log-normal shadowing evolving with Gudmundson correlation
and independent Rician/Rayleigh fading. Interference at a receiver is the sum
of the mean received powers from the other transmitters that collide in the
same slot (each with probability ``tx_activity`` after carrier sensing).
"""
from __future__ import annotations

import numpy as np

from sentinel_core.channel import ChannelProfile, outage_table


class LinkField:
    def __init__(self, profile: ChannelProfile, n: int, rng: np.random.Generator):
        self.p = profile
        self.n = n
        self.rng = rng
        z = rng.normal(0, profile.shadow_sigma_db, (n, n))
        self.shadow = np.triu(z, 1) + np.triu(z, 1).T
        self.last = None

    def _evolve_shadow(self, pos: np.ndarray) -> None:
        if self.last is None or self.p.shadow_sigma_db <= 0:
            self.last = pos.copy()
            return
        moved = np.linalg.norm(pos - self.last, axis=1)
        rho = np.exp(-(moved[:, None] + moved[None, :]) / self.p.shadow_decorr_m)
        z = self.rng.normal(0, self.p.shadow_sigma_db, (self.n, self.n))
        z = np.triu(z, 1) + np.triu(z, 1).T
        self.shadow = rho * self.shadow + np.sqrt(1 - rho ** 2) * z
        self.last = pos.copy()

    def mean_rx_mw(self, d: np.ndarray, obs: np.ndarray, tx_offset_db: np.ndarray) -> np.ndarray:
        rx_db = self.p.rx_power_dbm(d, obs) + tx_offset_db[:, None]
        return 10.0 ** (rx_db / 10.0)

    def deliver(self, pos: np.ndarray, obs: np.ndarray, can_tx: np.ndarray,
                can_rx: np.ndarray, tx_offset_db: np.ndarray) -> np.ndarray:
        """Boolean matrix D[t, r]: broadcast from t decoded by r this tick."""
        self._evolve_shadow(pos)
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        mean = self.mean_rx_mw(d, obs, tx_offset_db)
        k_los = self.p.k_factor(0)
        k_nlos = self.p.k_factor(1)
        k = np.where(obs > 0, k_nlos, k_los)
        s = np.sqrt(k / (k + 1.0))
        sd = np.sqrt(1.0 / (2.0 * (k + 1.0)))
        re = self.rng.normal(0, 1, (self.n, self.n)) * sd + s
        im = self.rng.normal(0, 1, (self.n, self.n)) * sd
        signal = mean * 10.0 ** (self.shadow / 10.0) * (re * re + im * im)
        active = (self.rng.random(self.n) < self.p.tx_activity) & can_tx
        interf_total = (active[:, None] * mean).sum(0)            # per receiver
        interf = interf_total[None, :] - active[:, None] * mean   # exclude own signal
        noise = 10.0 ** (self.p.noise_dbm / 10.0)
        sinr_db = 10 * np.log10(np.maximum(signal, 1e-30) / (noise + np.maximum(interf, 0)))
        ok = sinr_db >= self.p.gamma_th_db
        ok &= can_tx[:, None] & can_rx[None, :]
        np.fill_diagonal(ok, False)
        return ok

    def reliable_graph(self, pos: np.ndarray, obs: np.ndarray, eps: float) -> np.ndarray:
        """Links whose predicted outage (shadowing + fading averaged) is <= eps."""
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        g = outage_table(self.p)(d, obs) <= eps
        np.fill_diagonal(g, False)
        return g
