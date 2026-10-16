"""Probabilistic search map shared through grow-only look counters.

Each cell holds the residual probability that an undetected target is present.
After ``k`` independent looks with detection probability ``P_d`` the Bayesian
update gives ``p_k = p0 q^k / (p0 q^k + 1 - p0)`` with ``q = 1 - P_d``. Looks
are stored per source UAV (a G-counter CRDT); merging two maps is the
element-wise maximum per source, so repeated gossip never double-counts.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import fftconvolve


class SearchMap:
    def __init__(self, size_m: Tuple[float, float], cell_m: float, n_sources: int,
                 p0: float = 0.05, p_detect: float = 0.8, origin=(0.0, 0.0)):
        self.cell = float(cell_m)
        self.origin = np.asarray(origin, float)
        self.shape = (int(np.ceil(size_m[0] / cell_m)), int(np.ceil(size_m[1] / cell_m)))
        self.p0 = p0
        self.q = 1.0 - p_detect
        self.looks = np.zeros((n_sources,) + self.shape, dtype=np.uint16)
        self.total = np.zeros(self.shape, dtype=np.int64)     # sum over sources (cached)
        self.ver = np.zeros(n_sources, dtype=np.int64)        # looks recorded per source row
        self._kernels = {}
        xs = self.origin[0] + (np.arange(self.shape[0]) + 0.5) * self.cell
        ys = self.origin[1] + (np.arange(self.shape[1]) + 0.5) * self.cell
        self.cx, self.cy = np.meshgrid(xs, ys, indexing='ij')

    # ------------------------------------------------------------- updates
    def _window(self, pos, radius: float):
        """Index window and disk mask around ``pos`` (cost independent of map size)."""
        i0 = max(int((pos[0] - radius - self.origin[0]) / self.cell), 0)
        i1 = min(int((pos[0] + radius - self.origin[0]) / self.cell) + 2, self.shape[0])
        j0 = max(int((pos[1] - radius - self.origin[1]) / self.cell), 0)
        j1 = min(int((pos[1] + radius - self.origin[1]) / self.cell) + 2, self.shape[1])
        if i0 >= i1 or j0 >= j1:
            return None
        sl = (slice(i0, i1), slice(j0, j1))
        m = (self.cx[sl] - pos[0]) ** 2 + (self.cy[sl] - pos[1]) ** 2 <= radius * radius
        return sl, m

    def record_look(self, source: int, pos, radius: float) -> None:
        w = self._window(pos, radius)
        if w is None:
            return
        sl, m = w
        cur = self.looks[source][sl]
        inc = m & (cur < np.iinfo(np.uint16).max)
        cur[inc] += 1
        self.total[sl][inc] += 1
        self.ver[source] += 1

    def rows(self):
        """Gossip payload: the G-counter rows and their versions."""
        return self.looks, self.ver

    def merge_rows(self, looks: np.ndarray, ver: np.ndarray) -> None:
        """G-counter merge; only rows with a newer version are touched.

        Row s is only ever incremented by its owner, so a copy with a larger
        version dominates element-wise and replaces the local row.
        """
        for s in np.flatnonzero(ver > self.ver):
            self.total += looks[s].astype(np.int64) - self.looks[s]
            self.looks[s] = looks[s]
            self.ver[s] = ver[s]

    def merge(self, other: 'SearchMap') -> None:
        self.merge_rows(other.looks, other.ver)

    # ------------------------------------------------------------- queries
    def residual(self) -> np.ndarray:
        pq = self.p0 * self.q ** self.total
        return pq / (pq + 1.0 - self.p0)

    def coverage_fraction(self) -> float:
        return float((self.total > 0).mean())

    def disk_kernel(self, radius: float) -> np.ndarray:
        if radius in self._kernels:
            return self._kernels[radius]
        r = int(np.ceil(radius / self.cell))
        g = np.arange(-r, r + 1) * self.cell
        gx, gy = np.meshgrid(g, g, indexing='ij')
        k = (gx * gx + gy * gy <= radius * radius).astype(float)
        self._kernels[radius] = k
        return k

    def gain_field(self, radius: float, exclude: list = ()) -> np.ndarray:
        """Expected detections from a look centred on each cell.

        ``exclude`` lists (x, y) look centres already claimed by neighbours;
        cells inside those footprints contribute nothing (no double search).
        """
        p = self.residual() * (1.0 - self.q)
        for c in exclude:
            w = self._window(c, radius)
            if w is not None:
                p[w[0]][w[1]] = 0.0
        return np.maximum(fftconvolve(p, self.disk_kernel(radius), mode='same'), 0.0)

    def sample(self, field: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Nearest-cell lookup of ``field`` at world points (P x 2)."""
        ij = np.floor((np.asarray(pts, float) - self.origin) / self.cell).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, self.shape[0] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, self.shape[1] - 1)
        return field[ij[:, 0], ij[:, 1]]
