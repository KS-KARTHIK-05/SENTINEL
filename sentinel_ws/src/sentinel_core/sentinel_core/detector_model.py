"""Stochastic thermal-detector model for the fast simulator.

The confidence produced for a true target is

    c ~ clip(N(mu, sigma^2), 0, 1),
    mu = c_max * (1 - occ)^gamma * exp(-max(0, r - r_ref) / r_scale)

where r is the slant range and ``occ`` the view-dependent occlusion. A target
partially hidden by a structure in direction ``block_dir`` is more occluded
when viewed from that side and less occluded from the opposite side:

    occ = clip(occ0 + kappa * sin(theta) * cos(az - block_dir), 0, 1)

(theta: off-nadir angle, az: azimuth of the UAV seen from the target). Buried
targets have occ = 1 from every viewpoint. False positives from warm clutter
arrive as a Poisson process per look with low confidences. All parameters are
calibrated from YOLO runs (``scripts/calibrate_detector.py``) and swept in the
detection-uncertainty sensitivity study.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Target:
    pos: np.ndarray                 # x, y [m]
    occ0: float = 0.0
    block_dir: float = 0.0          # rad
    buried: bool = False


@dataclass
class DetectorModel:
    c_max: float = 0.92
    gamma: float = 1.5
    r_ref: float = 35.0
    r_scale: float = 60.0
    sigma: float = 0.08
    kappa: float = 0.8
    fp_rate: float = 0.02           # expected false positives per look
    fp_mu: float = 0.22
    fp_sigma: float = 0.08
    report_min: float = 0.25        # detector NMS / confidence floor
    fp_noise_coupled: bool = False  # E6c: confidence noise sigma also perturbs false positives
    loc_sigma_a: float = 0.3        # geolocation noise = a + b * slant range [m]
    loc_sigma_b: float = 0.02

    def occlusion(self, tgt: Target, uav_xyz: np.ndarray) -> float:
        if tgt.buried:
            return 1.0
        v = uav_xyz[:2] - tgt.pos
        h = max(float(uav_xyz[2]), 1e-3)
        horiz = float(np.linalg.norm(v))
        sin_t = horiz / np.hypot(horiz, h)
        az = np.arctan2(v[1], v[0])
        return float(np.clip(tgt.occ0 + self.kappa * sin_t * np.cos(az - tgt.block_dir), 0, 1))

    def mean_conf(self, tgt: Target, uav_xyz: np.ndarray) -> float:
        r = float(np.linalg.norm(np.r_[uav_xyz[:2] - tgt.pos, uav_xyz[2]]))
        vis = 1.0 - self.occlusion(tgt, uav_xyz)
        return self.c_max * vis ** self.gamma * np.exp(-max(0.0, r - self.r_ref) / self.r_scale)

    def look(self, targets: List[Target], uav_xyz: np.ndarray, footprint_r: float,
             rng: np.random.Generator, clutter_xy=None) -> List[Tuple[np.ndarray, float, int]]:
        """Detections (world xy, confidence, target index or -1 for FP) of one look."""
        out = []
        for k, t in enumerate(targets):
            if np.linalg.norm(t.pos - uav_xyz[:2]) > footprint_r:
                continue
            c = float(np.clip(rng.normal(self.mean_conf(t, uav_xyz), self.sigma), 0, 1))
            if c >= self.report_min:
                r = float(np.linalg.norm(np.r_[uav_xyz[:2] - t.pos, uav_xyz[2]]))
                s = self.loc_sigma_a + self.loc_sigma_b * r
                out.append((t.pos + rng.normal(0, s, 2), c, k))
        for _ in range(rng.poisson(self.fp_rate)):
            if clutter_xy is not None and len(clutter_xy):
                base = clutter_xy[rng.integers(len(clutter_xy))]
                if np.linalg.norm(base - uav_xyz[:2]) > footprint_r:
                    continue
            else:
                ang, rr = rng.random() * 2 * np.pi, footprint_r * np.sqrt(rng.random())
                base = uav_xyz[:2] + rr * np.array([np.cos(ang), np.sin(ang)])
            sd = np.hypot(self.fp_sigma, self.sigma) if self.fp_noise_coupled else self.fp_sigma
            c = float(np.clip(rng.normal(self.fp_mu, sd), 0, 1))
            if c >= self.report_min:
                out.append((np.asarray(base, float) + rng.normal(0, 0.5, 2), c, -1))
        return out
