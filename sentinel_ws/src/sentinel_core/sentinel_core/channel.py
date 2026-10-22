"""Air-to-air (UAV-to-UAV) wireless channel model.

The received signal-to-interference-plus-noise ratio is built from:

* log-distance path loss anchored at the free-space loss at ``d0``,
* an additional per-obstruction penalty for non-line-of-sight (NLOS) links,
* log-normal shadowing with Gudmundson spatial correlation,
* small-scale Rician fading (Rayleigh when K = 0),
* aggregate interference from concurrently transmitting UAVs.

A packet is lost when the instantaneous SINR falls below the decoding
threshold ``gamma_th_db`` (block-fading outage model). ``outage_prob`` gives
the analytical outage probability averaged over shadowing and fading, which
the mobility layer uses as a link-reliability predictor; ``LinkSampler`` draws
per-packet realisations for the simulators.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2

C_LIGHT = 299_792_458.0
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(24)


@dataclass(frozen=True)
class ChannelProfile:
    """Link-budget and propagation parameters of one environment class."""

    name: str
    freq_hz: float = 2.4e9
    tx_power_dbm: float = 20.0
    antenna_gain_dbi: float = 2.0          # per side
    bandwidth_hz: float = 20e6
    noise_figure_db: float = 7.0
    gamma_th_db: float = 5.0               # decoding threshold (lowest MCS)
    d0_m: float = 1.0
    path_loss_exp: float = 2.0
    shadow_sigma_db: float = 0.0
    shadow_decorr_m: float = 20.0          # Gudmundson decorrelation distance
    rician_k_db: float = 10.0              # LoS K-factor; -inf -> Rayleigh
    nlos_k_db: float = float('-inf')       # K-factor once obstructed
    nlos_loss_db: float = 0.0              # extra loss per obstruction
    max_obstructions: int = 3              # cap on counted obstructions
    tx_activity: float = 0.1               # per-slot transmit probability of interferers

    # ------------------------------------------------------------------ budget
    @property
    def noise_dbm(self) -> float:
        return -174.0 + 10.0 * np.log10(self.bandwidth_hz) + self.noise_figure_db

    @property
    def pl0_db(self) -> float:
        return 20.0 * np.log10(4.0 * np.pi * self.d0_m * self.freq_hz / C_LIGHT)

    def path_loss_db(self, d, n_obs=0):
        d = np.maximum(np.asarray(d, dtype=float), self.d0_m)
        n_obs = np.minimum(np.asarray(n_obs), self.max_obstructions)
        return (self.pl0_db + 10.0 * self.path_loss_exp * np.log10(d / self.d0_m)
                + self.nlos_loss_db * n_obs)

    def rx_power_dbm(self, d, n_obs=0):
        return (self.tx_power_dbm + 2.0 * self.antenna_gain_dbi
                - self.path_loss_db(d, n_obs))

    def mean_snr_db(self, d, n_obs=0, interference_mw=0.0):
        noise_mw = 10.0 ** (self.noise_dbm / 10.0) + interference_mw
        return self.rx_power_dbm(d, n_obs) - 10.0 * np.log10(noise_mw)

    def k_factor(self, n_obs) -> float:
        k_db = self.rician_k_db if n_obs == 0 else self.nlos_k_db
        return 0.0 if np.isinf(k_db) else 10.0 ** (k_db / 10.0)


def _fading_cdf(z, k_lin: float):
    """CDF of unit-mean Rician power gain evaluated at z (Rayleigh if k=0)."""
    z = np.maximum(z, 0.0)
    if k_lin <= 0.0:
        return -np.expm1(-z)
    return ncx2.cdf(2.0 * (k_lin + 1.0) * z, df=2, nc=2.0 * k_lin)


def outage_prob(profile: ChannelProfile, d, n_obs: int = 0,
                interference_mw: float = 0.0):
    """P(SINR < gamma_th) averaged over log-normal shadowing and fading."""
    snr_db = profile.mean_snr_db(d, n_obs, interference_mw)
    k_lin = profile.k_factor(int(np.minimum(n_obs, profile.max_obstructions)))
    sigma = profile.shadow_sigma_db
    if sigma <= 0.0:
        return _fading_cdf(10.0 ** ((profile.gamma_th_db - snr_db) / 10.0), k_lin)
    # Gauss-Hermite quadrature over X ~ N(0, sigma^2) dB
    snr_db = np.asarray(snr_db, dtype=float)[..., None]
    x = np.sqrt(2.0) * sigma * _GH_X
    p = _fading_cdf(10.0 ** ((profile.gamma_th_db - snr_db - x) / 10.0), k_lin)
    return (p * _GH_W).sum(axis=-1) / np.sqrt(np.pi)


class OutageTable:
    """Interpolated ``outage_prob`` (no interference) on a log-distance grid.

    The analytical expression needs a non-central chi-square CDF per call;
    the table makes the per-candidate evaluation inside PSO cheap. The maximum
    interpolation error is checked in the unit tests.
    """

    def __init__(self, profile: ChannelProfile, d_min: float = 1.0, d_max: float = 5000.0,
                 n: int = 4000):
        self.profile = profile
        self.logd = np.linspace(np.log(d_min), np.log(d_max), n)
        d = np.exp(self.logd)
        self.p = np.stack([outage_prob(profile, d, k)
                           for k in range(profile.max_obstructions + 1)])

    def __call__(self, d, n_obs=0):
        d = np.asarray(d, float)
        k = np.minimum(np.asarray(n_obs, int), self.profile.max_obstructions)
        x = np.log(np.clip(d, np.exp(self.logd[0]), np.exp(self.logd[-1])))
        if np.ndim(k) == 0:
            return np.interp(x, self.logd, self.p[int(k)])
        out = np.empty(np.broadcast(x, k).shape)
        xb, kb = np.broadcast_arrays(x, k)
        for kk in np.unique(kb):
            sel = kb == kk
            out[sel] = np.interp(xb[sel], self.logd, self.p[kk])
        return out


_TABLES: Dict[ChannelProfile, OutageTable] = {}


def outage_table(profile: ChannelProfile) -> OutageTable:
    t = _TABLES.get(profile)
    if t is None:
        t = _TABLES[profile] = OutageTable(profile)
    return t


def reliable_range(profile: ChannelProfile, eps: float, n_obs: int = 0,
                   interference_mw: float = 0.0, d_max: float = 1e4) -> float:
    """Largest distance R_eps with outage probability <= eps."""
    f = lambda d: float(outage_prob(profile, d, n_obs, interference_mw)) - eps  # noqa: E731
    lo = profile.d0_m
    if f(lo) > 0.0:
        return 0.0
    if f(d_max) < 0.0:
        return d_max
    return brentq(f, lo, d_max, xtol=1e-3)


# Environment classes. Exponents, shadowing and K-factors follow the ranges
# reported for low-altitude air-to-air and urban street-level links
# (3GPP TR 36.777; ITU-R P.1411); ``urban_rubble`` is the nominal disaster
# scenario and is calibrated so that its NLOS reliable range is close to the
# 50 m figure used by SENTINEL v1.
PROFILES: Dict[str, ChannelProfile] = {
    'open': ChannelProfile('open', path_loss_exp=2.0, shadow_sigma_db=2.0,
                           rician_k_db=10.0, nlos_k_db=3.0, nlos_loss_db=6.0),
    'suburban': ChannelProfile('suburban', path_loss_exp=2.5, shadow_sigma_db=4.0,
                               rician_k_db=6.0, nlos_k_db=0.0, nlos_loss_db=10.0),
    'urban_rubble': ChannelProfile('urban_rubble', path_loss_exp=3.0, shadow_sigma_db=6.0,
                                   rician_k_db=3.0, nlos_loss_db=6.0),
    'severe': ChannelProfile('severe', path_loss_exp=3.5, shadow_sigma_db=8.0,
                             rician_k_db=float('-inf'), nlos_loss_db=8.0),
}


def get_profile(name: str, **overrides) -> ChannelProfile:
    return replace(PROFILES[name], **overrides) if overrides else PROFILES[name]


class LinkSampler:
    """Draws per-packet link realisations with spatially correlated shadowing."""

    def __init__(self, profile: ChannelProfile, rng: np.random.Generator):
        self.profile = profile
        self.rng = rng
        self._shadow: Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]] = {}

    def _shadowing(self, i: int, j: int, pi: np.ndarray, pj: np.ndarray) -> float:
        sigma = self.profile.shadow_sigma_db
        if sigma <= 0.0:
            return 0.0
        key = (i, j) if i < j else (j, i)
        a, b = (pi, pj) if i < j else (pj, pi)
        prev = self._shadow.get(key)
        if prev is None:
            x = self.rng.normal(0.0, sigma)
        else:
            x0, a0, b0 = prev
            moved = np.linalg.norm(a - a0) + np.linalg.norm(b - b0)
            rho = np.exp(-moved / self.profile.shadow_decorr_m)
            x = rho * x0 + np.sqrt(1.0 - rho * rho) * self.rng.normal(0.0, sigma)
        self._shadow[key] = (x, a.copy(), b.copy())
        return x

    def _fading(self, n_obs: int) -> float:
        k = self.profile.k_factor(min(n_obs, self.profile.max_obstructions))
        s = np.sqrt(k / (k + 1.0))
        sd = np.sqrt(1.0 / (2.0 * (k + 1.0)))
        re, im = self.rng.normal(s, sd), self.rng.normal(0.0, sd)
        return re * re + im * im

    def rx_power_mw(self, i, j, pi, pj, n_obs: int) -> float:
        d = float(np.linalg.norm(np.asarray(pi) - np.asarray(pj)))
        p_db = (self.profile.rx_power_dbm(d, n_obs)
                + self._shadowing(i, j, np.asarray(pi, float), np.asarray(pj, float)))
        return 10.0 ** (p_db / 10.0) * self._fading(n_obs)

    def sinr_db(self, signal_mw: float, interference_mw: float = 0.0) -> float:
        noise_mw = 10.0 ** (self.profile.noise_dbm / 10.0)
        return 10.0 * np.log10(max(signal_mw, 1e-30) / (noise_mw + interference_mw))

    def delivered(self, i, j, pi, pj, n_obs: int, interference_mw: float = 0.0) -> bool:
        s = self.rx_power_mw(i, j, pi, pj, n_obs)
        return self.sinr_db(s, interference_mw) >= self.profile.gamma_th_db
