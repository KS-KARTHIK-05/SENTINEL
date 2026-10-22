"""Evaluation metrics and statistics (definitions match Section VI of the paper).

Network
  NRR(t)        fraction of alive UAVs in the main component of the reliable
                graph G_eps (the component containing the responder node if one
                exists, otherwise the largest component)
  connected(t)  G_eps restricted to alive nodes has a single component
  lambda2(t)    algebraic connectivity of G_eps over alive nodes
  recovery      time from a failure until connected(t) holds again
Information
  TDR           |true targets confirmed in at least one alive ledger| / |targets|
  info loss     |targets ever confirmed but held by no alive ledger| / |ever confirmed|
  false / dup   confirmed entries matching no target / matching an already
                matched target, per ledger (the v1 '15 coordinates' check)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


def match_entries(entry_pos: Sequence[np.ndarray], targets: np.ndarray,
                  gate: float = 5.0) -> Tuple[set, int, int]:
    """Greedy nearest matching of ledger entries to ground-truth targets.

    Returns (matched target indices, false entries, duplicate entries).
    """
    matched, false, dup = set(), 0, 0
    if len(targets) == 0:
        return matched, len(entry_pos), 0
    for p in entry_pos:
        d = np.linalg.norm(targets - np.asarray(p)[None], axis=1)
        k = int(np.argmin(d))
        if d[k] > gate:
            false += 1
        elif k in matched:
            dup += 1
        else:
            matched.add(k)
    return matched, false, dup


def main_component(comps: List[List[int]], base: int | None) -> List[int]:
    if not comps:
        return []
    if base is not None:
        for c in comps:
            if base in c:
                return c
    return max(comps, key=len)


def recovery_times(t: np.ndarray, connected: np.ndarray, fail_times: Iterable[float]) -> List[float]:
    """Time to regain connectivity after each failure (nan if never regained)."""
    out = []
    for tf in fail_times:
        idx = np.flatnonzero(t >= tf)
        if len(idx) == 0:
            continue
        after = connected[idx]
        ok = np.flatnonzero(after)
        out.append(float(t[idx[ok[0]]] - tf) if len(ok) else float('nan'))
    return out


def time_to_reach(t: np.ndarray, y: np.ndarray, level: float) -> float:
    idx = np.flatnonzero(y >= level)
    return float(t[idx[0]]) if len(idx) else float('nan')


# ------------------------------------------------------------------ statistics
def bootstrap_ci(x: Sequence[float], n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None]).sum()
    lt = (a[:, None] < b[None]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def paired_wilcoxon_holm(ref: Sequence[float], others: Dict[str, Sequence[float]]) -> Dict[str, dict]:
    """Paired (same seeds) Wilcoxon signed-rank tests with Holm-Bonferroni correction."""
    raw = {}
    for name, x in others.items():
        a, b = np.asarray(ref, float), np.asarray(x, float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
            raw[name] = 1.0
        else:
            raw[name] = float(stats.wilcoxon(a[ok], b[ok]).pvalue)
    order = sorted(raw, key=raw.get)
    m = len(order)
    out, running = {}, 0.0
    for k, name in enumerate(order):
        running = max(running, min(1.0, (m - k) * raw[name]))
        out[name] = {'p_raw': raw[name], 'p_holm': running,
                     'cliffs_delta': cliffs_delta(ref, others[name])}
    return out
