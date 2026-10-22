"""Replicated world-frame target ledger with confidence fusion.

Every UAV keeps its own ``Ledger``. Observations are associated to entries by
world-frame distance (gating), fused per observer view in log-odds space, and
exchanged with neighbours by gossip. An entry is *replicated* once at least one
other UAV has acknowledged holding it; ``unreplicated_count`` is the quantity
``U_i(t)`` that drives the information-aware spring tension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

HIGH_CONF = 0.70
LOW_CONF = 0.40
FUSION_RULE = 'corroborative'     # corroborative (default, Amendment A3) | naive_bayes (ablation)


def set_thresholds(low: float = 0.40, high: float = 0.70) -> None:
    """Set the SA band [low, high) and the confirmation threshold (process-wide)."""
    global LOW_CONF, HIGH_CONF
    LOW_CONF, HIGH_CONF = float(low), float(high)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class Entry:
    uid: Tuple[int, int]                 # (creator uav, local sequence) - globally unique
    pos: np.ndarray                      # world-frame x, y [m]
    views: Dict[Tuple[int, int], float]  # (observer, view sector) -> best confidence
    t_first: float
    t_update: float
    holders: Set[int] = field(default_factory=set)   # UAVs known to hold a copy
    sa_requested: bool = False
    weight: float = 1.0                  # accumulated position weight

    @property
    def confidence(self) -> float:
        """Corroborative log-odds fusion of views from distinct viewpoints.

        fused logit = best view's logit + sum of the *positive* logits of the
        other views. A weak detection from an occluded angle is not evidence of
        absence, so it cannot veto a good view (plain naive-Bayes summation lets
        many low-confidence views pull a confirmed victim below the threshold,
        which hurt dense swarms); corroborating views still raise confidence.
        With one view this reduces to that view's confidence.
        """
        if FUSION_RULE == 'naive_bayes':
            return sigmoid(sum(logit(c) for c in self.views.values()))
        ls = sorted((logit(c) for c in self.views.values()), reverse=True)
        return sigmoid(ls[0] + sum(max(0.0, x) for x in ls[1:]))

    @property
    def confirmed(self) -> bool:
        return self.confidence >= HIGH_CONF

    @property
    def tentative(self) -> bool:
        return LOW_CONF <= self.confidence < HIGH_CONF

    def copy(self) -> 'Entry':
        return Entry(self.uid, self.pos.copy(), dict(self.views), self.t_first, self.t_update,
                     set(self.holders), self.sa_requested, self.weight)


class Ledger:
    def __init__(self, owner: int, gate_m: float = 5.0, n_sectors: int = 8):
        self.owner = owner
        self.gate_m = gate_m
        self.n_sectors = n_sectors
        self.entries: Dict[Tuple[int, int], Entry] = {}
        self.aliases: Dict[Tuple[int, int], Tuple[int, int]] = {}   # merged uid -> kept uid
        self._seq = 0

    # ----------------------------------------------------------- association
    def _nearest(self, pos: np.ndarray) -> Optional[Entry]:
        best, best_d = None, self.gate_m
        for e in self.entries.values():
            d = float(np.linalg.norm(e.pos - pos))
            if d <= best_d:
                best, best_d = e, d
        return best

    def _sector(self, observer_pos: np.ndarray, target_pos: np.ndarray) -> int:
        v = np.asarray(observer_pos[:2], float) - target_pos
        ang = np.arctan2(v[1], v[0]) % (2 * np.pi)
        return int(ang / (2 * np.pi / self.n_sectors)) % self.n_sectors

    def observe(self, pos, conf: float, t: float, observer_pos) -> Entry:
        """Add one detection (world frame) made by this ledger's owner."""
        pos = np.asarray(pos[:2], float)
        key = (self.owner, self._sector(np.asarray(observer_pos, float), pos))
        e = self._nearest(pos)
        if e is None:
            e = Entry((self.owner, self._seq), pos.copy(), {key: conf}, t, t, {self.owner})
            self._seq += 1
            self.entries[e.uid] = e
            return e
        w = conf
        e.pos = (e.pos * e.weight + pos * w) / (e.weight + w)
        e.weight += w
        e.views[key] = max(e.views.get(key, 0.0), conf)
        e.t_update = t
        e.holders.add(self.owner)
        return e

    # ---------------------------------------------------------------- gossip
    def digest(self) -> List[Entry]:
        return [e.copy() for e in self.entries.values()]

    def _resolve(self, uid: Tuple[int, int]) -> Tuple[int, int]:
        while uid in self.aliases:
            uid = self.aliases[uid]
        return uid

    def merge(self, remote: Iterable[Entry]) -> List[Tuple[int, int]]:
        """Merge a neighbour's entries; returns the remote uids now held (for the ack).

        Entries describing the same physical target but created independently
        by different UAVs are fused by ``consolidate`` (smallest uid kept).
        """
        held = []
        for r in remote:
            local = self.entries.get(self._resolve(r.uid))
            if local is None:
                local = r.copy()
                local.holders.add(self.owner)
                self.entries[local.uid] = local
            else:
                for k, v in r.views.items():
                    local.views[k] = max(v, local.views.get(k, 0.0))
                local.holders |= r.holders | {self.owner}
                local.sa_requested |= r.sa_requested
                local.t_update = max(local.t_update, r.t_update)
            held.append(r.uid)
        self.consolidate()
        return held

    def consolidate(self) -> None:
        """Fuse entries whose positions converged inside the gate (same target)."""
        changed = True
        while changed:
            changed = False
            es = sorted(self.entries.values(), key=lambda e: e.uid)
            for a_i, a in enumerate(es):
                for b in es[a_i + 1:]:
                    if np.linalg.norm(a.pos - b.pos) <= self.gate_m:
                        a.pos = (a.pos * a.weight + b.pos * b.weight) / (a.weight + b.weight)
                        a.weight += b.weight
                        for k, v in b.views.items():
                            a.views[k] = max(v, a.views.get(k, 0.0))
                        a.holders |= b.holders
                        a.sa_requested |= b.sa_requested
                        a.t_first = min(a.t_first, b.t_first)
                        a.t_update = max(a.t_update, b.t_update)
                        self.aliases[b.uid] = a.uid
                        del self.entries[b.uid]
                        changed = True
                        break
                if changed:
                    break

    def ack(self, uids: Iterable[Tuple[int, int]], by: int) -> None:
        for uid in uids:
            e = self.entries.get(self._resolve(uid))
            if e is not None:
                e.holders.add(by)

    def forget_holder(self, uav: int) -> None:
        """Drop a failed UAV from the holder sets (its copies are gone)."""
        for e in self.entries.values():
            e.holders.discard(uav)

    # --------------------------------------------------------------- queries
    def unreplicated_count(self) -> int:
        return sum(1 for e in self.entries.values()
                   if e.confidence >= LOW_CONF and e.holders <= {self.owner})

    def confirmed(self) -> List[Entry]:
        return [e for e in self.entries.values() if e.confirmed]

    def tentative(self) -> List[Entry]:
        return [e for e in self.entries.values() if e.tentative]
