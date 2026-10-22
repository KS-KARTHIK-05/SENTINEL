"""Failure schedules for robustness experiments.

kinds:
  crash    - the UAV and every ledger copy it holds disappear (hard failure)
  link_cut - the UAV keeps flying but can neither send nor receive
  brownout - transmit power decays linearly by ``drop_db`` over ``duration``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class FailureEvent:
    t: float
    uav: int
    kind: str = 'crash'
    duration: float = 0.0
    drop_db: float = 0.0


def single(t: float, uav: int, kind: str = 'crash') -> List[FailureEvent]:
    return [FailureEvent(t, uav, kind)]


def simultaneous(t: float, uavs: Sequence[int], kind: str = 'crash') -> List[FailureEvent]:
    return [FailureEvent(t, u, kind) for u in uavs]


def cascading(t0: float, uavs: Sequence[int], gap: float, kind: str = 'crash') -> List[FailureEvent]:
    return [FailureEvent(t0 + k * gap, u, kind) for k, u in enumerate(uavs)]


def poisson(rate_per_s: float, horizon: float, n_uav: int, rng: np.random.Generator,
            max_failures: int | None = None, kind: str = 'crash') -> List[FailureEvent]:
    events, t, alive = [], 0.0, list(range(n_uav))
    while alive:
        t += rng.exponential(1.0 / rate_per_s)
        if t >= horizon or (max_failures is not None and len(events) >= max_failures):
            break
        u = alive.pop(int(rng.integers(len(alive))))
        events.append(FailureEvent(t, u, kind))
    return events


def brownout(t: float, uav: int, duration: float, drop_db: float) -> List[FailureEvent]:
    return [FailureEvent(t, uav, 'brownout', duration, drop_db)]


def random_subset(n_uav: int, k: int, rng: np.random.Generator) -> List[int]:
    return sorted(int(u) for u in rng.choice(n_uav, size=min(k, n_uav), replace=False))
