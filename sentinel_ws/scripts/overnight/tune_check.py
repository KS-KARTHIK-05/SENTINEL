#!/usr/bin/env python3
"""Tuning objective of stored tuned values on the current code.

Evaluates ``sentinel_sim.tune``'s objective (mean report value over the tuning scenarios and
seeds) for each method with the values in a tuned file, exactly as a tuning trial would. Used to
check whether a method's tuning still holds on the current code (same value as recorded) and to
compare old and re-tuned values.

    python tune_check.py --tuned configs/tuned.yaml --methods sentinel B5_mappo B0_pso --jobs 18
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

from sentinel_sim import tune as T


def configs(method: str, over: dict) -> list:
    out = []
    for sc in T.TUNE_SCENARIOS:
        for s in T.TUNE_SEEDS:
            c = copy.deepcopy(T.TUNE_BASE)
            c.update(copy.deepcopy(sc))
            for k, v in over.items():
                c.setdefault(k, {}).update(v) if isinstance(v, dict) else c.__setitem__(k, v)
            c.update(method=method, seed=s)
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tuned', required=True)
    ap.add_argument('--methods', nargs='+', required=True)
    ap.add_argument('--jobs', type=int, default=18)
    ap.add_argument('--policy', default='configs/mappo_policy.npz',
                    help='B5 policy (the tuning scenarios set none; the selection passed it)')
    a = ap.parse_args()
    tuned = yaml.safe_load(open(a.tuned)) or {}
    tuned.setdefault('B5_mappo', {})['policy_path'] = os.path.abspath(a.policy)
    res = {}
    with ProcessPoolExecutor(a.jobs) as pool:
        futs = {m: [pool.submit(T._eval, c) for c in configs(m, tuned.get(m, {}))]
                for m in a.methods}
        for m, fs in futs.items():
            vals = [f.result() for f in fs]
            res[m] = float(np.mean(vals))
            print(f'{m}: objective {res[m]:.4f} ({a.tuned})', flush=True)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
