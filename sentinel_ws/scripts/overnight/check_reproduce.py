#!/usr/bin/env python3
"""Re-run stored missions and require identical summaries (no-regression check).

Picks one stored run per method from an existing results folder, re-runs its exact config and
compares every summary column except wall-clock timings.

    python check_reproduce.py ~/sentinel_ws/results/e3_failures --per-method 1
    python check_reproduce.py ~/sentinel_ws/results/e3_failures --spread-seeds
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from sentinel_sim.sim import run

SKIP = {'wall_s', 'compute_ms_mean', 'compute_ms_p95', 'run_id', 'config'}


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--per-method', type=int, default=1)
    ap.add_argument('--methods', nargs='*')
    ap.add_argument('--spread-seeds', action='store_true',
                    help='per method, the first stored row of the lowest seed and the last of the '
                         'highest seed (two seeds, usually two settings) instead of the first rows')
    ap.add_argument('--jobs', type=int, default=1, help='missions re-run in parallel')
    a = ap.parse_args()
    df = pd.read_parquet(f'{a.results}/summary.parquet')
    rows = []
    for m, g in df.groupby('cfg_method'):
        if a.methods and m not in a.methods:
            continue
        if a.spread_seeds:
            pick = pd.concat([g[g.cfg_seed == g.cfg_seed.min()].head(1),
                              g[g.cfg_seed == g.cfg_seed.max()].tail(1)])
        else:
            pick = g.sort_values('cfg_seed').head(a.per_method)
        rows += [row for _, row in pick.iterrows()]
    with ProcessPoolExecutor(a.jobs) as pool:
        futs = [pool.submit(_summary, row['config']) for row in rows]
        bad = 0
        for row, fut in zip(rows, futs):
            new = fut.result()
            diff = [k for k in new if k not in SKIP and k in row and not same(new[k], row[k])]
            print(f'{row["cfg_method"]:22s} seed {row["cfg_seed"]:3d}: '
                  f'{"IDENTICAL" if not diff else "DIFFERENT " + str(diff)}', flush=True)
            bad += bool(diff)
    sys.exit(1 if bad else 0)


def _summary(config: str) -> dict:
    return run(json.loads(config))['summary']


if __name__ == '__main__':
    main()
