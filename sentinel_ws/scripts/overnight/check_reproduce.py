#!/usr/bin/env python3
"""Re-run stored missions and require identical summaries (no-regression check).

Picks one stored run per method from an existing results folder, re-runs its exact config and
compares every summary column except wall-clock timings.

    python check_reproduce.py ~/sentinel_ws/results/e3_failures --per-method 1
"""
from __future__ import annotations

import argparse
import json
import math
import sys

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
    a = ap.parse_args()
    df = pd.read_parquet(f'{a.results}/summary.parquet')
    bad = 0
    for m, g in df.groupby('cfg_method'):
        if a.methods and m not in a.methods:
            continue
        for _, row in g.sort_values('cfg_seed').head(a.per_method).iterrows():
            new = run(json.loads(row['config']))['summary']
            diff = [k for k in new if k not in SKIP and k in row and not same(new[k], row[k])]
            print(f'{m:22s} seed {row["cfg_seed"]:3d}: '
                  f'{"IDENTICAL" if not diff else "DIFFERENT " + str(diff)}', flush=True)
            bad += bool(diff)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
