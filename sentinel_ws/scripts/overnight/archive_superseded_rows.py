#!/usr/bin/env python3
"""Copy the rows that the baseline re-run of 29 Oct supersedes (PLAN.md section 9) into
``results/baseline_rerun_2026-10-29/superseded_rows/<experiment>.parquet``. The original result
folders are not changed; ``make_ios.py`` replaces these rows by the re-run rows when it reads them.

    python archive_superseded_rows.py
"""
from __future__ import annotations

import os

import pandas as pd

import make_rerun_configs as RC

RES = os.path.expanduser('~/sentinel_ws/results')
OUT = os.path.join(RES, 'baseline_rerun_2026-10-29', 'superseded_rows')
# where each experiment's stored rows live (C1/C2 v2 were run on 29 Oct into the E5 re-check folder)
ROOT = {'robust_fitted_v2': os.path.join(RES, 'e5_recheck_2026-10-29'),
        'e6_fastsim_v2': os.path.join(RES, 'e5_recheck_2026-10-29')}


def main():
    os.makedirs(OUT, exist_ok=True)
    for spec in RC.SPECS:
        exp = os.path.basename(spec)
        src = os.path.join(ROOT.get(exp, RES), exp, 'summary.parquet')
        df = pd.read_parquet(src)
        rows = df[df.cfg_method.isin(RC.METHODS.get(spec, RC.AFFECTED))]
        assert len(rows), exp
        rows.to_parquet(os.path.join(OUT, f'{exp}.parquet'))
        print(f'{exp}: {len(rows)} of {len(df)} rows from {src}')


if __name__ == '__main__':
    main()
