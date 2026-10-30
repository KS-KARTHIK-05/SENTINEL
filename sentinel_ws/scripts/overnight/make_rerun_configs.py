#!/usr/bin/env python3
"""Configs for the baseline re-run of 29 Oct (docs/analysis_plan_icius2026.md, section 9): B0-B4, B6 and their fusion
variants, re-tuned on the current code, in every experiment that contains them, plus E7's
no-projection ablation (stale behaviour, see METHODS).

Each config is the original experiment spec (same base, grid and seeds) restricted to the affected
methods, with ``tuned_file`` pointing to the re-tuned values. ``--pilot`` keeps one grid cell,
one seed and the first two affected methods (queue and timing check only).

    python make_rerun_configs.py --tuned configs/tuned_v4.yaml --out configs/experiments/rerun_2026-10-29
"""
from __future__ import annotations

import argparse
import os

import yaml

BASELINES = ['B0_pso', 'B1_v1', 'B2_commpso', 'B3_lambda2', 'B4_potential', 'B6_lawnmower']
AFFECTED = set(BASELINES) | {m + '_fusion' for m in BASELINES}
# rows computed before Amendment A3 (old fusion rule), then rows that used the old tuned values
SPECS = ['e3_failures', 'e2_channel', 'e1_scalability', 'e4_dynamic', 'e6_detection',
         'e6c_false_alarms', 'e8_fastsim', 'e11_dense_diag',
         'e9_v1_rerun', 'e11b_dense_fusion', 'e12_attribution',
         'overnight/robust_fitted_v2', 'overnight/e6_fastsim_v2', 'e7_ablation']
# E7's no-projection ablation predates the 22 Oct change that evaluates the tether condition
# (which stamps tether_until) also without the projection; its rows do not reproduce on HEAD
METHODS = {'e7_ablation': {'abl_no_projection'}}


def rerun_spec(spec: dict, tuned: str, pilot: bool = False, methods: set = AFFECTED) -> dict:
    out = dict(spec)
    out['methods'] = [m for m in spec['methods'] if m in methods]
    assert out['methods'], 'no affected method'
    out['tuned_file'] = tuned
    if pilot:
        out['methods'] = out['methods'][:2]
        out['seeds'] = 1
        if 'grid' in out:
            out['grid'] = {k: v[:1] for k, v in out['grid'].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tuned', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pilot', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in SPECS:
        src = f'configs/experiments/{name}.yaml'
        spec = yaml.safe_load(open(src))
        new = rerun_spec(spec, a.tuned, a.pilot, METHODS.get(name, AFFECTED))
        dst = os.path.join(a.out, os.path.basename(name) + '.yaml')
        with open(dst, 'w') as f:
            f.write('# Baseline re-run (29 Oct, docs/analysis_plan_icius2026.md, section 9): '
                    f'{src} restricted to B0-B4, B6\n'
                    f'# (and fusion variants), tuned values from {a.tuned}.'
                    + (' PILOT.' if a.pilot else '') + '\n')
            yaml.safe_dump(new, f, sort_keys=False)
        print(f'{dst}: {new["methods"]}')


if __name__ == '__main__':
    main()
