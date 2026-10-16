"""Equal-budget hyperparameter tuning of every method (fair baselines, R2.6).

Every method gets the same Optuna TPE budget (``--trials``), evaluated on the
same tuning scenarios with seeds disjoint from the evaluation seeds (1000+).
The objective is the time-discounted report value averaged over a dense and a
sparse communication regime. The best parameters are written as
``method_overrides`` (YAML) that the runner loads through ``tuned_file``.

    python -m sentinel_sim.tune --methods sentinel B0_pso ... --trials 30 --out configs/tuned.yaml
"""
from __future__ import annotations

import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')     # one BLAS thread per worker process (set before numpy)

import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
import yaml

TUNE_SCENARIOS = [
    {'n_uav': 8, 'channel': {'profile': 'urban_rubble'}},
    {'n_uav': 4, 'channel': {'profile': 'severe'}},
]
TUNE_BASE = {'duration_s': 600, 'map': {'kind': 'generate', 'size': 400, 'n_targets': 25,
                                        'base': True},
             'failures': {'kind': 'poisson', 'rate': 0.01}}
TUNE_SEEDS = [1000, 1001, 1002]

_EXPLORE = {'agent.w_goal': ('float', 0.0, 2.0), 'agent.sector_bias': ('float', 0.0, 3.0)}
SPACES = {
    'sentinel': {**_EXPLORE, 'agent.t_hold_s': ('float', 0.0, 180.0),
                 'spring.lam0': ('log', 0.01, 3.0), 'spring.beta': ('float', 0.0, 2.0)},
    # note: 'agent.explore_lam' is NOT part of the SENTINEL design (round-2 decision,
    # 2026-10-10); it was wrongly left in this space for the first A3 re-tune (corrected A4).
    'B0_pso': dict(_EXPLORE),
    'B1_v1': {**_EXPLORE, 'spring.lam0': ('log', 0.01, 3.0), 'spring.r_max': ('float', 30.0, 90.0)},
    'B2_commpso': {**_EXPLORE, 'spring.lam0': ('log', 0.01, 3.0),
                   'spring.eps0': ('log', 0.01, 0.3)},
    'B3_lambda2': {**_EXPLORE, 'agent_attrs.gain': ('log', 1.0, 200.0),
                   'agent_attrs.lam_min': ('log', 0.001, 0.2)},
    'B4_potential': {**_EXPLORE, 'agent_attrs.gain': ('log', 0.1, 20.0),
                     'agent_attrs.hyst': ('float', 1.0, 15.0)},
    'B6_lawnmower': {'agent_attrs.lane_factor': ('float', 1.0, 2.0)},
}


def _set(d, dotted, v):
    ks = dotted.split('.')
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = v


def _eval(cfg):
    from .sim import run
    return run(cfg)['summary']['report_value']


def tune(method, trials, pool, seed=0):
    space = SPACES[method]

    def objective(trial):
        over = {}
        for name, (kind, lo, hi) in space.items():
            v = (trial.suggest_float(name, lo, hi, log=True) if kind == 'log'
                 else trial.suggest_float(name, lo, hi))
            _set(over, name, v)
        cfgs = []
        for sc in TUNE_SCENARIOS:
            for s in TUNE_SEEDS:
                c = copy.deepcopy(TUNE_BASE)
                c.update(copy.deepcopy(sc))
                for k, v in over.items():
                    c.setdefault(k, {}).update(v) if isinstance(v, dict) else c.__setitem__(k, v)
                c.update(method=method, seed=s)
                cfgs.append(c)
        return float(np.mean(list(pool.map(_eval, cfgs))))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if not space:
        return {}, objective(study.ask())
    study.optimize(objective, n_trials=trials, n_jobs=3)
    best = {}
    for k, v in study.best_params.items():
        _set(best, k, float(v))
    return best, study.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=list(SPACES))
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--jobs', type=int, default=18)
    ap.add_argument('--out', default='configs/tuned.yaml')
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    tuned = {}
    if os.path.exists(a.out):
        tuned = yaml.safe_load(open(a.out)) or {}
    with ProcessPoolExecutor(a.jobs) as pool:
        for m in a.methods:
            best, val = tune(m, a.trials, pool)
            tuned[m] = best
            print(f'{m}: best report_value {val:.4f} params {best}', flush=True)
            with open(a.out, 'w') as f:
                yaml.safe_dump(tuned, f, sort_keys=True)


if __name__ == '__main__':
    main()
