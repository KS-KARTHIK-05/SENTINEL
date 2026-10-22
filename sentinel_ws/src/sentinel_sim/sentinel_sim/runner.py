"""Batch experiment runner.

An experiment YAML has a ``base`` config (see sim.DEFAULTS), a ``grid`` whose
keys are dotted config paths with lists of values (cartesian product), a list
of ``methods`` and a number of ``seeds``. Every (cell, method, seed) run is
executed in a process pool; summaries go to ``<out>/summary.parquet`` and
time series to ``<out>/series.parquet``. Existing runs are skipped, so an
interrupted experiment resumes where it stopped.

    sentinel_run configs/e1_scalability.yaml --out results/e1 --jobs 18
"""
from __future__ import annotations

import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')     # one BLAS thread per worker process (set before numpy)

import argparse
import copy
import hashlib
import itertools
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml


def _set(d: dict, dotted: str, value) -> None:
    keys = dotted.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _deep_merge(d: dict, u: dict) -> dict:
    for k, v in (u or {}).items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_merge(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def _method_overrides(method: str, overrides: dict) -> dict:
    """Tuned values for a method; SENTINEL variants/ablations inherit SENTINEL's,
    except for the parameters the variant itself fixes."""
    from .methods import METHODS
    if method in overrides:
        return copy.deepcopy(overrides[method])
    if not (method.startswith('sentinel') or method.startswith('abl_')) \
            or 'sentinel' not in overrides:
        return {}
    o = copy.deepcopy(overrides['sentinel'])
    _, a_over, s_over = METHODS[method]
    for k in a_over:
        o.get('agent', {}).pop(k, None)
    for k in s_over:
        o.get('spring', {}).pop(k, None)
    return o


def expand(spec: dict):
    """Configs in precedence order: base < tuned method values < grid cell values."""
    base = spec.get('base', {})
    overrides = dict(spec.get('method_overrides', {}))
    if spec.get('tuned_file'):
        with open(spec['tuned_file']) as f:
            for m, o in (yaml.safe_load(f) or {}).items():
                overrides.setdefault(m, o)
    grid = spec.get('grid', {})
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
        for method in spec['methods']:
            cell = _deep_merge(copy.deepcopy(base), _method_overrides(method, overrides))
            for k, v in zip(keys, combo):
                _set(cell, k, copy.deepcopy(v))
            for seed in range(spec.get('seed0', 0), spec.get('seed0', 0) + spec['seeds']):
                cfg = copy.deepcopy(cell)
                cfg['method'] = method
                cfg['seed'] = seed
                yield cfg


def run_id(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _work(cfg: dict):
    # one thread per process: the pool provides the parallelism
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from .sim import run
    try:
        res = run(cfg)
    except Exception:  # noqa: BLE001 - report and continue with other runs
        return run_id(cfg), cfg, None, traceback.format_exc()
    return run_id(cfg), cfg, res, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--no-series', action='store_true')
    ap.add_argument('--skip-methods', nargs='*', default=[])
    ap.add_argument('--only-methods', nargs='*', default=[])
    a = ap.parse_args(argv)
    with open(a.spec) as f:
        spec = yaml.safe_load(f)
    os.makedirs(a.out, exist_ok=True)
    sum_path = os.path.join(a.out, 'summary.parquet')
    ser_path = os.path.join(a.out, 'series.parquet')
    done = set(pd.read_parquet(sum_path)['run_id']) if os.path.exists(sum_path) else set()
    todo = [c for c in expand(spec) if run_id(c) not in done
            and c['method'] not in a.skip_methods
            and (not a.only_methods or c['method'] in a.only_methods)]
    print(f'{len(todo)} runs to do ({len(done)} already done), {a.jobs} workers', flush=True)
    rows, series, errors = [], [], []

    def flush():
        if rows:
            df = pd.DataFrame(rows)
            if os.path.exists(sum_path):
                df = pd.concat([pd.read_parquet(sum_path), df], ignore_index=True)
            df.to_parquet(sum_path, index=False)
            rows.clear()
        if series:
            ds = pd.concat(series, ignore_index=True)
            if os.path.exists(ser_path):
                ds = pd.concat([pd.read_parquet(ser_path), ds], ignore_index=True)
            ds.to_parquet(ser_path, index=False)
            series.clear()

    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(_work, c) for c in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            rid, cfg, res, err = fut.result()
            if err:
                errors.append({'run_id': rid, 'cfg': json.dumps(cfg), 'error': err})
                print(f'[{k}/{len(todo)}] ERROR {cfg["method"]} seed {cfg["seed"]}', flush=True)
                continue
            rows.append({'run_id': rid, 'config': json.dumps(cfg), **res['summary']})
            if not a.no_series:
                s = pd.DataFrame(res['series'])
                s['run_id'] = rid
                series.append(s)
            if k % 20 == 0 or k == len(todo):
                flush()
                print(f'[{k}/{len(todo)}] done', flush=True)
    flush()
    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(a.out, 'errors.csv'), index=False)
        print(f'{len(errors)} runs failed; see errors.csv')


if __name__ == '__main__':
    main()
