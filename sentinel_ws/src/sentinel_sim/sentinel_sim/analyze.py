"""Statistics, tables and figures for an experiment folder (summary.parquet).

For every cell (combination of the varied configuration fields) and method:
mean and 95 % bootstrap CI of each metric, plus paired Wilcoxon signed-rank
tests of SENTINEL against every other method over common seeds, Holm-
corrected, with Cliff's delta. Outputs Markdown and LaTeX tables and a
reports-vs-latency Pareto figure per cell.

    python -m sentinel_sim.analyze results/e2 --ref sentinel --metrics tdr_base report_value
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from sentinel_core.metrics import bootstrap_ci, paired_wilcoxon_holm

DEFAULT_METRICS = ['tdr_base', 'report_value', 'rep_within_60s', 'latency_p90_s', 'info_loss', 'assoc_loss',
                   'coverage', 'frac_connected', 'compute_ms_mean']
CELL_KEYS = ['cfg_map_size', 'cfg_n_uav', 'cfg_profile', 'cfg_failures', 'cfg_n_moving',
             'cfg_det_sigma', 'cfg_det_fp_rate', 'cfg_det_cmax', 'cfg_map_kind',
             'cfg_thr_low', 'cfg_thr_high']
HIGHER_IS_BETTER = {'latency_p90_s': False, 'latency_mean_s': False, 'info_loss': False, 'assoc_loss': False,
                    'compute_ms_mean': False}


def cell_columns(df):
    return [c for c in CELL_KEYS if c in df and df[c].nunique() > 1]


def summarise(df, metrics, ref):
    cells = cell_columns(df)
    rows, tests = [], []
    groups = df.groupby(cells) if cells else [((), df)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        cell = dict(zip(cells, key))
        for m in metrics:
            for meth, gm in g.groupby('cfg_method'):
                mean, lo, hi = bootstrap_ci(gm[m].values)
                rows.append({**cell, 'metric': m, 'method': meth, 'mean': mean, 'lo': lo,
                             'hi': hi, 'n': int(gm[m].notna().sum())})
            if ref in set(g.cfg_method):
                piv = g.pivot_table(index='cfg_seed', columns='cfg_method', values=m)
                if ref in piv:
                    others = {c: piv[c].values for c in piv.columns if c != ref}
                    for meth, r in paired_wilcoxon_holm(piv[ref].values, others).items():
                        tests.append({**cell, 'metric': m, 'vs': meth, **r})
    return pd.DataFrame(rows), pd.DataFrame(tests), cells


def table(stats, tests, cells, metrics, ref):
    out = []
    for key, g in (stats.groupby(cells) if cells else [((), stats)]):
        key = key if isinstance(key, tuple) else (key,)
        out.append('### ' + ', '.join(f'{c[4:]}={v}' for c, v in zip(cells, key)) + '\n')
        wide = g.pivot_table(index='method', columns='metric', values='mean')
        lo = g.pivot_table(index='method', columns='metric', values='lo')
        hi = g.pivot_table(index='method', columns='metric', values='hi')
        lines = ['| method | ' + ' | '.join(metrics) + ' |', '|' + '---|' * (len(metrics) + 1)]
        for meth in wide.index:
            cells_txt = []
            for m in metrics:
                if m not in wide:
                    cells_txt.append('')
                    continue
                txt = f'{wide.at[meth, m]:.3f} [{lo.at[meth, m]:.3f}, {hi.at[meth, m]:.3f}]'
                if meth != ref and len(tests):
                    t = tests[(tests.metric == m) & (tests.vs == meth)]
                    for c, v in zip(cells, key):
                        t = t[t[c] == v]
                    if len(t) and t.p_holm.iloc[0] < 0.05:
                        txt += ' *'
                cells_txt.append(txt)
            lines.append(f'| {meth} | ' + ' | '.join(cells_txt) + ' |')
        out.append('\n'.join(lines) + '\n')
    out.append(f'`*` = differs from {ref} (paired Wilcoxon, Holm-corrected p < 0.05).\n')
    return '\n'.join(out)


def pareto_plot(df, cells, out_dir, x='latency_p90_s', y='tdr_base'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key, g in (df.groupby(cells) if cells else [((), df)]):
        key = key if isinstance(key, tuple) else (key,)
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for meth, gm in g.groupby('cfg_method'):
            ax.errorbar(gm[x].mean(), gm[y].mean(), xerr=gm[x].sem(), yerr=gm[y].sem(),
                        fmt='o', ms=4, capsize=2, label=meth)
        ax.set_xlabel('p90 report latency [s]')
        ax.set_ylabel('victims reported to responder')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5, ncol=2)
        name = '_'.join(f'{c[4:]}{v}' for c, v in zip(cells, key)) or 'all'
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'pareto_{name}.pdf'))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--ref', default='sentinel')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS)
    ap.add_argument('--no-plots', action='store_true')
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(a.folder, 'summary.parquet'))
    metrics = [m for m in a.metrics if m in df]
    stats, tests, cells = summarise(df, metrics, a.ref)
    stats.to_csv(os.path.join(a.folder, 'stats.csv'), index=False)
    tests.to_csv(os.path.join(a.folder, 'tests.csv'), index=False)
    md = table(stats, tests, cells, metrics, a.ref)
    with open(os.path.join(a.folder, 'tables.md'), 'w') as f:
        f.write(md)
    if not a.no_plots:
        pareto_plot(df, cells, a.folder)
    print(md)


if __name__ == '__main__':
    main()
