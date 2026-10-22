#!/usr/bin/env python3
"""Consolidate all experiments into docs/RESULTS.md, LaTeX tables and figures.

Every number in the paper is meant to come from this script (no hand-typed
results). Statistics: mean and 95 % bootstrap CI over seeds; '*' marks a
difference from SENTINEL by paired Wilcoxon signed-rank test (common seeds),
Holm-corrected over the methods of a cell, p < 0.05.

    python make_results.py --results ~/sentinel_ws/results --docs ~/sentinel_ws/docs \
        --paper ~/sentinel_ws/src/SENTINEL/paper
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from sentinel_core.metrics import bootstrap_ci, paired_wilcoxon_holm

REF = 'sentinel'
NAMES = {'sentinel': 'SENTINEL', 'B0_pso': 'B0 PSO', 'B1_v1': 'B1 Earlier design',
         'B2_commpso': 'B2 Comm-PSO', 'B3_lambda2': 'B3 $\\lambda_2$-grad.',
         'B4_potential': 'B4 Potential', 'B5_mappo': 'B5 MAPPO', 'B6_lawnmower': 'B6 Lawnmower',
         'sentinel_strict': 'SENTINEL-strict', 'abl_distance_spring': 'distance spring',
         'abl_nn_set': 'NN set', 'abl_no_info_tension': 'no info. tension',
         'abl_no_projection': 'no projection', 'abl_no_sa': 'no SA', 'abl_no_fusion': 'no fusion',
         'abl_naive_fusion': 'naive-Bayes fusion', 'abl_no_sa_no_fusion': 'no SA, no fusion'}
ORDER = list(NAMES)
METRIC_LABEL = {'rep_within_60s': 'Reported $\\le$60\\,s', 'tdr_base': 'Reported (total)',
                'report_value': 'Report value', 'latency_p90_s': 'p90 latency [s]',
                'info_loss': 'Info. loss', 'assoc_loss': 'Assoc. loss', 'coverage': 'Coverage',
                'compute_ms_mean': 'Compute [ms]', 'false_entries_max': 'False entries',
                'link_violation_rate': 'Link viol.', 'frac_connected': 'Connected',
                'sa_triggers': 'SA triggers'}


def load(res, exp):
    f = os.path.join(res, exp, 'summary.parquet')
    if not os.path.exists(f):
        return None
    df = pd.read_parquet(f)
    if len(df) == 0:
        return None
    cfg = df.config.map(json.loads)
    df['fail'] = cfg.map(lambda c: c.get('failures', {}).get('kind', 'none'))
    df['det'] = cfg.map(lambda c: ','.join(f'{k}={v}' for k, v in c.get('detector', {}).items()) or 'default')
    df['t_hold'] = cfg.map(lambda c: c.get('agent', {}).get('t_hold_s'))
    df['thr'] = cfg.map(lambda c: '{low}/{high}'.format(**c.get('thresholds', {'low': 0.4, 'high': 0.7})))
    return df


def md_label(m):
    return (METRIC_LABEL.get(m, m).replace('$\\le$', '≤').replace('\\,', ' ')
            .replace('$', '').replace('\\', ''))


def tex(s):
    # escape text only; leave $...$ math segments untouched
    parts = str(s).split('$')
    return '$'.join(p if i % 2 else p.replace('_', '\\_').replace('%', '\\%')
                    for i, p in enumerate(parts))


def cell_stats(g, metric):
    """{method: (mean, lo, hi, star)} for one cell."""
    out = {}
    piv = g.pivot_table(index='cfg_seed', columns='cfg_method', values=metric)
    tests = {}
    if REF in piv:
        tests = paired_wilcoxon_holm(piv[REF].values,
                                     {c: piv[c].values for c in piv.columns if c != REF})
    for m in piv.columns:
        mean, lo, hi = bootstrap_ci(piv[m].values)
        star = m != REF and m in tests and tests[m]['p_holm'] < 0.05
        out[m] = (mean, lo, hi, star)
    return out


def fmt(v, digits=3, ci=True):
    mean, lo, hi, star = v
    if not np.isfinite(mean):
        return '–'
    s = f'{mean:.{digits}f}'
    if ci:
        s += f' [{lo:.{digits}f}, {hi:.{digits}f}]'
    return s + ('*' if star else '')


def md_table(df, key, metrics, title, digits=3):
    lines = [f'### {title}', '']
    for k, g in df.groupby(key):
        lines.append(f'**{key} = {k}**\n')
        stats = {m: cell_stats(g, m) for m in metrics}
        meths = [m for m in ORDER if m in g.cfg_method.unique()]
        lines.append('| method | ' + ' | '.join(md_label(m) for m in metrics) + ' |')
        lines.append('|' + '---|' * (len(metrics) + 1))
        for meth in meths:
            lines.append(f'| {NAMES.get(meth, meth).replace("$", "").replace(chr(92), "")} | '
                         + ' | '.join(fmt(stats[m].get(meth, (np.nan,) * 3 + (False,)), digits)
                                      for m in metrics) + ' |')
        lines.append('')
    return '\n'.join(lines)


def latex_table(df, key, metric, title, label, digits=3, wide=False, colnames=None):
    """Methods x cell values (mean, '*' significance) for one metric."""
    keys = list(df.groupby(key).groups)
    meths = [m for m in ORDER if m in df.cfg_method.unique()]
    stats = {k: cell_stats(g, metric) for k, g in df.groupby(key)}
    best = {k: max((v[0] for v in stats[k].values() if np.isfinite(v[0])), default=np.nan)
            for k in keys}
    rows = []
    for meth in meths:
        cells = []
        for k in keys:
            v = stats[k].get(meth)
            if v is None or not np.isfinite(v[0]):
                cells.append('--')
                continue
            s = f'{v[0]:.{digits}f}' + ('$^*$' if v[3] else '')
            cells.append(f'\\textbf{{{s}}}' if abs(v[0] - best[k]) < 1e-12 else s)
        rows.append(f'{NAMES.get(meth, meth)} & ' + ' & '.join(cells) + ' \\\\')
    head = ' & '.join(tex(colnames.get(k, k) if colnames else k) for k in keys)
    env = 'table*' if wide else 'table'
    return (f'% generated by make_results.py -- do not edit by hand\n'
            f'\\begin{{{env}}}[t]\\centering\\caption{{{tex(title)}}}\\label{{{label}}}\n'
            f'\\setlength{{\\tabcolsep}}{{4pt}}\\begin{{adjustbox}}{{max width=\\{"textwidth" if wide else "columnwidth"}}}'
            f'\\begin{{tabular}}{{l{"c" * len(keys)}}}\\toprule\n'
            f'Method & {head} \\\\\\midrule\n' + '\n'.join(rows)
            + f'\n\\bottomrule\\end{{tabular}}\\end{{adjustbox}}\n\\end{{{env}}}\n')


def fig_lines(df, x, metric, by, out, xlabel, ylabel, logx=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for meth in [m for m in ORDER if m in df.cfg_method.unique()]:
        g = df[df.cfg_method == meth]
        xs, ys, los, his = [], [], [], []
        for xv, gg in g.groupby(x):
            m, lo, hi = bootstrap_ci(gg[metric].values)
            xs.append(xv); ys.append(m); los.append(lo); his.append(hi)
        lw = 2.0 if meth == REF else 1.0
        ax.plot(xs, ys, marker='o', ms=3, lw=lw, label=NAMES.get(meth, meth))
        ax.fill_between(xs, los, his, alpha=0.15)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.expanduser('~/sentinel_ws/results'))
    ap.add_argument('--docs', default=os.path.expanduser('~/sentinel_ws/docs'))
    ap.add_argument('--paper', default=os.path.expanduser('~/sentinel_ws/src/SENTINEL/paper'))
    a = ap.parse_args()
    tdir, fdir = os.path.join(a.paper, 'tables'), os.path.join(a.paper, 'figures')
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)
    md = ['# SENTINEL v2 — results (generated by scripts/make_results.py)', '',
          'Mean [95 % bootstrap CI] over 30 seeds unless stated; `*` = differs from SENTINEL '
          '(paired Wilcoxon, Holm-corrected, p < 0.05). Primary metric: share of all victims '
          'reported to the responder within 60 s of first confirmation (`Reported ≤60 s`).', '']
    main_m = ['rep_within_60s', 'tdr_base', 'latency_p90_s', 'info_loss', 'coverage']
    spec = [
        ('e2_channel', 'cfg_profile', main_m, 'E2 — channel conditions (N=8, 400 m)'),
        ('e3_failures', ['cfg_profile', 'fail'], main_m, 'E3 — failure models (N=8, 400 m)'),
        ('e4_dynamic', 'cfg_n_moving', main_m, 'E4 — moving obstacles (urban, N=8)'),
        ('e1_scalability', ['cfg_map_size', 'cfg_n_uav'], ['rep_within_60s', 'tdr_base', 'compute_ms_mean'],
         'E1 — scalability'),
        ('e7_ablation', ['cfg_profile', 'cfg_n_uav'], ['rep_within_60s', 'tdr_base', 'latency_p90_s',
                                                        'link_violation_rate'], 'E7 — ablation'),
        ('e11_dense_diag', 'cfg_n_uav', ['rep_within_60s', 'tdr_base', 'false_entries_max'],
         'E11 — dense swarm (200 m, N=50)'),
        ('e6_detection', 'det', ['rep_within_60s', 'tdr_base', 'false_entries_max'],
         'E6 — detector uncertainty (urban, N=8)'),
        ('e6c_false_alarms', 'det', ['rep_within_60s', 'tdr_base', 'false_entries_max', 'sa_triggers'],
         'E6c — realistic false alarms (urban, N=8)'),
        ('e6b_thresholds', 'thr', ['rep_within_60s', 'tdr_base', 'false_entries_max', 'sa_triggers'],
         'E6b — SA / confirmation thresholds'),
        ('e9_v1_rerun', 'cfg_map_kind', ['tdr', 'nrr_final', 'frac_connected', 'info_loss', 'assoc_loss',
                                         'false_entries_max'], 'E9 — faithful re-run of the v1 scenario'),
        ('e8_fastsim', 'cfg_map_size', ['rep_within_60s', 'tdr_base', 'coverage', 'pdr'],
         'E8 — fast-simulator side of the SITL cross-check'),
    ]
    for exp, key, metrics, title in spec:
        df = load(a.results, exp)
        if df is None:
            md.append(f'### {title}\n\n_not available_\n'); continue
        if isinstance(key, list):
            df['cell'] = [' / '.join(map(str, r)) for r in df[key].itertuples(index=False)]
            key = 'cell'
        md.append(md_table(df, key, [m for m in metrics if m in df], title))
        if exp == 'e11_dense_diag':
            with open(os.path.join(tdir, f'{exp}_rep60.tex'), 'w') as f:
                f.write(latex_table(df, key, 'rep_within_60s',
                                    'Dense swarm (200 m, N=50): victims reported within 60 s.', f'tab:{exp}',
                                    colnames={'50': 'N=50'}))
    cap = (' Mean over 30 seeds; best per column in bold; $^*$: differs from SENTINEL '
           '(paired Wilcoxon, Holm-corrected, $p<0.05$).')
    e3 = load(a.results, 'e3_failures')
    if e3 is not None:
        for prof, g in e3.groupby('cfg_profile'):
            with open(os.path.join(tdir, f'e3_failures_{prof}_rep60.tex'), 'w') as f:
                f.write(latex_table(g, 'fail', 'rep_within_60s',
                                    f'Victims reported within 60 s under failure models ({prof.replace("_", "-")} channel, N=8, 400 m).' + cap,
                                    f'tab:e3_{prof}', wide=True,
                                    colnames={'none': 'none', 'single': 'single', 'simultaneous': '30\\% simult.',
                                              'cascading': 'cascading', 'poisson': 'Poisson', 'brownout': 'brownout'}))
    e1t = load(a.results, 'e1_scalability')
    if e1t is not None:
        for size, g in e1t.groupby('cfg_map_size'):
            with open(os.path.join(tdir, f'e1_scal_{int(size)}m_rep60.tex'), 'w') as f:
                f.write(latex_table(g, 'cfg_n_uav', 'rep_within_60s',
                                    f'Victims reported within 60 s vs swarm size N ({int(size)} m map, urban, Poisson failures).' + cap,
                                    f'tab:e1_{int(size)}', wide=False,
                                    colnames={n: f'N={n}' for n in g.cfg_n_uav.unique()}))
    e7t = load(a.results, 'e7_ablation')
    if e7t is not None:
        e7t['cell'] = [f'{p.replace("_rubble", "")}, N={n}' for p, n in zip(e7t.cfg_profile, e7t.cfg_n_uav)]
        with open(os.path.join(tdir, 'e7_ablation_rep60.tex'), 'w') as f:
            f.write(latex_table(e7t, 'cell', 'rep_within_60s',
                                'Ablation: victims reported within 60 s (400 m, Poisson failures).' + cap,
                                'tab:e7', wide=False))
    e2t, e4t, e6ct, e11t = (load(a.results, x) for x in ('e2_channel', 'e4_dynamic', 'e6c_false_alarms', 'e11_dense_diag'))
    if e2t is not None:
        with open(os.path.join(tdir, 'e2_channel_rep60.tex'), 'w') as f:
            f.write(latex_table(e2t, 'cfg_profile', 'rep_within_60s',
                                'Victims reported within 60 s per channel profile (N=8, 400 m, Poisson failures).' + cap,
                                'tab:e2', colnames={'urban_rubble': 'urban'}))
    if e4t is not None:
        with open(os.path.join(tdir, 'e4_dynamic_rep60.tex'), 'w') as f:
            f.write(latex_table(e4t, 'cfg_n_moving', 'rep_within_60s',
                                'Victims reported within 60 s with moving obstacles (urban, N=8).' + cap, 'tab:e4',
                                colnames={0: '0 moving', 5: '5 moving', 15: '15 moving'}))
    if e6ct is not None:
        e6ct['fpcell'] = [json.loads(c)['detector']['fp_mu'] for c in e6ct.config]
        e6ct['fpcell'] = [f'$\\mu_{{\\mathrm{{fp}}}}$={m}, $r$={json.loads(c)["detector"]["fp_rate"]}' for m, c in zip(e6ct.fpcell, e6ct.config)]
        with open(os.path.join(tdir, 'e6c_false_alarms_rep60.tex'), 'w') as f:
            f.write(latex_table(e6ct, 'fpcell', 'rep_within_60s',
                                'Victims reported within 60 s with hot-clutter false alarms (urban, N=8, $\\sigma$=0.15).' + cap,
                                'tab:e6c'))
    pf2 = os.path.join(a.results, 'perception', 'perception.csv')
    if os.path.exists(pf2):
        pp = pd.read_csv(pf2)
        lines = []
        for model, g in pp.groupby('model', sort=False):
            r0 = g.iloc[0]
            lat = {b: g[g.backend == b].total_ms_p50.iloc[0] for b in g.backend if 'total_ms_p50' in g}
            nm = 'YOLO26n' if '26' in model else 'YOLOv8n'
            lines.append(f'{nm} & {r0.params_M:.2f} & {r0.gflops:.1f} & {r0.mAP50:.3f} & {r0.mAP50_95:.3f} & '
                         f'{r0.precision:.3f} & {r0.recall:.3f} & {lat.get("torch", float("nan")):.2f} & '
                         f'{lat.get("engine", float("nan")):.2f} \\\\')
        with open(os.path.join(tdir, 'perception.tex'), 'w') as f:
            f.write('% generated by make_results.py -- do not edit by hand\n'
                    '\\begin{table*}[t]\\centering\\caption{Thermal detectors trained with the same protocol '
                    '(HIT-UAV, 100 epochs, AdamW, 640 px, seed 0). Accuracy: validation metrics recorded by the training run. '
                    'Latency: end-to-end p50 on an RTX 3060 (desktop), PyTorch FP32 and TensorRT FP16.}\\label{tab:perception}\n'
                    '\\begin{tabular}{lcccccccc}\\toprule\n'
                    'Model & Params [M] & GFLOPs & mAP50 & mAP50-95 & P & R & PyTorch [ms] & TensorRT [ms] \\\\\\midrule\n'
                    + '\n'.join(lines) + '\n\\bottomrule\\end{tabular}\n\\end{table*}\n')
    # E10 (single method, T_hold sweep): trade-off table + figure
    e10 = load(a.results, 'e10_thold')
    if e10 is not None:
        e10['th'] = e10.t_hold.map(lambda v: '∞' if v and v > 1e4 else f'{v:g}')
        lines = ['### E10 — delivery trigger T_hold (SENTINEL)', '',
                 '| regime | T_hold | Reported ≤60 s | Reported (total) | p90 latency [s] | Info. loss |',
                 '|---|---|---|---|---|---|']
        for (p, n), g in e10.groupby(['cfg_profile', 'cfg_n_uav']):
            for th, gg in sorted(g.groupby('t_hold'), key=lambda kv: kv[0]):
                cells = [bootstrap_ci(gg[m].values) for m in ['rep_within_60s', 'tdr_base',
                                                               'latency_p90_s', 'info_loss']]
                lines.append(f'| {p}, N={n} | {"∞" if th > 1e4 else f"{th:g}"} | '
                             + ' | '.join(f'{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]' for c in cells) + ' |')
        md.append('\n'.join(lines) + '\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for (p, n), g in e10.groupby(['cfg_profile', 'cfg_n_uav']):
            pts = g.groupby('t_hold')[['latency_p90_s', 'rep_within_60s']].mean().sort_index()
            ax.plot(pts.latency_p90_s, pts.rep_within_60s, marker='o', ms=3, label=f'{p}, N={n}')
        ax.set_xlabel('p90 report latency [s]'); ax.set_ylabel('reported within 60 s')
        ax.grid(alpha=0.3); ax.legend(fontsize=5)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, 'e10_thold_tradeoff.pdf')); plt.close(fig)
    e1 = load(a.results, 'e1_scalability')
    if e1 is not None:
        for size, g in e1.groupby('cfg_map_size'):
            fig_lines(g, 'cfg_n_uav', 'rep_within_60s', 'cfg_method',
                      os.path.join(fdir, f'e1_scal_rep60_{int(size)}m.pdf'), 'swarm size N',
                      'reported within 60 s', logx=True)
            fig_lines(g, 'cfg_n_uav', 'compute_ms_mean', 'cfg_method',
                      os.path.join(fdir, f'e1_compute_{int(size)}m.pdf'), 'swarm size N',
                      'per-agent compute [ms]', logx=True)
    # Perception benchmark (R1.5)
    pf = os.path.join(a.results, 'perception', 'perception.csv')
    if os.path.exists(pf):
        p = pd.read_csv(pf)
        cols = ['model', 'backend', 'params_M', 'gflops', 'mAP50', 'mAP50_95', 'precision', 'recall',
                'infer_ms_p50', 'total_ms_p50', 'total_ms_p95', 'fps_p50']
        md.append('### Perception — YOLO26n vs YOLOv8n (R1.5)\n')
        md.append(p[[c for c in cols if c in p]].round(3).to_markdown(index=False))
        md.append('\nAccuracy: ' + '; '.join(sorted(set(p.acc_source))) + '. Latency: RTX 3060 desktop, '
                  '640 px, 300 frames; TensorRT engines built locally from ONNX.\n')
    # E8 SITL vs fast simulator
    cf = os.path.join(a.results, 'e8_sitl', 'e8_comparison.csv')
    if os.path.exists(cf):
        c = pd.read_csv(cf)
        md.append('### E8 — SITL (PX4 + Gazebo + ROS 2) vs fast simulator, same worlds and seeds\n')
        md.append(c.to_markdown(index=False))
        md.append('\nMeans by method:\n')
        md.append(c.groupby('method').mean(numeric_only=True).round(3).to_markdown())
        md.append('\n3 worlds x 2 methods, 300 s, 3 UAVs, crash of UAV 1 at 150 s; sanity cross-check, '
                  'not statistically powered.\n')
    # Tuning sensitivity (Amendment A4): A3 configuration with explore_lam vs final
    rows = []
    for exp, key in [('e2_channel', 'cfg_profile'), ('e1_scalability', ['cfg_map_size', 'cfg_n_uav']),
                     ('e3_failures', ['cfg_profile', 'fail'])]:
        alt = os.path.join(a.results, exp, 'summary_A3explore.parquet')
        cur = load(a.results, exp)
        if not os.path.exists(alt) or cur is None:
            continue
        old = pd.read_parquet(alt)
        old['fail'] = old.config.map(lambda c: json.loads(c).get('failures', {}).get('kind', 'none'))
        for k, g in cur[cur.cfg_method == REF].groupby(key):
            go = old[old.cfg_method == REF]
            kk = k if isinstance(k, tuple) else (k,)
            for col, val in zip(key if isinstance(key, list) else [key], kk):
                go = go[go[col] == val]
            rows.append({'experiment': exp, 'cell': ' / '.join(map(str, kk)),
                         'final (A4)': round(g.rep_within_60s.mean(), 3),
                         'with explore_lam (A3)': round(go.rep_within_60s.mean(), 3) if len(go) else None,
                         'coverage final': round(g.coverage.mean(), 3),
                         'coverage A3': round(go.coverage.mean(), 3) if len(go) else None})
    if rows:
        md.append('### Tuning sensitivity (Amendment A4): reported within 60 s, SENTINEL\n')
        md.append(pd.DataFrame(rows).to_markdown(index=False))
        md.append('\nThe A3 configuration (soft exploration tether, explore_lam=0.155, lam0=2.39) helps dense '
                  'swarms but cuts severe-channel coverage; main results use the defined design (A4).\n')
    # MAPPO learning curve (B5 convergence evidence, R2.6)
    cfile = os.path.join(os.path.dirname(__file__), '..', 'src', 'sentinel_sim', 'configs',
                         'mappo_policy_long_curve.npy')
    if os.path.exists(cfile):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cv = np.load(cfile)
        k = 10
        sm = np.convolve(cv[:, 1], np.ones(k) / k, mode='valid')
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        ax.plot(cv[:, 0], cv[:, 1], lw=0.6, alpha=0.4, label='per iteration')
        ax.plot(cv[k - 1:, 0], sm, lw=1.5, label=f'{k}-iteration mean')
        ax.set_xlabel('PPO iteration (16 episodes each)'); ax.set_ylabel('report value (training)')
        ax.grid(alpha=0.3); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, 'mappo_curve.pdf')); plt.close(fig)
        md.append(f'### B5 MAPPO training\n\n{len(cv)} iterations x 16 episodes; mean training report value '
                  f'first 50 it: {cv[:50, 1].mean():.3f}, iterations 150-599: {cv[150:, 1].mean():.3f} '
                  f'(plateau), last 50: {cv[-50:, 1].mean():.3f}. See configs/mappo_selection.md.\n')
    with open(os.path.join(a.docs, 'RESULTS.md'), 'w') as f:
        f.write('\n'.join(md))
    print('wrote', os.path.join(a.docs, 'RESULTS.md'), 'and', tdir, fdir)


if __name__ == '__main__':
    main()
