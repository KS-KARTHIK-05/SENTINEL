#!/usr/bin/env python3
"""Generate every table, figure and in-text number of the ICIUS 2026 revision (IOS Press template),
plus the repository file docs/icius2026_extended_results.md with the material the 6-page paper drops.

Nothing is typed by hand: tables report the mean +/- half-width of the 95 % bootstrap CI over the
seeds, '*' = paired Wilcoxon signed-rank test vs SENTINEL, Holm-corrected, p < 0.05; all values are
read from the per-seed results in ~/sentinel_ws/results, the stored run configurations and the code.

    source ~/sentinel_venv/bin/activate
    python paper_icius_ios/scripts/make_ios.py
"""
from __future__ import annotations

import glob
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from sentinel_core import ledger
from sentinel_core.agent import AgentConfig
from sentinel_core.channel import PROFILES, reliable_range
from sentinel_core.detector_model import DetectorModel
from sentinel_core.metrics import bootstrap_ci, paired_wilcoxon_holm
from sentinel_sim import sim as SIMMOD

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.expanduser('~/sentinel_ws/results')
OUT = os.path.join(HERE, '..', 'generated')
FIG = os.path.join(HERE, '..', 'figures')
EXT = os.path.join(HERE, '..', '..', 'docs', 'icius2026_extended_results.md')
CONFIGS = os.path.join(HERE, '..', '..', 'sentinel_ws', 'src', 'sentinel_sim', 'configs')
TUNED = os.path.join(CONFIGS, 'tuned_v4.yaml')      # B0-B4/B6 re-tuned 29 Oct, the rest as tuned_v3
TUNED_OLD = os.path.join(CONFIGS, 'tuned_v3.yaml')
# overnight experiments 2026-10-28 (plan: docs/analysis_plan_icius2026.md): B7/B8 runs are merged into
# the experiments whose settings they repeat; the other folders are read on their own.
OVN = os.path.join(RES, 'overnight_2026-10-28')
# pre-submission check of the E5 fitted detector model (29 Oct, plan section 9): corrected false-alarm
# conversion, E5 fit, check against the real detector, C1 (robustness) and C2 (E6 fast side) re-run
E5R = os.path.join(RES, 'e5_recheck_2026-10-29')
E5OLD = os.path.join(OVN, 'superseded_fp_conversion')       # the pre-fix fit, C1 and C2 (archived)
# baseline re-run (29 Oct, plan section 9): the rows of B0-B4, B6 and their fusion variants (and E7's
# no-projection ablation) are replaced by rows re-run on the released code with re-tuned values
RERUN = os.path.join(RES, 'baseline_rerun_2026-10-29')
sys.path.insert(0, os.path.join(HERE, '..', '..', 'sentinel_ws', 'scripts', 'overnight'))
import make_rerun_configs as RC  # noqa: E402
RERUN_EXPS = {os.path.basename(e): RC.METHODS.get(e, RC.AFFECTED) for e in RC.SPECS}
OVN_MERGE = {'e3_failures': 'b7b8_failures', 'e1_scalability': 'b7b8_scalability', 'e2_channel': 'b7b8_channel',
             'e12_attribution': 'b7b8_attribution', 'e11b_dense_fusion': 'b7b8_dense_fusion'}
REF = 'sentinel'
BASE = ['B0_pso', 'B1_v1', 'B2_commpso', 'B3_lambda2', 'B4_potential', 'B5_mappo', 'B6_lawnmower', 'B7_wall',
        'B8_relay']
NAMES = {'sentinel': 'SENTINEL', 'B0_pso': 'B0 PSO', 'B1_v1': 'B1 first version',
         'B2_commpso': 'B2 comm.-aware PSO', 'B3_lambda2': 'B3 $\\lambda_2$ control',
         'B4_potential': 'B4 edge potentials', 'B5_mappo': 'B5 MAPPO', 'B6_lawnmower': 'B6 sweep',
         'B7_wall': 'B7 hard wall', 'B8_relay': 'B8 relay chain'}
SHORT = {m: m.split('_')[0] for m in BASE}
M = 'rep_within_60s'
macros: dict[str, str] = {}
md: list[str] = []


def mac(name: str, value) -> None:
    macros[name] = str(value)


def _strip(cfg: str, paths: set) -> str:
    c = json.loads(cfg)
    for path in paths:
        d = c
        for k in path[:-1]:
            d = d.get(k, {})
        d.pop(path[-1], None)
    return json.dumps(c, sort_keys=True)


def _leaves(d: dict, pre: tuple = ()) -> list:
    return [x for k, v in d.items()
            for x in (_leaves(v, pre + (k,)) if isinstance(v, dict) else [(pre + (k,), v)])]


def _get(cfg: str, path: tuple):
    d = json.loads(cfg)
    for k in path:
        d = d[k]
    return d


def rerun(df: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Replace the re-run methods' rows of an experiment by the baseline re-run's rows. Asserts that
    the re-run has exactly the stored rows' cells and seeds: each re-run config equals a stored one
    except for the method's tuned values, which are the old values in the stored and the new ones in
    the re-run config."""
    if exp not in RERUN_EXPS:
        return df
    methods = RERUN_EXPS[exp] & set(df.cfg_method)
    new = pd.read_parquet(os.path.join(RERUN, exp, 'summary.parquet'))
    assert set(new.cfg_method) == methods, (exp, set(new.cfg_method) ^ methods)
    old = df[df.cfg_method.isin(methods)]
    t_old, t_new = yaml.safe_load(open(TUNED_OLD)), yaml.safe_load(open(TUNED))

    def keys(d: pd.DataFrame, tuned: dict) -> list:
        out = []
        for cfg, m in zip(d.config, d.cfg_method):
            b = m[:-len('_fusion')] if m.endswith('_fusion') else m
            if b not in RC.BASELINES:
                out.append(_strip(cfg, set()))
                continue
            for path, v in _leaves(tuned[b]):
                assert _get(cfg, path) == v, (exp, m, path)
            out.append(_strip(cfg, {p for p, _ in _leaves(t_old[b]) + _leaves(t_new[b])}))
        return sorted(out)
    k_old, k_new = keys(old, t_old), keys(new, t_new)
    assert len(set(k_old)) == len(k_old) and k_old == k_new, exp
    return pd.concat([df[~df.cfg_method.isin(methods)], new], ignore_index=True)


def load(exp: str, orig: bool = False) -> pd.DataFrame:
    """An experiment's rows; unless ``orig``, with the baseline re-run's rows (see ``rerun``)."""
    paths = [os.path.join(RES, exp, 'summary.parquet')]
    if exp in OVN_MERGE:
        paths.append(os.path.join(OVN, OVN_MERGE[exp], 'summary.parquet'))
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    assert not df.run_id.duplicated().any(), exp
    if not orig:
        df = rerun(df, exp)
    cfg = df.config.map(json.loads)
    df['fail'] = cfg.map(lambda c: c.get('failures', {}).get('kind', 'none'))
    df['th'] = cfg.map(lambda c: c.get('agent', {}).get('t_hold_s'))
    return df


def load_ovn(exp: str, root: str = OVN, orig: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(root, exp, 'summary.parquet'))
    if not orig and root != E5OLD:
        df = rerun(df, exp)
    df['fail'] = df.config.map(json.loads).map(lambda c: c.get('failures', {}).get('kind', 'none'))
    return df


CLAIMS: dict[str, object] = {}       # outcome of every re-checked claim (printed, and in the extended file)


SEEDS = int(load('e2_channel', orig=True).cfg_seed.nunique())


def per_seed(df: pd.DataFrame, method: str, metric: str = M) -> np.ndarray:
    g = df[df.cfg_method == method].sort_values('cfg_seed')
    assert len(g) == SEEDS, (method, len(g))
    return g[metric].to_numpy(float)


def plain(s: str) -> str:
    """LaTeX label -> markdown text."""
    return (s.replace('$\\lambda_2$', 'λ₂').replace('$T_h{=}\\infty$', 'T_h = ∞').replace('$N{=}', 'N = ')
            .replace('$', '').replace('\\,', ' ').replace('{', '').replace('}', ''))


def fmt(vals: np.ndarray, star: bool, d: int = 2) -> tuple[str, str]:
    m, lo, hi = bootstrap_ci(vals)
    hw = (hi - lo) / 2
    return (f'{m:.{d}f}$\\pm${hw:.{d}f}' + ('$^*$' if star else ''),
            f'{m:.{d}f} ± {hw:.{d}f}' + ('*' if star else ''))


def build(cols: list[tuple[str, dict]], rows: list[str], names: dict, tags: dict | None = None):
    """Return LaTeX rows and markdown rows; stars vs REF within each column, best in bold."""
    stars, best = {}, {}
    for h, data in cols:
        res = paired_wilcoxon_holm(data[REF], {k: v for k, v in data.items() if k != REF})
        stars[h] = {k: r['p_holm'] < 0.05 for k, r in res.items()}
        best[h] = max(data, key=lambda k: np.mean(data[k]))
    tex, mdr = [], []
    for r in rows:
        tc, mc = [], []
        for h, data in cols:
            if r not in data:
                tc.append('--'); mc.append('–'); continue
            t, m_ = fmt(data[r], stars[h].get(r, False))
            if best[h] == r:
                t, m_ = f'\\textbf{{{t}}}', f'**{m_}**'
            if tags and (r, h) in tags:
                t += f'\\,({tags[(r, h)]})'; m_ += f' ({tags[(r, h)]})'
            tc.append(t); mc.append(m_)
        tex.append(f'{names[r]} & ' + ' & '.join(tc) + ' \\\\')
        mdr.append(f'| {plain(names[r])} | ' + ' | '.join(mc) + ' |')
    return tex, mdr


TOP: list[tuple[str, bool]] = []    # (cell, SENTINEL has the highest mean) over the main comparisons
WORSE: list[str] = []                # cells where a baseline in its original form is significantly ahead


def never_worse_check(cell: str, ref: np.ndarray, others: dict) -> None:
    """Record every original-form baseline significantly ahead of SENTINEL (pre-registered rule)."""
    r = paired_wilcoxon_holm(ref, others)
    for m, v in r.items():
        if v['p_holm'] < 0.05 and np.mean(others[m]) > np.mean(ref):
            WORSE.append(f'{cell}: {SHORT.get(m, m)} ({np.mean(others[m]):.3f} vs {np.mean(ref):.3f})')


def md_table(title: str, header: list[str], rows: list[str], note: str = '') -> None:
    md.append(f'### {title}\n')
    md.append('| ' + ' | '.join(plain(h) for h in header) + ' |')
    md.append('|' + '---|' * len(header))
    md.extend(rows)
    md.append('')
    if note:
        md.append(note + '\n')


def write(name: str, text: str) -> None:
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('% generated by paper_icius_ios/scripts/make_ios.py -- do not edit\n' + text)


def capstat() -> str:
    return (f'Mean $\\pm$ 95\\,\\% CI half-width over {SEEDS} seeds; best in bold; $^*$: differs from '
            'SENTINEL ($p<0.05$, paired Wilcoxon, Holm).')


MDSTAT = (f'Mean ± half-width of the 95 % bootstrap confidence interval over {SEEDS} seeds; best in bold; '
          '* = differs from SENTINEL (paired Wilcoxon signed-rank test, Holm-corrected, p < 0.05).')


def tabular(spec: str, header: str, rows: list[str], sep: str = '3pt') -> str:
    return (f'\\footnotesize\\setlength{{\\tabcolsep}}{{{sep}}}\\begin{{tabular}}{{{spec}}}\\toprule\n'
            + header + '\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}')


# ----------------------------------------------------------------------------- channel
def channel() -> None:
    order = [('open', 'open'), ('suburban', 'suburban'), ('urban_rubble', 'urban rubble'), ('severe', 'severe')]
    p0 = PROFILES['urban_rubble']
    tex, mdr = [], []
    for key, label in order:
        p = PROFILES[key]
        k = 'Rayleigh' if not np.isfinite(p.rician_k_db) else f'{p.rician_k_db:g}'
        r0, r1 = reliable_range(p, 0.05, 0), reliable_range(p, 0.05, 1)
        tex.append(f'{label} & {p.path_loss_exp:g} & {p.shadow_sigma_db:g} & {k} & {p.nlos_loss_db:g} & '
                   f'{r0:.0f} & {r1:.0f} \\\\')
        mdr.append(f'| {label} | {p.path_loss_exp:g} | {p.shadow_sigma_db:g} | {k} | {p.nlos_loss_db:g} | '
                   f'{r0:.0f} | {r1:.0f} |')
        if key == 'urban_rubble':
            mac('RurbanZero', f'{r0:.0f}'); mac('RurbanOne', f'{r1:.0f}')
        if key == 'severe':
            mac('RsevZero', f'{r0:.0f}'); mac('RsevOne', f'{r1:.0f}')
    budget = p0.tx_power_dbm + 2 * p0.antenna_gain_dbi - (p0.noise_dbm + p0.gamma_th_db)
    for k, v in [('Ptx', f'{p0.tx_power_dbm:g}'), ('Gant', f'{p0.antenna_gain_dbi:g}'),
                 ('Bw', f'{p0.bandwidth_hz / 1e6:g}'), ('NF', f'{p0.noise_figure_db:g}'),
                 ('Gth', f'{p0.gamma_th_db:g}'), ('Noise', f'{p0.noise_dbm:.0f}'), ('PLzero', f'{p0.pl0_db:.0f}'),
                 ('Budget', f'{budget:.0f}'), ('Freq', f'{p0.freq_hz / 1e9:g}'), ('Eps', '0.05')]:
        mac(k, v)
    head = ('Profile & $\\eta$ & $\\sigma_X$ [dB] & $K$ [dB] & $L_o$ [dB] & $R_{0.05}(0)$ [m] & $R_{0.05}(1)$ [m]')
    write('tab_channel.tex',
          '\\begin{table}[tb]\\caption{Channel profiles and reliable ranges at 5\\,\\% outage with no and one '
          'obstruction ($K$: line-of-sight Rician factor; $L_o$: loss per obstruction).}\\label{tab:channel}\\centering\n'
          + tabular('lcccccc', head, tex, '4pt') + '\\end{table}\n')
    md_table('Channel profiles and reliable ranges (paper Table 1)',
             ['Profile', 'η', 'σ_X [dB]', 'K [dB]', 'L_o [dB]', 'R_0.05(0) [m]', 'R_0.05(1) [m]'], mdr,
             f'Link budget: f = {macros["Freq"]} GHz, P_t = {macros["Ptx"]} dBm, G = {macros["Gant"]} dBi per side, '
             f'B = {macros["Bw"]} MHz, noise figure {macros["NF"]} dB (noise floor {macros["Noise"]} dBm), '
             f'γ_th = {macros["Gth"]} dB: {macros["Budget"]} dB of path loss before fading margins '
             f'(PL(1 m) = {macros["PLzero"]} dB).')


# ----------------------------------------------------------------------------- channel results (E2)
def e2() -> None:
    df = load('e2_channel')
    cols = []
    for prof, h in [('open', 'open'), ('suburban', 'suburban'), ('urban_rubble', 'urban rubble'), ('severe', 'severe')]:
        g = df[df.cfg_profile == prof]
        cols.append((h, {m: per_seed(g, m) for m in [REF] + BASE}))
    _, mdr = build(cols, [REF] + BASE, NAMES)
    md_table('Victims reported within 60 s per channel profile (400 m, N = 8, Poisson failures)',
             ['Method'] + [h for h, _ in cols], mdr, MDSTAT)
    sig_all = []
    for prof, key in [('open', 'Open'), ('suburban', 'Sub'), ('urban_rubble', 'Urb'), ('severe', 'Sev')]:
        g = df[df.cfg_profile == prof]
        s = per_seed(g, REF)
        bb = max(BASE, key=lambda m: per_seed(g, m).mean())
        r = paired_wilcoxon_holm(s, {m: per_seed(g, m) for m in BASE})
        mac(f'Ch{key}S', f'{s.mean():.3f}'); mac(f'Ch{key}B', f'{per_seed(g, bb).mean():.3f}')
        mac(f'Ch{key}Bname', SHORT[bb])
        sig_all.append(all(r[m]['p_holm'] < 0.05 for m in BASE))
        CLAIMS[f'channel {prof}: SENTINEL significantly above every baseline'] = sig_all[-1]
        CLAIMS[f'channel {prof}: best baseline'] = SHORT[bb]
        CLAIMS[f'channel {prof}: not significant vs'] = [SHORT[m] for m in BASE if r[m]['p_holm'] >= 0.05]
        never_worse_check(f'channel {prof}', s, {m: per_seed(g, m) for m in BASE})
        TOP.append((f'channel {prof}', s.mean() > per_seed(g, bb).mean()))
    # "leads in all four channel profiles, significantly against all except in the severe one"
    assert all(t for c, t in TOP if c.startswith('channel')) and sig_all == [True, True, True, False], sig_all


# ----------------------------------------------------------------------------- failures (E3)
def e3() -> None:
    full = load('e3_failures')
    order = [('none', 'none', 'none'), ('single', 'single', 'single crash'),
             ('simultaneous', 'simult.', '30 % simultaneous'), ('cascading', 'casc.', 'cascading'),
             ('poisson', 'Poisson', 'Poisson ×2'), ('brownout', 'brown.', 'brownout')]
    lab_ = {'none': 'no failures', 'single': 'a single crash', 'simultaneous': 'simultaneous crashes',
            'cascading': 'cascading crashes', 'poisson': 'Poisson crashes', 'brownout': 'brownout'}
    ns_urb = []
    for prof in ['urban_rubble', 'severe']:
        df = full[full.cfg_profile == prof]
        cols, gains = [], []
        for k, h, _ in order:
            g = df[df.fail == k]
            data = {m: per_seed(g, m) for m in [REF] + BASE}
            cols.append((h, data))
            gains.append(np.mean(data[REF]) / max(np.mean(data[m]) for m in BASE) - 1)
            never_worse_check(f'failures {prof} {k}', data[REF], {m: data[m] for m in BASE})
            r = paired_wilcoxon_holm(data[REF], {m: data[m] for m in BASE})
            CLAIMS[f'failures {prof} {k}: not significant vs'] = [SHORT[m] for m in BASE if r[m]['p_holm'] >= 0.05]
            TOP.append((f'failures {prof} {k}', gains[-1] > 0))
            if prof == 'urban_rubble':
                ns_urb += [f'{SHORT[m]} under {lab_[k]}' for m in BASE if r[m]['p_holm'] >= 0.05]
        tex, mdr = build(cols, [REF] + BASE, NAMES)
        md_table(f'Victims reported within 60 s under six failure settings ({prof.replace("_", " ")}, 400 m, N = 8)',
                 ['Method'] + [lab for *_, lab in order], mdr, MDSTAT)
        CLAIMS[f'failures {prof}: gain over best baseline [%]'] = [round(100 * x, 1) for x in gains]
        if prof == 'urban_rubble':
            mac('EthreeGainLo', f'{100 * min(gains):.0f}'); mac('EthreeGainHi', f'{100 * max(gains):.0f}')
            # "X-Y % more ... in every failure setting, significantly except against <one baseline> under <one>"
            assert min(gains) > 0 and len(ns_urb) == 1, (gains, ns_urb)
            mac('EthreeNotSig', ns_urb[0])
            for m, k in ((REF, 'S'), ('B7_wall', 'Bseven'), ('B8_relay', 'Beight')):
                means = [np.mean(d[m]) for _, d in cols]
                mac(f'EthreeRange{k}Lo', f'{min(means):.2f}'); mac(f'EthreeRange{k}Hi', f'{max(means):.2f}')
            head = 'Method & ' + ' & '.join(h for _, h, _ in order)
            write('tab_e3.tex',
                  '\\begin{table}[tb]\\caption{Share reported within 60\\,s under six failure settings (urban '
                  'rubble, 400\\,m, $N=8$): none; one crash; 30\\,\\% of the swarm at once (simult.); three '
                  'cascading crashes (casc.); Poisson crashes at twice the default rate; transmit-power brownout '
                  '(brown.). ' + capstat() + '}\\label{tab:e3}\\centering\n'
                  + tabular('lcccccc', head, tex, '2.2pt') + '\\end{table}\n')
    il = full.groupby(['cfg_method', 'cfg_profile', 'fail']).info_loss.mean()
    avg = full.groupby('cfg_method').info_loss.mean()
    mac('LossAvgMax', f'{100 * avg[[REF] + BASE].max():.1f}')
    wk = il.loc[[REF] + BASE].idxmax()
    assert wk[1] == 'severe' and wk[2] == 'poisson', wk
    mac('LossWorst', f'{100 * il.loc[[REF] + BASE].max():.1f}')
    mac('LossWorstS', f'{100 * il.loc[(REF, wk[1], wk[2])]:.1f}')


# ----------------------------------------------------------------------------- swarm size (E1)
def e1() -> None:
    df = load('e1_scalability')
    for size in [400, 200]:
        a = df[df.cfg_map_size == size]
        ns = sorted(a.cfg_n_uav.unique())
        cols = [(f'$N={n}$', {m: per_seed(a[a.cfg_n_uav == n], m) for m in [REF] + BASE}) for n in ns]
        tex, mdr = build(cols, [REF] + BASE, NAMES)
        for h, data in cols:
            never_worse_check(f'swarm {size} m {h}', data[REF], {m: data[m] for m in BASE})
        md_table(f'Victims reported within 60 s vs swarm size ({size} m, urban rubble, Poisson failures)',
                 ['Method'] + [h for h, _ in cols], mdr, MDSTAT)
        if size == 400:
            mac('EoneNmin', ns[0]); mac('EoneNmax', ns[-1])
            head = 'Method & ' + ' & '.join(h for h, _ in cols)
            write('tab_e1.tex',
                  '\\begin{table}[tb]\\caption{Share reported within 60\\,s vs.\\ swarm size '
                  f'({size}\\,m, urban, Poisson failures); statistics as in Table~\\ref{{tab:e3}}.'
                  '}\\label{tab:e1}\\centering\n'
                  + tabular('lccccc', head, tex, '3pt') + '\\end{table}\n')
            top4, sig4 = [], []
            for n in ns:
                g = a[a.cfg_n_uav == n]
                r = paired_wilcoxon_holm(per_seed(g, REF), {m: per_seed(g, m) for m in BASE})
                CLAIMS[f'swarm 400 m N={n}: not significant vs'] = [SHORT[m] for m in BASE
                                                                    if r[m]['p_holm'] >= 0.05]
                bb = max(BASE, key=lambda m: per_seed(g, m).mean())
                top4.append(per_seed(g, REF).mean() > per_seed(g, bb).mean())
                sig4.append(all(r[m]['p_holm'] < 0.05 for m in BASE))
                TOP.append((f'swarm 400 m N={n}', top4[-1]))
                if not top4[-1]:
                    mac('EoneFourNotTop', f'$N={n}$, {SHORT[bb]} {per_seed(g, bb).mean():.2f} vs.\\ '
                                          f'{per_seed(g, REF).mean():.2f}')
            # "highest mean from N=a, significantly above every baseline from N=b": both hold from a size on
            i, j = top4.index(True), sig4.index(True)
            assert all(top4[i:]) and all(sig4[j:]) and i == 1 and j > i, (top4, sig4)
            mac('EoneTopFrom', ns[i]); mac('EoneSigFrom', ns[j])
        else:
            sig = []
            for n in ns:
                g = a[a.cfg_n_uav == n]
                r = paired_wilcoxon_holm(per_seed(g, REF), {m: per_seed(g, m) for m in BASE})
                sig.append((n, all(r[m]['p_holm'] < 0.05 for m in BASE)))
            mac('EoneTwoHundredSigN', ', '.join(str(n) for n, s in sig if s))
            g = a[a.cfg_n_uav == ns[-1]]
            mac('EoneDenseS', f'{per_seed(g, REF).mean():.3f}')
            mac('EoneDenseMappo', f'{per_seed(g, "B5_mappo").mean():.3f}')
            bb = max(BASE, key=lambda m: per_seed(g, m).mean())
            mac('EoneDenseBest', f'{per_seed(g, bb).mean():.3f}'); mac('EoneDenseBestName', SHORT[bb])
            nottop = []
            for n in ns:
                gg = a[a.cfg_n_uav == n]
                top = max([REF] + BASE, key=lambda m: per_seed(gg, m).mean())
                CLAIMS[f'swarm 200 m N={n}: highest mean'] = SHORT.get(top, top)
                TOP.append((f'swarm 200 m N={n}', top == REF))
                if top != REF:
                    nottop.append(f'$N={n}$ ({SHORT[top]})')
                    mac('EoneTwoHundredNotTopS', f'{per_seed(gg, REF).mean():.2f}')
                    mac('EoneTwoHundredNotTopB', f'{per_seed(gg, top).mean():.2f}')
            mac('EoneTwoHundredNotTop', ', '.join(nottop) if nottop else 'none')
            mac('EoneTwoHundredNnotTop', len(nottop))
            # "a baseline has the highest mean at every size (names), none significantly" (WORSE checks the latter)
            assert len(nottop) == len(ns), nottop
            mac('EoneTwoHundredTopNames', ', '.join(sorted({CLAIMS[f'swarm 200 m N={n}: highest mean'] for n in ns})))
    comp = df[df.cfg_method == REF].groupby(['cfg_map_size', 'cfg_n_uav']).compute_ms_mean.mean()
    mac('CompLo', f'{comp.min():.0f}'); mac('CompHi', f'{comp.max():.0f}')


# ----------------------------------------------------------------------------- attribution (E7, E12, E11, E11b)
def attribution() -> None:
    e7, e12 = load('e7_ablation'), load('e12_attribution')
    e2_, e11, e11b, e1_ = load('e2_channel'), load('e11_dense_diag'), load('e11b_dense_fusion'), load('e1_scalability')

    def sel(df, prof=None, n=None, size=None):
        g = df
        if prof is not None:
            g = g[g.cfg_profile == prof]
        if n is not None:
            g = g[g.cfg_n_uav == n]
        if size is not None:
            g = g[g.cfg_map_size == size]
        return g

    cols, tags = [], {}
    for prof, h in [('urban_rubble', 'urban, $N{=}8$'), ('severe', 'severe, $N{=}8$')]:
        a7, a12, a2 = sel(e7, prof, 8), sel(e12, prof, 8), sel(e2_, prof)
        data = {REF: per_seed(a12, REF), 'strict': per_seed(a7, 'sentinel_strict'),
                'nofus': per_seed(a12, 'abl_no_fusion'), 'nodel': per_seed(a12, 'sentinel_th_inf'),
                'none3': per_seed(a12, 'abl_no_sa_no_fusion_no_delivery')}
        assert np.allclose(data[REF], per_seed(a7, REF)) and np.allclose(data[REF], per_seed(a2, REF))
        bb = max(BASE, key=lambda m: per_seed(a2, m).mean())
        bf = max(BASE, key=lambda m: per_seed(a12, m + '_fusion').mean())
        data['best'], data['bestf'] = per_seed(a2, bb), per_seed(a12, bf + '_fusion')
        tags[('best', h)], tags[('bestf', h)] = SHORT[bb], SHORT[bf]
        cols.append((h, data))
    d50 = sel(e11, n=50)
    dense = {REF: per_seed(d50, REF), 'nofus': per_seed(d50, 'abl_no_fusion')}
    g1 = sel(e1_, n=50, size=200)
    assert np.allclose(dense[REF], per_seed(g1, REF)) and np.allclose(dense[REF], per_seed(e11b, REF))
    bb = max(BASE, key=lambda m: per_seed(g1, m).mean())
    bf = max(BASE, key=lambda m: per_seed(e11b, m + '_fusion').mean())
    dense['best'], dense['bestf'] = per_seed(g1, bb), per_seed(e11b, bf + '_fusion')
    hd = 'dense, $N{=}50$'
    tags[('best', hd)], tags[('bestf', hd)] = SHORT[bb], SHORT[bf]
    cols.append((hd, dense))
    names = {REF: 'SENTINEL', 'nofus': 'without fusion', 'nodel': 'without delivery ($T_h{=}\\infty$)',
             'none3': 'without fusion, flank, delivery', 'strict': 'always connected',
             'best': 'best baseline', 'bestf': 'best baseline + fusion'}
    rows = [REF, 'nofus', 'nodel', 'none3', 'strict', 'best', 'bestf']
    tex, mdr = build(cols, rows, names, tags)
    tex = [t for t in tex if not t.startswith('always connected')]   # given in the text instead
    write('tab_attr.tex',
          '\\begin{table}[tb]\\caption{Share reported within 60\\,s with SENTINEL components removed, and for '
          'the best baseline (named) with and without SENTINEL\'s fusion rule. Urban, severe: '
          '400\\,m; dense: 200\\,m; statistics as in Table~\\ref{tab:e3}.}\\label{tab:attr}\\centering\n'
          + tabular('lccc', 'Variant & ' + ' & '.join(h for h, _ in cols), tex, '3pt') + '\\end{table}\n')
    md_table('Attribution (paper Table 4)', ['Variant'] + [h for h, _ in cols], mdr, MDSTAT)
    # full single-component ablation (E7) for the extended file
    cols7 = []
    for prof in ['severe', 'urban_rubble']:
        for n in [4, 8]:
            g = sel(e7, prof, n)
            cols7.append((f'{prof.replace("_rubble", "")}, N = {n}',
                          {m: per_seed(g, m) for m in g.cfg_method.unique()}))
    abl = {'sentinel': 'SENTINEL', 'sentinel_strict': 'always connected', 'abl_distance_spring': 'distance spring (R_max = 50 m)',
           'abl_nn_set': 'nearest-neighbour preserved set', 'abl_no_info_tension': 'no information-dependent tension',
           'abl_no_projection': 'no projection', 'abl_no_sa': 'no flank', 'abl_no_fusion': 'no fusion',
           'abl_naive_fusion': 'naive-Bayes fusion'}
    _, mdr = build(cols7, list(abl), abl)
    md_table('Single-component ablation (400 m, Poisson failures)', ['Variant'] + [h for h, _ in cols7], mdr, MDSTAT)
    # in-text numbers
    u12 = sel(e12, 'urban_rubble', 8)
    means = [per_seed(sel(e7, p, n), 'sentinel_strict').mean() for p in ['urban_rubble', 'severe'] for n in [4, 8]]
    mac('StrictLo', f'{min(means):.2f}'); mac('StrictHi', f'{max(means):.2f}')
    for k, m in [('Nofus', 'abl_no_fusion'), ('Nodel', 'sentinel_th_inf'),
                 ('Nonethree', 'abl_no_sa_no_fusion_no_delivery'), ('NofusNosa', 'abl_no_sa_no_fusion')]:
        mac('Urb' + k, f'{per_seed(u12, m).mean():.3f}')
    mac('UrbS', f'{per_seed(u12, REF).mean():.3f}')
    fus = {b: per_seed(u12, b + '_fusion').mean() for b in BASE}
    bfu = max(fus, key=fus.get)
    mac('UrbBestFus', f'{fus[bfu]:.3f}'); mac('UrbBestFusName', SHORT[bfu])
    mac('UrbBestOrig', f'{per_seed(sel(e2_, "urban_rubble"), bfu).mean():.3f}')
    for n in (4, 8):
        g = sel(e12, 'urban_rubble', n)
        r = paired_wilcoxon_holm(per_seed(g, REF), {b: per_seed(g, b + '_fusion') for b in BASE})
        CLAIMS[f'urban N={n}: baselines + fusion not significantly behind'] = [
            SHORT[b] for b in BASE
            if not (r[b]['p_holm'] < 0.05 and per_seed(g, b + '_fusion').mean() < per_seed(g, REF).mean())]
    # "rises ... (best), no longer significantly behind SENTINEL; the other baselines stay significantly behind"
    assert CLAIMS['urban N=8: baselines + fusion not significantly behind'] == [SHORT[bfu]]
    mac('NBase', len(BASE))
    mac('BaseWord', {7: 'seven', 8: 'eight', 9: 'nine'}[len(BASE)])
    dfus = {b: per_seed(e11b, b + '_fusion').mean() for b in BASE}
    r = paired_wilcoxon_holm(dense[REF], {b: per_seed(e11b, b + '_fusion') for b in BASE})
    above = [b for b in BASE if r[b]['p_holm'] < 0.05 and dfus[b] > dense[REF].mean()]
    CLAIMS['dense: baselines + fusion significantly above SENTINEL (Holm over all nine)'] = [SHORT[b] for b in above]
    # the text follows Table 4 (Holm over its rows): the best baseline with fusion significantly above SENTINEL
    rt = paired_wilcoxon_holm(dense[REF], {k: dense[k] for k in ('nofus', 'best', 'bestf')})
    assert rt['bestf']['p_holm'] < 0.05 and dense['bestf'].mean() > dense[REF].mean()
    CLAIMS['dense: best baseline + fusion significantly above SENTINEL (Table 4)'] = tags[('bestf', hd)]
    mac('DenseFusLo', f'{min(dfus.values()):.3f}'); mac('DenseFusHi', f'{max(dfus.values()):.3f}')
    mac('DenseSigAbove', tags[('bestf', hd)])
    mac('DenseS', f'{dense[REF].mean():.3f}'); mac('DenseNofus', f'{dense["nofus"].mean():.3f}')
    mac('DenseNaive', f'{per_seed(d50, "abl_naive_fusion").mean():.3f}')
    g = sel(e7, 'urban_rubble', 8)
    mac('DistSpring', f'{per_seed(g, "abl_distance_spring").mean():.3f}')
    rr = paired_wilcoxon_holm(per_seed(g, REF), {m: per_seed(g, m) for m in g.cfg_method.unique() if m != REF})
    assert rr['abl_distance_spring']['p_holm'] >= 0.05


# ----------------------------------------------------------------------------- link safety
def safety() -> None:
    tot = bad = 0.0
    for f in glob.glob(os.path.join(RES, 'e*', 'summary.parquet')):
        if os.path.basename(os.path.dirname(f)).endswith('_oldmetric'):
            continue
        d = pd.read_parquet(f)
        if 'preserved_link_steps' not in d:
            continue
        d = d[(d.cfg_method.str.startswith('sentinel') | d.cfg_method.str.startswith('abl_'))
              & ~d.cfg_method.isin(['sentinel_strict', 'abl_no_projection'])]
        bad += float((d.link_violation_rate.fillna(0) * d.preserved_link_steps).sum())
        tot += float(d.preserved_link_steps.sum())
    # overnight runs (plan section 4): SENTINEL-family runs with the default epsilon, without the E3 reference
    # (a duplicate of E7) and without the epsilon variants (violations counted against their own epsilon)
    for exp, root, keep in (
            ('sensitivity', OVN, lambda m: m.startswith(('sentinel_lam0', 'sentinel_beta', 'abl_distance'))),
            ('robust_fitted_v2', E5R, lambda m: m == REF)):
        d = load_ovn(exp, root)
        d = d[d.cfg_method.map(keep)]
        bad += float((d.link_violation_rate.fillna(0) * d.preserved_link_steps).sum())
        tot += float(d.preserved_link_steps.sum())
    mac('ViolAll', f'{bad:.0f}'); mac('StepsAll', f'{tot:,.0f}'.replace(',', '\\,'))
    mac('ViolRate', f'{bad / tot:.1e}'.replace('e-0', '\\times10^{-') + '}')
    e12 = load('e12_attribution')
    g = e12[e12.cfg_method.isin([REF, 'abl_no_projection'])].copy()
    g['bad'] = g.link_violation_rate.fillna(0) * g.preserved_link_steps
    s = g.groupby('cfg_method')[['bad', 'preserved_link_steps']].sum()
    runs = g[g.bad > 0].groupby('cfg_method').size()
    nruns = g.groupby('cfg_method').size()
    assert nruns[REF] == nruns['abl_no_projection']
    piv = g.pivot_table(index=['cfg_profile', 'cfg_n_uav', 'cfg_seed'], columns='cfg_method', values='bad')
    p = stats.wilcoxon(piv[REF], piv['abl_no_projection']).pvalue
    r_on = s.loc[REF, 'bad'] / s.loc[REF, 'preserved_link_steps']
    r_off = s.loc['abl_no_projection', 'bad'] / s.loc['abl_no_projection', 'preserved_link_steps']
    fmtn = lambda v: f'{v:,.0f}'.replace(',', '\\,')  # noqa: E731
    mac('ViolOn', f'{s.loc[REF, "bad"]:.0f}'); mac('StepsOn', fmtn(s.loc[REF, 'preserved_link_steps']))
    mac('ViolOff', f'{s.loc["abl_no_projection", "bad"]:.0f}')
    mac('StepsOff', fmtn(s.loc['abl_no_projection', 'preserved_link_steps']))
    mac('RunsOn', int(runs.get(REF, 0))); mac('RunsOff', int(runs.get('abl_no_projection', 0)))
    mac('RunsTotal', int(nruns[REF])); mac('SafetySettings', int(g.groupby(['cfg_profile', 'cfg_n_uav']).ngroups))
    mac('ViolFold', f'{r_off / r_on:.0f}')
    e = int(np.floor(np.log10(p))) + 1
    assert p < 10 ** e
    mac('ViolP', f'10^{{{e}}}')


# ----------------------------------------------------------------------------- delivery trigger (E10) + figure
def e10() -> None:
    df = load('e10_thold')
    md.append('### Delivery trigger: hold time T_h (SENTINEL, tuned parameters)\n')
    md.append('| regime | T_h [s] | reported ≤ 60 s | reported (total) | p90 latency [s] | information loss |')
    md.append('|---|---|---|---|---|---|')
    for (prof, n), g in df.groupby(['cfg_profile', 'cfg_n_uav']):
        for th, gg in sorted(g.groupby('th'), key=lambda kv: kv[0]):
            c = [bootstrap_ci(gg[m].values) for m in [M, 'tdr_base', 'latency_p90_s', 'info_loss']]
            md.append(f'| {prof.replace("_", " ")}, N = {n} | {"∞" if th > 1e4 else f"{th:g}"} | '
                      + ' | '.join(f'{x[0]:.3f} ± {(x[2] - x[1]) / 2:.3f}' for x in c) + ' |')
    md.append('')
    for prof, k in [('urban_rubble', 'Urb'), ('severe', 'Sev')]:
        g = df[(df.cfg_profile == prof) & (df.cfg_n_uav == 8)]
        piv = g.pivot_table(index='cfg_seed', columns='th', values=M)
        lat = g.groupby('th').latency_p90_s.mean()
        inf = piv.columns.max()
        best = max([c for c in piv.columns if c != inf], key=lambda c: piv[c].mean())
        r = paired_wilcoxon_holm(piv[inf].values, {f'{c:g}': piv[c].values for c in piv.columns if c != inf})
        assert r[f'{best:g}']['p_holm'] < 0.05
        mac(f'Th{k}Best', f'{best:g}'); mac(f'Th{k}BestV', f'{piv[best].mean():.3f}')
        mac(f'Th{k}Inf', f'{piv[inf].mean():.3f}')
        mac(f'Th{k}LatBest', f'{lat[best]:.0f}'); mac(f'Th{k}LatInf', f'{lat[inf]:.0f}')
    for prof in ['urban_rubble', 'severe']:
        g = df[(df.cfg_profile == prof) & (df.cfg_n_uav == 4)]
        piv = g.pivot_table(index='cfg_seed', columns='th', values=M)
        inf = piv.columns.max()
        r = paired_wilcoxon_holm(piv[inf].values, {f'{c:g}': piv[c].values for c in piv.columns if c != inf})
        assert not any(v['p_holm'] < 0.05 for v in r.values())
    vals = sorted(df.th.dropna().unique())
    mac('ThList', ', '.join('\\infty' if v > 1e4 else f'{v:g}' for v in vals))
    # figure: Type 1 fonts via LaTeX text rendering; distinct markers and line styles for grayscale
    import matplotlib
    matplotlib.use('pgf')          # text typeset by pdfLaTeX with Type 1 Times (mathptmx)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False, 'font.family': 'serif',
                         'font.size': 9, 'text.usetex': True,
                         'pgf.preamble': r'\usepackage[T1]{fontenc}\usepackage{mathptmx}'})
    styles = {('urban_rubble', 4): ('o', '-'), ('urban_rubble', 8): ('s', '--'),
              ('severe', 4): ('^', ':'), ('severe', 8): ('D', '-.')}
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for (prof, n), g in df.groupby(['cfg_profile', 'cfg_n_uav']):
        pts = g.groupby('th')[['latency_p90_s', M]].mean().sort_index()
        mk, ls = styles[(prof, n)]
        ax.plot(pts.latency_p90_s, pts[M], marker=mk, ls=ls, color='black', ms=4, lw=1,
                mfc='white' if n == 4 else 'black', label=f'{prof.replace("_", " ")}, $N={n}$')
        ax.annotate(r'$T_h{=}0$', (pts.latency_p90_s.iloc[0], pts[M].iloc[0]), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('90th-percentile report latency [s]')
    ax.set_ylabel('reported within 60\\,s')
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'thold_tradeoff.pdf'))
    plt.close(fig)


# ----------------------------------------------------------------------------- cross-checks, detector, setup
def e9_e8() -> None:
    e9 = load('e9_v1_rerun')
    v = {m: per_seed(e9, m, 'tdr') for m in [REF, 'B0_pso', 'B1_v1']}
    r = paired_wilcoxon_holm(v[REF], {m: v[m] for m in ['B0_pso', 'B1_v1']})
    assert r['B1_v1']['p_holm'] < 0.05 and r['B0_pso']['p_holm'] >= 0.05
    mac('NineS', f'{v[REF].mean():.2f}'); mac('NineBzero', f'{v["B0_pso"].mean():.2f}')
    mac('NineBone', f'{v["B1_v1"].mean():.2f}')
    # "the mesh stays connected throughout in k of n runs (one B0 run is split for x % of the mission)"
    g9 = e9[e9.cfg_method.isin([REF, 'B0_pso', 'B1_v1'])]
    dis = g9[g9.frac_connected < 1]
    assert len(dis) == 1 and dis.cfg_method.iloc[0] == 'B0_pso', dis[['cfg_method', 'cfg_seed', 'frac_connected']]
    mac('NineConnRuns', len(g9) - len(dis)); mac('NineRuns', len(g9))
    mac('NineDisPct', f'{100 * (1 - dis.frac_connected.iloc[0]):.0f}')
    CLAIMS['E9: runs not always connected'] = [
        f'{SHORT[m]} seed {s}: {f:.3f}' for m, s, f in zip(dis.cfg_method, dis.cfg_seed, dis.frac_connected)]
    assoc = e9.groupby('cfg_method').assoc_loss.mean()
    mac('NineAssocLo', f'{100 * assoc.min():.0f}'); mac('NineAssocHi', f'{100 * assoc.max():.0f}')
    # the PX4 numbers come from the re-run on the current code (sitl_e8r_e6); the 15 Oct E8 runs used a stale
    # install with naive-Bayes fusion (docs/analysis_plan_icius2026.md, section 7)


def perception() -> None:
    p = pd.read_csv(os.path.join(RES, 'perception', 'perception.csv'))
    for model, k in [('yolo26n_thermal_best', 'Yn'), ('yolov8n_thermal_best', 'Ve')]:
        g = p[p.model == model]
        assert len(g), model
        r = g.iloc[0]
        mac(f'{k}Map', f'{r.mAP50:.3f}'); mac(f'{k}MapB', f'{r.mAP50_95:.3f}')
        mac(f'{k}Trt', f'{g[g.backend == "engine"].total_ms_p50.iloc[0]:.2f}')
    y, v = p[p.model == 'yolo26n_thermal_best'].iloc[0], p[p.model == 'yolov8n_thermal_best'].iloc[0]
    mac('YnParamRed', f'{100 * (1 - y.params_M / v.params_M):.0f}')
    mac('YnFlopRed', f'{100 * (1 - y.gflops / v.gflops):.0f}')


def setup() -> None:
    n = sum(len(pd.read_parquet(f)) for f in glob.glob(os.path.join(RES, 'e*', 'summary.parquet'))
            if not os.path.basename(os.path.dirname(f)).endswith('_oldmetric'))
    n += sum(len(pd.read_parquet(f)) for f in glob.glob(os.path.join(OVN, '*', 'summary.parquet'))
             if not os.path.basename(os.path.dirname(f)).startswith('pilot'))
    # C1/C2 with the corrected false-alarm rate (the baseline re-run replaces rows one for one)
    n += sum(len(pd.read_parquet(os.path.join(E5R, e, 'summary.parquet')))
             for e in ('robust_fitted_v2', 'e6_fastsim_v2'))
    tv = yaml.safe_load(open(TUNED))
    mac('BsevenGoal', f'{tv["B7_wall"]["agent"]["w_goal"]:.2f}')
    mac('BeightRho', f'{tv["B8_relay"]["agent_attrs"]["rho"]:.2f}')
    mac('BeightSpacing', f'{tv["B8_relay"]["agent_attrs"]["spacing"]:.2f}')
    mac('Missions', f'{round(n, -3):,}'.replace(',', '\\,'))
    cfg = json.loads(load('e2_channel').config.iloc[0])
    mac('Victims', cfg['map']['n_targets']); mac('Duration', f'{cfg["duration_s"]:g}')
    mac('Area', f'{cfg["map"]["size"]:g}'); mac('Nuav', cfg['n_uav']); mac('FailRate', f'{cfg["failures"]["rate"]:g}')
    mac('Seeds', SEEDS); mac('SeedMax', int(load('e2_channel').cfg_seed.max()))
    c = AgentConfig()
    tuned = yaml.safe_load(open(TUNED))['sentinel']
    mac('Th', f'{tuned["agent"]["t_hold_s"]:.0f}')
    mac('Vmax', f'{c.v_max:g}'); mac('Alt', f'{c.altitude:g}'); mac('Foot', f'{c.footprint_r:g}')
    mac('Flank', f'{c.sa_offset_m:g}'); mac('PlanPeriod', f'{c.plan_period_s:g}'); mac('Horizon', f'{c.horizon_s:g}')
    mac('PsoP', c.pso.particles); mac('PsoI', c.pso.iterations)
    mac('PsoChi', f'{c.pso.chi:g}'); mac('PsoC', f'{c.pso.c1:g}')
    mac('Clo', f'{ledger.LOW_CONF:.2f}'); mac('Chi', f'{ledger.HIGH_CONF:.2f}')
    lg = inspect.signature(ledger.Ledger.__init__).parameters
    mac('Gate', f'{lg["gate_m"].default:g}'); mac('Sectors', lg['n_sectors'].default)
    mac('Theta', f'{SIMMOD.REPORT_VALUE_THETA_S:g}')
    mac('Dt', f'{SIMMOD.DEFAULTS["dt"]:g}')
    mac('LamZero', f'{tuned["spring"]["lam0"]:.2f}'); mac('Beta', f'{tuned["spring"]["beta"]:.2f}')
    mac('FlankAngle', f'{np.degrees(np.arctan(c.sa_offset_m / c.altitude)):.1f}')
    mac('FlankSlant', f'{100 * (np.hypot(c.altitude, c.sa_offset_m) / c.altitude - 1):.0f}')
    e9 = json.loads(load('e9_v1_rerun').config.iloc[0])
    assert e9['map']['kind'] == 'disaster_25'
    mac('NineArea', '25'); mac('NineN', e9['n_uav'])
    c8 = json.loads(load('e8_fastsim').config.iloc[0])
    mac('EightArea', f'{c8["map"]["size"]:g}'); mac('EightN', c8['n_uav'])
    mac('EightWorlds', load('e8_fastsim').cfg_seed.nunique())
    c11 = json.loads(load('e11_dense_diag').config.iloc[0])
    mac('DenseArea', f'{c11["map"]["size"]:g}'); mac('DenseN', c11['n_uav'])
    # parameter table for the extended file
    d = DetectorModel()
    md.append('### Default parameters\n')
    md.append('| group | values |\n|---|---|')
    md.append(f'| kinematics | Δt = {macros["Dt"]} s, v = {c.v_max:g} m/s, h = {c.altitude:g} m, camera footprint {c.footprint_r:g} m |')
    md.append(f'| planning | every {c.plan_period_s:g} s, horizon {c.horizon_s:g} s, PSO {c.pso.particles} particles × '
              f'{c.pso.iterations} iterations, χ = {c.pso.chi:g}, c1 = c2 = {c.pso.c1:g} |')
    md.append(f'| tuned | T_h = {tuned["agent"]["t_hold_s"]:.0f} s, λ0 = {tuned["spring"]["lam0"]:.2f}, '
              f'β = {tuned["spring"]["beta"]:.2f}, w_g = {tuned["agent"]["w_goal"]:.3f}, sector bias '
              f'{tuned["agent"]["sector_bias"]:.2f} |')
    md.append(f'| spring | ε = {c.r_hard_eps:g}, w_c = {c.w_cov:g}, w_d = {c.w_det:g}, w_t = {c.w_tr:g} |')
    md.append(f'| sensing | c_lo = {ledger.LOW_CONF:.2f}, c_hi = {ledger.HIGH_CONF:.2f}, flank ±{c.sa_offset_m:g} m, '
              f'gate {lg["gate_m"].default:g} m, {lg["n_sectors"].default} sectors |')
    md.append(f'| detector model | c_max = {d.c_max:g}, γ = {d.gamma:g}, r_ref = {d.r_ref:g} m, r_s = {d.r_scale:g} m, '
              f'σ_c = {d.sigma:g}, κ_o = {d.kappa:g}, false alarms {d.fp_rate:g} per look |')
    md.append(f'| objective | θ = {macros["Theta"]} s in J |\n')


# ----------------------------------------------------------------------------- overnight 2026-10-28
SENS = {'lam0': ('λ0', ['sentinel_lam0_0.3', 'sentinel_lam0_4.8']),
        'beta': ('β', ['sentinel_beta_0', 'sentinel_beta_4']),
        'eps': ('ε', ['sentinel_eps_0.01', 'sentinel_eps_0.2']),
        'rmax': ('distance spring R_max', ['abl_distance_spring_r30', 'abl_distance_spring_r50',
                                           'abl_distance_spring_r70'])}
SENS_LABEL = {'sentinel': 'tuned (λ0 = 1.20, β = 1.09, ε = 0.05)', 'sentinel_lam0_0.3': 'λ0 = 0.3',
              'sentinel_lam0_4.8': 'λ0 = 4.8', 'sentinel_beta_0': 'β = 0', 'sentinel_beta_4': 'β = 4',
              'sentinel_eps_0.01': 'ε = 0.01', 'sentinel_eps_0.2': 'ε = 0.20',
              'abl_distance_spring_r30': 'distance spring, R_max = 30 m',
              'abl_distance_spring_r50': 'distance spring, R_max = 50 m',
              'abl_distance_spring_r70': 'distance spring, R_max = 70 m'}


def sensitivity() -> None:
    """E3: one-at-a-time sensitivity of the spring parameters (Holm within each parameter)."""
    df = load_ovn('sensitivity')
    e7 = load('e7_ablation')
    profs = ['urban_rubble', 'severe']
    for prof in profs:     # the reference re-run equals the stored E7 runs (deterministic)
        assert np.allclose(per_seed(df[df.cfg_profile == prof], REF),
                           per_seed(e7[(e7.cfg_profile == prof) & (e7.cfg_n_uav == 8)], REF))
    pv = {}
    for key, (_, vs) in SENS.items():
        raw = {}
        for prof in profs:
            g = df[df.cfg_profile == prof]
            for v in vs:
                raw[(prof, v)] = per_seed(g, v)
        ref = {prof: per_seed(df[df.cfg_profile == prof], REF) for prof in profs}
        # one Holm family per parameter: its variants x both channels
        tests = {f'{prof}|{v}': (ref[prof], x) for (prof, v), x in raw.items()}
        pr = {k: (1.0 if np.allclose(a, b) else float(stats.wilcoxon(a, b).pvalue))
              for k, (a, b) in tests.items()}
        order = sorted(pr, key=pr.get)
        run = 0.0
        for i, k in enumerate(order):
            run = max(run, min(1.0, (len(order) - i) * pr[k]))
            pv[tuple(k.split('|'))] = run
    md.append('### Sensitivity of the spring parameters (400 m, N = 8, Poisson failures)\n')
    md.append('One parameter at a time around the tuned SENTINEL; all other tuned values unchanged. '
              'ε is the outage target of both the spring and the preserved-link radius; link violations are '
              'counted against the variant\'s own ε. * = on-time share differs from tuned SENTINEL '
              '(paired Wilcoxon, Holm within each parameter over its variants and both channels).\n')
    md.append('| variant | channel | reported ≤ 60 s | link violations / link-steps | p90 latency [s] |')
    md.append('|---|---|---|---|---|')
    shares, viol, sig = [], [], []
    for v in [REF] + [x for _, vs in SENS.values() for x in vs]:
        for prof in profs:
            g = df[(df.cfg_profile == prof) & (df.cfg_method == v)]
            mean, lo, hi = bootstrap_ci(per_seed(g, v))
            bad = float((g.link_violation_rate.fillna(0) * g.preserved_link_steps).sum())
            steps = float(g.preserved_link_steps.sum())
            star = '*' if v != REF and pv[(prof, v)] < 0.05 else ''
            md.append(f'| {SENS_LABEL[v]} | {prof.replace("_", " ")} | {mean:.3f} ± {(hi - lo) / 2:.3f}{star} | '
                      f'{bad:.0f} / {steps:,.0f} | {g.latency_p90_s.mean():.0f} |')
            if v != REF:
                shares.append(mean); viol.append(bad / steps if steps else 0.0)
                if star:
                    sig.append((SENS_LABEL[v], prof, mean))
    md.append('')
    refm = {prof: per_seed(df[df.cfg_profile == prof], REF).mean() for prof in profs}
    delta = max(abs(per_seed(df[df.cfg_profile == prof], v).mean() - refm[prof])
                for prof in profs for _, vs in SENS.values() for v in vs)
    mac('SensMaxDelta', f'{delta:.3f}')
    grid = {key: sorted(float(v.rsplit('_', 1)[1].lstrip('r')) for v in vs) for key, (_, vs) in SENS.items()}
    for key, name in (('lam0', 'Lam'), ('beta', 'Beta'), ('eps', 'Eps'), ('rmax', 'Rmax')):
        mac(f'Sens{name}Lo', f'{grid[key][0]:g}'); mac(f'Sens{name}Hi', f'{grid[key][-1]:g}')
    CLAIMS['sensitivity: largest change of the share vs tuned SENTINEL'] = round(delta, 3)
    mac('SensShareLo', f'{min(shares):.3f}'); mac('SensShareHi', f'{max(shares):.3f}')
    mac('SensRefUrb', f'{refm["urban_rubble"]:.3f}'); mac('SensRefSev', f'{refm["severe"]:.3f}')
    mac('SensViolHi', f'{max(viol):.1e}'.replace('e-0', '\\times10^{-') + '}' if max(viol) > 0 else '0')
    mac('SensNsig', len(sig)); mac('SensNvar', len(shares))
    CLAIMS['sensitivity: significant variants'] = sig
    CLAIMS['sensitivity: share range'] = (round(min(shares), 3), round(max(shares), 3))
    CLAIMS['sensitivity: max violation rate'] = max(viol)


def detector_e4_e5() -> None:
    """E4 detector on rendered 25-30 m frames; E5 fitted detector model and robustness re-run."""
    e4 = os.path.join(OVN, 'e4')
    cnt = json.load(open(os.path.join(e4, 'dataset_counts.json')))
    ev = {t: json.load(open(os.path.join(e4, f'eval_{t}_test.json'))) for t in ('hituav', 'finetuned')}
    ft = json.load(open(os.path.join(e4, 'finetune_args.json')))
    for t, k in (('hituav', 'H'), ('finetuned', 'F')):
        e = ev[t]
        mac(f'Rd{k}Map', f'{e["map50"]:.2f}'); mac(f'Rd{k}MapB', f'{e["map50_95"]:.2f}')
        for thr, tk in ((0.4, 'Lo'), (0.7, 'Hi')):
            r = e[f'recall@{thr}']
            mac(f'Rd{k}Rec{tk}Open', f'{r["open"]:.2f}'); mac(f'Rd{k}Rec{tk}Occ', f'{r["partly occluded"]:.2f}')
            mac(f'Rd{k}Fa{tk}', f'{e[f"fa_debris_per_frame@{thr}"]:.3f}')
    for sp in ('train', 'val', 'test'):
        mac(f'Rd{sp.capitalize()}Worlds', cnt[sp]['worlds']); mac(f'Rd{sp.capitalize()}Frames', cnt[sp]['frames'])
        mac(f'Rd{sp.capitalize()}Boxes', cnt[sp]['boxes'])
    mac('RdEpochs', ft['epochs'])
    md.append('### Detector at the operating altitude (rendered thermal frames, 25 and 30 m)\n')
    md.append('Rendered imagery from the Gazebo thermal camera, not real 25–30 m footage. Worlds are split by '
              f'seed: train {cnt["train"]["worlds"]} worlds / {cnt["train"]["frames"]} frames / '
              f'{cnt["train"]["boxes"]} boxes, val {cnt["val"]["worlds"]} / {cnt["val"]["frames"]} / '
              f'{cnt["val"]["boxes"]}, test {cnt["test"]["worlds"]} / {cnt["test"]["frames"]} / {cnt["test"]["boxes"]}. '
              'Recall at IoU ≥ 0.5; false alarms per frame on warm debris. Fine-tuning: the HIT-UAV '
              f'checkpoint with its own training arguments ({ft["optimizer"]}, lr0 = {ft["lr0"]}, batch '
              f'{ft["batch"]}, {ft["imgsz"]} px, seed {ft["seed"]}, {ft["epochs"]} epochs, best by val fitness).\n')
    md.append('| model (test worlds) | mAP50 | mAP50-95 | recall ≥ 0.40 open / partly occluded | '
              'recall ≥ 0.70 open / partly occluded | false alarms per frame ≥ 0.40 / ≥ 0.70 |')
    md.append('|---|---|---|---|---|---|')
    for t, name in (('hituav', 'YOLO26n, HIT-UAV only'), ('finetuned', 'YOLO26n, fine-tuned on rendered 25–30 m frames')):
        e = ev[t]
        md.append(f'| {name} | {e["map50"]:.3f} | {e["map50_95"]:.3f} | '
                  f'{e["recall@0.4"]["open"]:.2f} / {e["recall@0.4"]["partly occluded"]:.2f} | '
                  f'{e["recall@0.7"]["open"]:.2f} / {e["recall@0.7"]["partly occluded"]:.2f} | '
                  f'{e["fa_debris_per_frame@0.4"]:.2f} / {e["fa_debris_per_frame@0.7"]:.2f} |')
    md.append(f'\nGround-truth victims in the test frames: {ev["finetuned"]["gt"]["open"]} open, '
              f'{ev["finetuned"]["gt"]["partly occluded"]} partly occluded.\n')
    # E5 fit (re-run 29 Oct with the corrected false-alarm conversion; victim parameters unchanged)
    fit = json.load(open(os.path.join(E5R, 'e5', 'fit.json')))
    fit0 = json.load(open(os.path.join(E5OLD, 'e5', 'fit.json')))
    f, h = fit['fitted'], fit['hand_set']
    assert all(f[k] == fit0['fitted'][k] for k in ('c_max', 'gamma', 'r_ref', 'r_scale', 'sigma', 'kappa'))
    rb = load_ovn('robust_fitted_v2', E5R)
    six_fast = load_ovn('e6_fastsim_v2', E5R)
    rate = {}
    for name, d in (('400', rb), ('100', six_fast)):
        v = d.cfg_det_fp_rate.unique()
        assert len(v) == 1
        rate[name] = float(v[0])
    md.append('### In-loop detector model fitted to detector outputs (E5)\n')
    md.append(f'Fitted by censored maximum likelihood to the fine-tuned detector on the val worlds '
              f'({fit["val"]["victim_views"]} victim views in {fit["val"]["frames"]} frames, slant range '
              f'{fit["val"]["slant_range_m"][0]:.0f}–{fit["val"]["slant_range_m"][1]:.0f} m). Parameters at a bound '
              f'(not identifiable in that range): {", ".join(fit["at_bound"]) or "none"}. False alarms: '
              f'{fit["val"]["false_alarms"]} in {fit["val"]["frames"]} val frames, '
              f'{fit["val"]["fa_per_debris_in_view"]:.4f} reported per warm-debris item in view (every debris item '
              f'of the world that projects into the frame, visible or hidden in rubble, as the simulator counts '
              f'them). The simulator reports a false-alarm candidate only if its confidence N(fp_mu, fp_sigma) '
              f'reaches 0.25 (probability {fit["val"]["p_reported_fa"]:.2f}), so the candidate rate is '
              f'{f["fp_per_debris_item"]:.4f} per look and debris item in view, and fp_rate (candidates per look, '
              f'spread over all debris items of the world) is {rate["400"]:.2f} in the 400 m worlds (480 items) '
              f'and {rate["100"]:.2f} in the 100 m worlds (30 items).\n')
    md.append('| parameter | hand-set | fitted |\n|---|---|---|')
    for k in ('c_max', 'gamma', 'r_ref', 'r_scale', 'sigma', 'kappa'):
        md.append(f'| {k} | {h[k]:.3g} | {f[k]:.3g} |')
    md.append(f'| fp_rate, 400 m worlds | {h["fp_rate"]:.3g} | {rate["400"]:.3g} |')
    md.append(f'| fp_rate, 100 m worlds | {h["fp_rate"]:.3g} | {rate["100"]:.3g} |')
    for k in ('fp_mu', 'fp_sigma'):
        md.append(f'| {k} | {h[k]:.3g} | {f[k]:.3g} |')
    md.append(f'\nTest log-likelihood: fitted {fit["test"]["loglik_fitted"]:.1f}, hand-set '
              f'{fit["test"]["loglik_hand_set"]:.1f} ({fit["test"]["victim_views"]} victim views).\n')
    md.append('| class | slant range [m] | n | observed / fitted / hand-set P(c ≥ 0.25) | P(c ≥ 0.40) | P(c ≥ 0.70) |')
    md.append('|---|---|---|---|---|---|')
    for row in fit['test']['calibration']:
        md.append(f'| {row["class"]} | {row["range_m"]} | {row["n"]} | ' + ' | '.join(
            f'{row[f"obs@{t}"]:.2f} / {row[f"fit@{t}"]:.2f} / {row[f"hand@{t}"]:.2f}' for t in (0.25, 0.4, 0.7)) + ' |')
    md.append('')
    # check: the simulator's DetectorModel.look on the same frames vs the real detector
    ca = json.load(open(os.path.join(E5R, 'check_after', 'check.json')))
    cb = json.load(open(os.path.join(E5R, 'check_before', 'check.json')))
    md.append('**Check against the real detector (same frames).** The simulator\'s `DetectorModel.look` run on the '
              'E4 frames (same camera poses, victims and debris; 200 draws), next to the fine-tuned YOLO26n on '
              'those frames (`scripts/overnight/e5_check.py`). Recall: share of in-image, non-buried victim views '
              'reported at or above the threshold (open / partly occluded); confidence: mean over reported views.\n')
    md.append('| frames | detector | false alarms per frame ≥ 0.25 / ≥ 0.40 | recall ≥ 0.40 | recall ≥ 0.70 | '
              'mean confidence |\n|---|---|---|---|---|---|')

    def crow(split, name, x):
        oc = ('open', 'partly occluded')
        fa = ' / '.join(f'{v:.4f}' if v == 0 or v >= 1e-3 else f'{v:.1e}'
                        for v in (x['fa_per_frame@0.25'], x['fa_per_frame@0.4']))
        return (f'| {split} | {name} | {fa} | '
                + ' | '.join(' / '.join(f'{x[k][c]:.2f}' for c in oc)
                             for k in ('recall@0.4', 'recall@0.7', 'mean_conf_reported')) + ' |')
    for split in ('val', 'test'):
        r = ca[split]['real']
        md.append(crow(split, f'YOLO26n ({r["fa_count@0.25"]} false alarms)', r))
        md.append(crow(split, 'fitted model, 400 m worlds', ca[split]['sim']['detector_fitted@480']))
        md.append(crow(split, 'fitted model, 100 m worlds', ca[split]['sim']['detector_fitted@30']))
        md.append(crow(split, 'first conversion, 400 m worlds (superseded)', cb[split]['sim']['detector_fitted@480']))
        md.append(crow(split, 'first conversion, 100 m worlds (superseded)', cb[split]['sim']['detector_fitted@30']))
        md.append(crow(split, 'hand-set model, 400 m worlds', cb[split]['sim']['hand@480']))
    rv, rt = ca['val']['real'], ca['test']['real']
    k_v, k_t = rv['fa_count@0.25'], rt['fa_count@0.25']
    p_vt = float(stats.binomtest(k_t, k_v + k_t, rt['frames'] / (rv['frames'] + rt['frames'])).pvalue)
    md.append(f'\nThe fit uses the val frames ({k_v} false alarms); the test frames have {k_t} '
              f'(exact conditional test of equal rates p = {p_vt:.2f}), so the model\'s false-alarm rate is about '
              f'{ca["test"]["sim"]["detector_fitted@480"]["fa_per_frame@0.25"] / rt["fa_per_frame@0.25"]:.1f}× the '
              f'test frames\' rate. Buried victims (no rendered signature): YOLO26n reported none of '
              f'{rv["buried"]["views"] + rt["buried"]["views"]} in-image views; the fitted model reports '
              f'{100 * ca["test"]["sim"]["detector_fitted@480"]["buried"]["reported_share"]:.0f} % of them at '
              f'confidence ≥ 0.25 (mean 0 plus noise σ = {f["sigma"]:.2f}), almost all below the 0.40 band.\n')
    sa = ca['test']['sim']['detector_fitted@480']
    for sp, x in (('Val', rv), ('Test', rt)):
        mac(f'Chk{sp}FaReal', f'{x["fa_per_frame@0.25"]:.3f}'); mac(f'Chk{sp}FaN', x['fa_count@0.25'])
        for wd, tag in ((480, 'C'), (30, 'E')):
            mac(f'Chk{sp}FaSim{tag}', f'{ca[sp.lower()]["sim"][f"detector_fitted@{wd}"]["fa_per_frame@0.25"]:.3f}')
            mac(f'Chk{sp}FaOld{tag}', f'{cb[sp.lower()]["sim"][f"detector_fitted@{wd}"]["fa_per_frame@0.25"]:.3f}')
    for k, tk in (('recall@0.4', 'Lo'), ('recall@0.7', 'Hi'), ('mean_conf_reported', 'Conf')):
        for c, ck in (('open', 'Open'), ('partly occluded', 'Occ')):
            mac(f'ChkTest{tk}{ck}Real', f'{rt[k][c]:.2f}'); mac(f'ChkTest{tk}{ck}Sim', f'{sa[k][c]:.2f}')
    mac('ChkValTestP', f'{p_vt:.2f}')
    mac('FitFpRateC', f'{rate["400"]:.1f}'); mac('FitFpRateE', f'{rate["100"]:.2f}')
    mac('FitFpRateOld', f'{fit0["fitted"]["fp_rate"]:.1f}'); mac('FitPrep', f'{fit["val"]["p_reported_fa"]:.2f}')
    mac('FitFaDebris', f'{fit["val"]["fa_per_debris_in_view"]:.3f}')
    mac('FitFaDebrisOld', f'{fit0["val"]["fa_per_debris_in_view"]:.3f}')
    mac('FitFaRatio', f'{rate["400"] / h["fp_rate"]:,.0f}'.replace(',', '\\,'))
    # first pass (superseded: stale rendered frames in worlds 107, 110-115, see the plan's section 9)
    fp = os.path.join(OVN, 'firstpass_stale_frames')
    fe = json.load(open(os.path.join(fp, 'e4', 'eval_finetuned_test.json')))
    ff = json.load(open(os.path.join(fp, 'e5', 'fit.json')))
    md.append('First pass, superseded: 91 frames of worlds 107 and 110–115 showed an earlier viewpoint than their '
              'recorded pose (frames rendered before the pose change took effect); those worlds were re-rendered '
              'with a per-frame pose check and E4/E5 re-run. First-pass values: fine-tuned test mAP50 '
              f'{fe["map50"]:.3f}, recall ≥ 0.40 {fe["recall@0.4"]["open"]:.2f} / '
              f'{fe["recall@0.4"]["partly occluded"]:.2f}; fit c_max {ff["fitted"]["c_max"]:.2f}, gamma '
              f'{ff["fitted"]["gamma"]:.2f}, sigma {ff["fitted"]["sigma"]:.2f}, kappa {ff["fitted"]["kappa"]:.2f}, '
              f'fp_rate {ff["fitted"]["fp_rate"]:.2f} (at a bound: {", ".join(ff["at_bound"]) or "none"}). '
              'Archived in results/overnight_2026-10-28/firstpass_stale_frames/.\n')
    mac('FitLLf', f'{fit["test"]["loglik_fitted"]:.0f}'); mac('FitLLh', f'{fit["test"]["loglik_hand_set"]:.0f}')
    mac('FitCmax', f'{f["c_max"]:.2f}'); mac('FitSigma', f'{f["sigma"]:.2f}')
    # E5 robustness (Table 2 urban settings, fitted detector)
    rob = _robust(rb)
    _, mdr = build(rob['cols'], [REF] + BASE, NAMES)
    md_table('Robustness: failure settings with the detector model fitted to detector outputs (urban rubble, '
             '400 m, N = 8, tuned values unchanged)', ['Method'] + [h_ for h_, _ in rob['cols']], mdr, MDSTAT)
    gains, bests, behind, allsig = rob['gains'], rob['bests'], rob['behind'], rob['allsig']
    sgn = lambda x: f'{100 * x:+.1f}'.replace('-', '$-$')  # noqa: E731
    mac('RobDiffLo', sgn(min(gains))); mac('RobDiffHi', sgn(max(gains)))
    smeans = [np.mean(d[REF]) for _, d in rob['cols']]
    mac('RobSLo', f'{min(smeans):.2f}'); mac('RobSHi', f'{max(smeans):.2f}')
    mac('RobNsig', sum(allsig)); mac('RobNset', len(rob['cols']))
    ntop = sum(g > 0 for g in gains)
    mac('RobNtop', ntop)
    # the paper says "highest mean in k of the six settings, k < 6, differences to the best baseline from
    # lo to hi, none significant": SENTINEL behind somewhere, no significant win over all baselines in any
    # setting and no significant loss to any baseline
    assert 0 < ntop < len(gains) and sum(allsig) == 0 and not rob['losses'], (gains, allsig, rob['losses'])
    mac('RobBestName', '/'.join(sorted({SHORT[b] for b in bests})))
    mac('RobTopLoss', ', '.join(f'{h_} ({SHORT[b]})'.replace(' %', '\\,\\%') for (h_, _), g, b in
                                zip(rob['cols'], gains, bests) if g <= 0))
    never = [SHORT[m] for m in BASE if behind[m] == 0]
    mac('RobNeverBehind', ', '.join(never) if never else 'none')
    always = [SHORT[m] for m in BASE if behind[m] == len(gains)]
    mac('RobAlwaysBehind', ', '.join(always) if always else 'none')
    CLAIMS['robustness: best baseline per setting'] = [SHORT[b] for b in bests]
    CLAIMS['robustness: settings in which each baseline is significantly behind'] = {SHORT[m]: v for m, v in behind.items()}
    CLAIMS['robustness: gain range [%]'] = [round(100 * x, 1) for x in gains]
    CLAIMS['robustness: settings with the highest SENTINEL mean'] = f'{ntop} of {len(gains)}'
    CLAIMS['robustness: settings with SENTINEL significantly above all'] = sum(allsig)
    CLAIMS['robustness: significant losses of SENTINEL'] = rob['losses']
    # superseded robustness and E6 fast-side runs (first false-alarm conversion)
    old = _robust(load_ovn('robust_fitted', E5OLD))
    os_ = [np.mean(d[REF]) for _, d in old['cols']]
    fs0 = load_ovn('e6_fastsim', E5OLD).pivot(index='cfg_seed', columns='cfg_method', values='tdr_base')
    mac('RobOldGainLo', f'{100 * min(old["gains"]):.0f}'); mac('RobOldGainHi', f'{100 * max(old["gains"]):.0f}')
    mac('RobOldSLo', f'{min(os_):.2f}'); mac('RobOldSHi', f'{max(os_):.2f}')
    mac('SixFastOldS', f'{fs0[REF].mean():.2f}'); mac('SixFastOldB', f'{fs0["B0_pso"].mean():.2f}')
    md.append(f'Superseded (first false-alarm conversion, fp_rate {fit0["fitted"]["fp_rate"]:.1f} in both world '
              'sizes; plan section 9, 29 Oct): with it, SENTINEL\'s lead over the best baseline was '
              f'{100 * min(old["gains"]):.1f}–{100 * max(old["gains"]):.1f} % (best baseline per setting: '
              f'{", ".join(SHORT[b] for b in old["bests"])}; settings with SENTINEL significantly above all: '
              f'{sum(old["allsig"])}; significant losses: {", ".join(old["losses"]) or "none"}), SENTINEL '
              f'{min(os_):.2f}–{max(os_):.2f}; E6 fast side SENTINEL {fs0[REF].mean():.2f}, '
              f'B0 {fs0["B0_pso"].mean():.2f}. Archived in '
              'results/overnight_2026-10-28/superseded_fp_conversion/.\n')


def _robust(rb: pd.DataFrame) -> dict:
    """Per failure setting: SENTINEL and B0-B8, gain over the best baseline, Holm tests."""
    order = [('none', 'none'), ('single', 'single crash'), ('simultaneous', '30 % simultaneous'),
             ('cascading', 'cascading'), ('poisson', 'Poisson ×2'), ('brownout', 'brownout')]
    cols, gains, allsig, losses, bests, behind = [], [], [], [], [], {m: 0 for m in BASE}
    for k, h_ in order:
        g = rb[rb.fail == k]
        data = {m: per_seed(g, m) for m in [REF] + BASE}
        cols.append((h_, data))
        gains.append(np.mean(data[REF]) / max(np.mean(data[m]) for m in BASE) - 1)
        bests.append(max(BASE, key=lambda m: np.mean(data[m])))
        r = paired_wilcoxon_holm(data[REF], {m: data[m] for m in BASE})
        for m in BASE:
            behind[m] += int(r[m]['p_holm'] < 0.05 and np.mean(data[m]) < np.mean(data[REF]))
        allsig.append(all(r[m]['p_holm'] < 0.05 and np.mean(data[m]) < np.mean(data[REF]) for m in BASE))
        losses += [f'{h_}: {SHORT[m]}' for m in BASE
                   if r[m]['p_holm'] < 0.05 and np.mean(data[m]) > np.mean(data[REF])]
    return {'cols': cols, 'gains': gains, 'allsig': allsig, 'losses': losses, 'bests': bests, 'behind': behind}


def _sitl_fast(df: pd.DataFrame, b0_rows: str | None = None) -> pd.DataFrame:
    """Fast-simulator side of a PX4 check: B0 with the values the PX4 runs used (configs/tuned.yaml), on the
    released code. ``b0_rows``: a folder whose B0 rows replace the stored ones (same configs)."""
    if b0_rows:
        new = pd.read_parquet(os.path.join(b0_rows, 'summary.parquet'))
        assert set(new.cfg_method) == {'B0_pso'} and set(new.run_id) == set(df[df.cfg_method == 'B0_pso'].run_id)
        df = pd.concat([df[df.cfg_method != 'B0_pso'], new], ignore_index=True)
    sitl = yaml.safe_load(open(os.path.join(CONFIGS, 'tuned.yaml')))['B0_pso']
    for cfg in df[df.cfg_method == 'B0_pso'].config:
        assert all(_get(cfg, path) == v for path, v in _leaves(sitl)), 'B0 values differ from the PX4 runs'
    return df


def _sitl_rows(folder: str, seeds) -> pd.DataFrame:
    rows = []
    for s in seeds:
        for m in (REF, 'B0_pso'):
            d = json.load(open(os.path.join(folder, f'sentinel_g100_s{s}_{m}', 'summary.json')))
            rows.append({'seed': s, 'method': m, 'tdr_base': d['tdr_base'], 'cov': d['coverage'],
                         'connected': d['frac_connected'], 'pdr': d['comm']['pdr']})
    return pd.DataFrame(rows)


def sitl_e8r_e6() -> None:
    """E8r (original PX4 check re-run on the current code) and E6 (10 worlds, YOLO26n in the loop)."""
    fast = _sitl_fast(load('e8_fastsim', orig=True), os.path.join(RERUN, 'sitl_match', 'e8_fastsim'))
    sitl = _sitl_rows(os.path.join(OVN, 'e8r_sitl'), range(3))
    f = fast.set_index(['cfg_seed', 'cfg_method'])
    cmp_rows = []
    for _, r in sitl.iterrows():
        q = f.loc[(r.seed, r.method)]
        cmp_rows.append({'method': r.method, 'SITL tdr_base': r.tdr_base, 'fast tdr_base': q.tdr_base,
                         'SITL connected': r.connected, 'fast connected': q.frac_connected,
                         'SITL pdr': r.pdr, 'fast pdr': q.pdr})
    mm = pd.DataFrame(cmp_rows).groupby('method').mean()
    diffs = [(mm[a] - mm[b]).abs().max() for a, b in
             [('SITL tdr_base', 'fast tdr_base'), ('SITL connected', 'fast connected'), ('SITL pdr', 'fast pdr')]]
    mac('EightMaxDiff', f'{max(diffs):.2f}')
    mac('EightSitlS', f'{mm.loc[REF, "SITL tdr_base"]:.2f}'); mac('EightFastS', f'{mm.loc[REF, "fast tdr_base"]:.2f}')
    mac('EightSitlB', f'{mm.loc["B0_pso", "SITL tdr_base"]:.2f}'); mac('EightFastB', f'{mm.loc["B0_pso", "fast tdr_base"]:.2f}')
    CLAIMS['E8r: SITL S/B0, fast S/B0, max diff'] = (mm.loc[REF, 'SITL tdr_base'], mm.loc['B0_pso', 'SITL tdr_base'],
                                                     mm.loc[REF, 'fast tdr_base'], mm.loc['B0_pso', 'fast tdr_base'],
                                                     max(diffs))
    md.append('### PX4/Gazebo check re-run on the current code (E8r; parameterised detector model)\n')
    md.append('| method | SITL / fast reported | SITL / fast connected | SITL / fast packet delivery |\n|---|---|---|---|')
    for m in (REF, 'B0_pso'):
        md.append(f'| {NAMES[m]} | {mm.loc[m, "SITL tdr_base"]:.2f} / {mm.loc[m, "fast tdr_base"]:.2f} | '
                  f'{mm.loc[m, "SITL connected"]:.2f} / {mm.loc[m, "fast connected"]:.2f} | '
                  f'{mm.loc[m, "SITL pdr"]:.2f} / {mm.loc[m, "fast pdr"]:.2f} |')
    md.append('')
    # E6
    six = _sitl_rows(os.path.join(OVN, 'e6_sitl'), range(10))
    piv = six.pivot(index='seed', columns='method', values='tdr_base')
    d = piv[REF] - piv['B0_pso']
    p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(piv[REF], piv['B0_pso']).pvalue)
    m_, lo, hi = bootstrap_ci(d.values)
    mac('SixN', len(piv)); mac('SixS', f'{piv[REF].mean():.2f}'); mac('SixB', f'{piv["B0_pso"].mean():.2f}')
    mac('SixDiff', f'{m_:+.2f}'); mac('SixDiffLo', f'{lo:+.2f}'); mac('SixDiffHi', f'{hi:+.2f}'); mac('SixP', f'{p:.2f}')
    fs = _sitl_fast(load_ovn('e6_fastsim_v2', E5R, orig=True)).pivot(index='cfg_seed', columns='cfg_method',
                                                                     values='tdr_base')
    mac('SixFastS', f'{fs[REF].mean():.2f}'); mac('SixFastB', f'{fs["B0_pso"].mean():.2f}')
    fs4 = load_ovn('e6_fastsim_v2', E5R).pivot(index='cfg_seed', columns='cfg_method', values='tdr_base')
    f84 = load('e8_fastsim').groupby('cfg_method').tdr_base.mean()
    CLAIMS['E6: SITL S, B0, diff [CI], p; fast S, B0'] = (piv[REF].mean(), piv['B0_pso'].mean(), m_, lo, hi, p,
                                                          fs[REF].mean(), fs['B0_pso'].mean())
    md.append('### PX4/Gazebo check on 10 worlds with YOLO26n in the loop (E6)\n')
    md.append('3 PX4 x500 UAVs, 100 m worlds (seeds 0–9), crash of one UAV at 150 s, 300 s; the fine-tuned YOLO26n '
              'runs on each UAV\'s thermal stream. Reported victims per world:\n')
    md.append('| world | SENTINEL | B0 |\n|---|---|---|')
    for s_, r in piv.iterrows():
        md.append(f'| {s_} | {r[REF]:.2f} | {r["B0_pso"]:.2f} |')
    md.append(f'\nMean SENTINEL − B0: {m_:+.3f} (95 % CI {lo:+.3f} to {hi:+.3f}), paired Wilcoxon p = {p:.3f}. '
              f'Fast simulator on the same worlds with the fitted detector model: SENTINEL {fs[REF].mean():.2f}, '
              f'B0 {fs["B0_pso"].mean():.2f}.\n')
    md.append('Both PX4 checks flew B0 with the values tuned before the baseline re-run (configs/tuned.yaml), so '
              'their fast-simulator side uses the same values on the released code (plan section 9, 30 Oct). With '
              f'the re-tuned B0 the fast simulator gives B0 {f84["B0_pso"]:.2f} on the E8r worlds (SENTINEL '
              f'{f84[REF]:.2f}) and {fs4["B0_pso"].mean():.2f} on the E6 worlds (SENTINEL {fs4[REF].mean():.2f}).\n')


def rerun_note() -> None:
    """Extended file: the baseline re-run of 29-30 Oct (plan section 9) and the rows it supersedes."""
    def objectives(log: str) -> dict:
        out = {}
        for ln in open(os.path.join(RERUN, log)):
            if ': objective ' in ln:
                m, v = ln.split(': objective ')
                out[m.strip()] = float(v.split()[0])
        return out
    o_old, o_new = objectives('tune_check_old_values.log'), objectives('tune_check_new_values.log')
    md.append('### Baseline re-run (29–30 Oct 2026): superseded rows\n')
    md.append('Before submission, re-running stored missions on the released code showed that the rows of B0–B4 '
              'and B6 computed before Amendment A3 (14 Oct) did not reproduce: they combined views merged from other '
              'UAVs with the earlier naive-Bayes rule, while SENTINEL, B5, B7, B8 and every later run use the '
              'corroborative rule. '
              'B0–B4 and B6 were therefore re-tuned on the released code with the same Optuna budget, objective and '
              'tuning seeds, and every row of B0–B4, B6 and their fusion variants was re-run (E1–E4, E6, E6c, E8 fast '
              'side, E9, E11, E11b, E12, robustness C1, E6 fast side C2), as was E7\'s no-projection ablation (its '
              'behaviour changed on 22 Oct). Same cells and seeds; the superseded rows are archived in '
              'results/baseline_rerun_2026-10-29/superseded_rows/ and the original folders are unchanged. All tables '
              'above use the re-run rows; the PX4 checks keep B0\'s earlier values on both sides (see there).\n')
    md.append('| method | tuning objective, old values | re-tuned |\n|---|---|---|')
    for m in RC.BASELINES:
        md.append(f'| {plain(NAMES[m])} | {o_old[m]:.4f} | {o_new[m]:.4f} |')
    md.append(f'| {NAMES[REF]} (not re-tuned) | {o_old[REF]:.4f} | – |\n')
    md.append('Mean share reported within 60 s (E9: confirmed victims) over all cells and seeds, '
              'superseded → re-run:\n')
    md.append('| experiment | method | superseded | re-run |\n|---|---|---|---|')
    for spec in RC.SPECS:
        exp = os.path.basename(spec)
        old = pd.read_parquet(os.path.join(RERUN, 'superseded_rows', f'{exp}.parquet'))
        new = pd.read_parquet(os.path.join(RERUN, exp, 'summary.parquet'))
        met = M if old[M].notna().any() else 'tdr'
        a, b = old.groupby('cfg_method')[met].mean(), new.groupby('cfg_method')[met].mean()
        for m in sorted(a.index):
            md.append(f'| {exp} | {m} | {a[m]:.3f} | {b[m]:.3f} |')
    md.append('')
    # headline comparisons with the superseded rows
    full = {o: load('e3_failures', orig=o) for o in (True, False)}

    def e3gain(df):
        u = df[df.cfg_profile == 'urban_rubble']
        return [np.mean(per_seed(u[u.fail == k], REF)) / max(np.mean(per_seed(u[u.fail == k], m)) for m in BASE) - 1
                for k in ('none', 'single', 'simultaneous', 'cascading', 'poisson', 'brownout')]

    def tops(df, size):
        a = df[df.cfg_map_size == size]
        best = [max([REF] + BASE, key=lambda m: per_seed(a[a.cfg_n_uav == n], m).mean())
                for n in sorted(a.cfg_n_uav.unique())]
        return [SHORT.get(t, 'SENTINEL') for t in best]
    e1o, e1n = load('e1_scalability', orig=True), load('e1_scalability')
    ro, rn = _robust(load_ovn('robust_fitted_v2', E5R, orig=True)), _robust(load_ovn('robust_fitted_v2', E5R))
    g = lambda x: ', '.join(f'{100 * v:+.1f}' for v in x)  # noqa: E731
    md.append('Headline comparisons, superseded → re-run rows:\n')
    md.append(f'- Urban failure settings (Table 2), SENTINEL over the best baseline [%]: {g(e3gain(full[True]))} → '
              f'{g(e3gain(full[False]))}.')
    md.append(f'- Highest mean per swarm size N = 3, 5, 10, 20, 50 at 400 m: {", ".join(tops(e1o, 400))} → '
              f'{", ".join(tops(e1n, 400))}; at 200 m: {", ".join(tops(e1o, 200))} → {", ".join(tops(e1n, 200))}.')
    md.append(f'- Robustness with the fitted detector model (corrected false-alarm rate), SENTINEL over the best '
              f'baseline [%]: {g(ro["gains"])} → {g(rn["gains"])}; significant in no setting in either case.\n')


EXTRA = r"""
### Comparison with the closest approaches (qualitative)

✓ addressed, ◦ partly, – not addressed.

| Approach | online sources | reporting deadline | obstruction/fading channel | UAV failures | free search |
|---|---|---|---|---|---|
| λ₂ connectivity control (Zavlanos & Pappas 2008; Sabattini et al. 2013) | – | – | – | ◦ | – |
| Edge potentials (Ji & Egerstedt 2007) | – | – | – | – | – |
| Intermittent connectivity (Kantaros & Zavlanos 2017) | – | ◦ | – | – | ✓ |
| Data ferrying / age of information (Cui et al. 2023; Bafarassat & Coleri 2026) | – | ✓ | ◦ | – | ✓ |
| Distributed PSO / Bayesian search (Yang et al. 2024; Niu et al. 2026) | ✓ | – | – | – | ✓ |
| MARL coordination (Yu et al. 2022; Shi et al. 2025) | ◦ | ◦ | ◦ | ◦ | ✓ |
| First SENTINEL version | ✓ | – | – | ◦ | ◦ |
| SENTINEL (this work) | ✓ | ✓ | ✓ | ✓ | ✓ |

### Results omitted from the paper for space

**Lemma 2 (LMST spans the reliable graph).** Let G⁻ contain edge (i, j) iff j ∈ S_i and i ∈ S_j, all
computed from the same positions, and assume every UAV has heard each of its reliable neighbours.
Then G⁻ contains the minimum spanning forest of the reliable graph and has the same connected
components. *Proof:* an MST edge missing from a local MST would be the unique heaviest edge of a cycle
in the local graph; that cycle exists in the reliable graph, a contradiction.

**Proposition 2 (probabilistic delivery path).** If the preserved spanning tree M is maintained
(Corollary of Proposition 1), the probability that the realised fading network restricted to M is
disconnected in a slot is at most Σ_{e∈M} P_out(e) ≤ |M| ε (union bound).

**Proposition 3 (exposure bound).** Let UAVs fly at speed v along paths at most κ times the
straight-line distance, let a hand-off finish within τ_c of reaching the responder's component, and let
an entry confirmed at τ_e be held by k UAVs. With D the largest distance to the responder,
W = T_h + κD/v + τ_c and W̄ = 3T_h + κD/v + τ_c, and independent failures at rate λ_f,
Pr[entry lost] ≤ (1 − e^{−λ_f W})(1 − e^{−λ_f W̄})^{k−1} ≤ (λ_f W̄)^k. *Proof:* if the creator survives
W the entry is secured; otherwise every other holder becomes responsible when the entry is older than
3T_h and secures it within W̄.
"""


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    md.extend(['# ICIUS 2026 (UN006): extended results', '',
               'Generated by `paper_icius_ios/scripts/make_ios.py` from the per-seed results; do not edit. '
               'This file holds the tables and results that the 6-page conference paper omits for space.', ''])
    channel(); setup(); e2(); e3(); e1(); attribution(); safety(); e10(); e9_e8(); perception()
    sensitivity(); detector_e4_e5(); sitl_e8r_e6(); rerun_note()
    mac('NeverWorse', 'yes' if not WORSE else 'no')
    # the conclusion: "more victims ... than the baselines in most settings, never significantly worse"
    ntop = sum(t for _, t in TOP)
    assert not WORSE and ntop > len(TOP) / 2, (WORSE, ntop, len(TOP))
    mac('TopN', ntop); mac('TopTotal', len(TOP))
    CLAIMS['main comparisons with the highest SENTINEL mean'] = f'{ntop} of {len(TOP)}: not in ' + ', '.join(
        c for c, t in TOP if not t)
    md.append('### Re-checked claims (overnight 2026-10-28, rules fixed in docs/analysis_plan_icius2026.md)\n')
    md.append('- Original-form baselines significantly ahead of SENTINEL in any cell of the main tables: '
              + ('; '.join(WORSE) if WORSE else 'none') + '.')
    for k, v in CLAIMS.items():
        md.append(f'- {k}: {v}')
    md.append('')
    print('WORSE:', WORSE)
    for k, v in CLAIMS.items():
        print('CLAIM', k, '=>', v)
    write('numbers.tex', ''.join(f'\\newcommand{{\\n{k}}}{{{v}}}\n' for k, v in sorted(macros.items())))
    md.append('### Other results omitted from the paper\n')
    md.append(f'- Information loss (confirmed victims whose every holder crashed before delivery), failure settings: '
              f'at most {macros["LossAvgMax"]} % averaged over the six settings for any method; worst cell '
              f'{macros["LossWorst"]} % (twice the Poisson rate, severe channel), SENTINEL {macros["LossWorstS"]} % there. '
              f'No advantage in information loss is claimed.')
    md.append(f'- Per-agent planning time of SENTINEL: {macros["CompLo"]}–{macros["CompHi"]} ms for N = '
              f'{macros["EoneNmin"]}–{macros["EoneNmax"]} (planning period {macros["PlanPeriod"]} s).')
    md.append(f'- Severe channel, best hold time: p90 report latency {macros["ThSevLatInf"]} s without delivery vs '
              f'{macros["ThSevLatBest"]} s at T_h = {macros["ThSevBest"]} s.')
    md.append(f'- Detector (desktop RTX 3060, TensorRT FP16): YOLO26n {macros["YnTrt"]} ms vs YOLOv8n {macros["VeTrt"]} ms; '
              f'validation mAP50-95 {macros["YnMapB"]} vs {macros["VeMapB"]}. No embedded latency was measured.')
    md.append('')
    with open(EXT, 'w') as f:
        f.write('\n'.join(md) + EXTRA)
    print(f'wrote {len(macros)} macros, 4 tables, 1 figure, {os.path.abspath(EXT)}')
