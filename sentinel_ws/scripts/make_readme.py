#!/usr/bin/env python3
"""Generate README.md from readme_template.md (same folder).

Every result number in the README is filled in from the generated paper numbers
(paper_icius_ios/generated/numbers.tex), the extended results (docs/icius2026_extended_results.md) and
the detector benchmark (~/sentinel_ws/results/perception/perception.csv); none is typed by hand. The
template's qualitative statements (which comparisons are significant, who has the highest mean) are
asserted against the claim checks that make_ios.py writes into the extended results, so the README
fails to build when the data stop supporting them.

    python paper_icius_ios/scripts/make_ios.py      # first, if the results changed
    python sentinel_ws/scripts/make_readme.py       # --check: fail if README.md is out of date
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
NUMBERS = os.path.join(REPO, 'paper_icius_ios', 'generated', 'numbers.tex')
EXT = os.path.join(REPO, 'docs', 'icius2026_extended_results.md')
PERC = os.path.expanduser('~/sentinel_ws/results/perception/perception.csv')
TEMPLATE = os.path.join(HERE, 'readme_template.md')
OUT = os.path.join(REPO, 'README.md')
SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def plain(v: str) -> str:
    """LaTeX macro value -> Markdown text."""
    v = re.sub(r'(\d)\\,(?=\d{3})', r'\1,', v)                     # 1\,348\,166 -> 1,348,166
    v = v.replace('\\,\\%', ' %').replace('\\%', '%').replace('vs.\\ ', 'vs. ').replace('\\infty', '∞')
    v = re.sub(r'\\times\s*10\^\{(-?\d+)\}', lambda m: '×10' + m.group(1).translate(SUP), v)
    v = re.sub(r'^10\^\{(-?\d+)\}$', lambda m: '10' + m.group(1).translate(SUP), v)
    v = v.replace('$-$', '−')
    v = re.sub(r'\$([^$]*)\$', lambda m: m.group(1).replace('=', ' = '), v)
    v = re.sub(r'^-(?=\d)', '−', v)
    assert not re.search(r'[\\${}]', v), v
    return v


def numbers() -> dict[str, str]:
    raw = {}
    for line in open(NUMBERS):
        m = re.match(r'\\newcommand\{\\n(\w+)\}\{(.*)\}\s*$', line)
        if m:
            raw[m.group(1)] = m.group(2)
    return raw


def claim(ext: str, key: str) -> str:
    m = re.search('^- ' + re.escape(key) + ': (.*)$', ext, re.M)
    assert m, f'claim not found in the extended results: {key}'
    return m.group(1)


def check_claims(raw: dict[str, str], ext: str) -> None:
    """The template's wording, checked against make_ios.py's claim checks."""
    f = {k: float(v) for k, v in raw.items() if re.fullmatch(r'[+-]?\d+(\.\d+)?', v)}
    # channel table: significantly ahead of every baseline except in the severe profile
    for p, s, b in (('open', 'ChOpenS', 'ChOpenB'), ('suburban', 'ChSubS', 'ChSubB'),
                    ('urban_rubble', 'ChUrbS', 'ChUrbB')):
        assert claim(ext, f'channel {p}: SENTINEL significantly above every baseline') == 'True', p
        assert f[s] > f[b], p
    assert claim(ext, 'channel severe: SENTINEL significantly above every baseline') == 'False'
    assert claim(ext, 'channel severe: not significant vs') == f"['{raw['ChSevBname']}']"
    assert f['ChSevS'] > f['ChSevB']
    # urban failures: ahead in every setting, not significant only against one baseline in one setting
    assert f['EthreeGainLo'] > 0
    ns = [claim(ext, f'failures urban_rubble {s}: not significant vs')
          for s in ('none', 'single', 'simultaneous', 'cascading', 'poisson', 'brownout')]
    other = [x for x in ns if x != '[]']
    assert other == ["['" + raw['EthreeNotSig'].split()[0] + "']"], ns
    # 400 m: significantly ahead of every baseline from EoneSigFrom on, not below it
    for n, v in re.findall(r'^- swarm 400 m N=(\d+): not significant vs: (.*)$', ext, re.M):
        assert (v == '[]') == (int(n) >= int(raw['EoneSigFrom'])), (n, v)
    # 200 m: a baseline has the highest mean at every size, and the names match
    tops = re.findall(r'^- swarm 200 m N=\d+: highest mean: (.*)$', ext, re.M)
    assert tops and 'SENTINEL' not in tops and sorted(set(tops)) == raw['EoneTwoHundredTopNames'].split(', ')
    # no original-form baseline significantly ahead anywhere (covers the 200 m "never significantly")
    assert claim(ext, 'Original-form baselines significantly ahead of SENTINEL in any cell of the main '
                      'tables') == 'none.' and raw['NeverWorse'] == 'yes'
    # fitted detector model: never significant either way
    assert claim(ext, 'robustness: settings with SENTINEL significantly above all') == '0'
    assert claim(ext, 'robustness: significant losses of SENTINEL') == '[]' and raw['RobNsig'] == '0'
    assert int(raw['RobNtop']) < int(raw['RobNset'])
    # attribution: one baseline + fusion no longer significantly behind in urban; dense: Table 4 vs nine-way Holm
    assert claim(ext, 'urban N=8: baselines + fusion not significantly behind') == f"['{raw['UrbBestFusName']}']"
    assert claim(ext, 'dense: best baseline + fusion significantly above SENTINEL (Table 4)') == raw['DenseSigAbove']
    assert claim(ext, 'dense: baselines + fusion significantly above SENTINEL (Holm over all nine)') == '[]'
    # sensitivity: never significant; PX4 ten-world check: no difference detected
    assert claim(ext, 'sensitivity: significant variants') == '[]' and raw['SensNsig'] == '0'
    assert f['SixP'] >= 0.05


def values(raw: dict[str, str], ext: str) -> dict[str, str]:
    v = {k: plain(x) for k, x in raw.items()}
    p = pd.read_csv(PERC).drop_duplicates('model').set_index('model')
    for key, model in (('Yn', 'yolo26n_thermal_best'), ('Ve', 'yolov8n_thermal_best')):
        v[key + 'Params'] = f'{p.loc[model, "params_M"]:.2f}'
        v[key + 'Gflops'] = f'{p.loc[model, "gflops"]:.1f}'
        # the same benchmark file as the paper's mAP50 macros
        assert f'{p.loc[model, "mAP50"]:.3f}' == raw[key + 'Map'], model
    m = re.search(r'^### Detector at the operating altitude \(rendered thermal frames, (\d+) and (\d+) m\)$', ext, re.M)
    assert m, 'rendered-frame heading not found'
    v['RdAltLo'], v['RdAltHi'] = m.groups()
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='fail if README.md differs from the generated text')
    a = ap.parse_args()
    raw, ext = numbers(), open(EXT).read()
    check_claims(raw, ext)
    v = values(raw, ext)
    text = open(TEMPLATE).read()
    missing = sorted(set(re.findall(r'\{\{(\w+)\}\}', text)) - set(v))
    assert not missing, f'unknown placeholders: {missing}'
    text = re.sub(r'\{\{(\w+)\}\}', lambda m: v[m.group(1)], text)
    if a.check:
        if open(OUT).read() != text:
            sys.exit('README.md is out of date: run sentinel_ws/scripts/make_readme.py')
        print('README.md is up to date')
        return
    with open(OUT, 'w') as f:
        f.write(text)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
