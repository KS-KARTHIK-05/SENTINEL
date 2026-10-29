#!/usr/bin/env python3
"""E5 sanity check: the in-loop detector model against the real detector on the same frames.

For every rendered frame of a split, the simulator's ``DetectorModel.look`` is run with that
frame's camera pose (x, y, z), the victims whose full box has its centre in the image (the views
the fit uses; buried victims are reported separately) and the world's warm-debris items. A
simulated false alarm counts if its position projects into the image and not into a victim box,
as for the real detector. ``look`` places false alarms at debris items drawn from the whole
world, so the false-alarm rate seen in a frame depends on the number of debris items in the world
the model runs in; it is evaluated for each world size the model was used in (``--world-debris``,
padding the debris list with items far outside the frame).

Real side: the detector's predictions on the same frames (``preds_<tag>_<split>.jsonl``), with
the definitions of ``e5_fit.observations`` (victim view reported at the highest confidence of a
prediction centred in its full box; any other prediction is a false alarm).

    python e5_check.py --root <E4 folder> --params <detector JSON> [--params ...] --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e4_capture import project  # noqa: E402
from e5_fit import DEBRIS, H_IMG, W_IMG, _inside, observations  # noqa: E402
from sentinel_core.detector_model import DetectorModel, Target  # noqa: E402

THRESHOLDS = (0.25, 0.40, 0.70)
FAR = 1.0e6                                   # padding debris items, never in view


def frames(root: str, tag: str, split: str):
    """Per frame: camera xyz, victims in view (Target, class), buried victims in view, debris."""
    out, truths, metas = [], {}, {}
    for line in open(os.path.join(root, f'preds_{tag}_{split}.jsonl')):
        p = json.loads(line)
        w = p['world']
        if w not in truths:
            truths[w] = json.load(open(os.path.join(root, 'truth', w + '.json')))
            metas[w] = {json.loads(m)['k']: json.loads(m)
                        for m in open(os.path.join(root, 'meta', w + '.jsonl'))}
        m, truth = metas[w][p['k']], truths[w]
        vic, boxes = [], []
        for lab, x0, y0, x1, y1 in m['full']:
            if lab == DEBRIS:
                continue
            boxes.append((x0, y0, x1, y1))
            t = truth['targets'][lab - 1]
            if t['buried'] or not (0 <= (x0 + x1) / 2 < W_IMG and 0 <= (y0 + y1) / 2 < H_IMG):
                continue
            vic.append((Target(np.asarray(t['pos'], float), t['occ0'], t['block_dir'], False),
                        'open' if t['occ0'] == 0 else 'partly occluded'))
        buried = []
        for t in truth['targets']:
            u, v = project(m['x'], m['y'], m['z'], m['yaw'], *t['pos'])
            if t['buried'] and 0 <= u < W_IMG and 0 <= v < H_IMG:
                buried.append((t['pos'], (u, v)))
        out.append({'world': w, 'k': p['k'], 'cam': np.array([m['x'], m['y'], m['z']]),
                    'yaw': m['yaw'], 'victims': vic, 'victim_boxes': boxes, 'buried': buried,
                    'debris': np.asarray(truth['clutter'], float).reshape(-1, 2),
                    'preds': p['preds'],
                    'n_debris_vis': sum(1 for b in m['vis'] if b[0] == DEBRIS)})
    return out


def in_frame(f: dict, xy) -> bool:
    u, v = project(f['cam'][0], f['cam'][1], f['cam'][2], f['yaw'], *xy)
    return 0 <= u < W_IMG and 0 <= v < H_IMG and not any(
        _inside((u, v), b) for b in f['victim_boxes'])


def poisson_ci(k: int, n: int):
    lo = 0.0 if k == 0 else stats.chi2.ppf(0.025, 2 * k) / 2
    hi = stats.chi2.ppf(0.975, 2 * k + 2) / 2
    return lo / n, hi / n


def real_side(root: str, tag: str, split: str, fr: list) -> dict:
    obs, fas, n_frames = observations(root, tag, split)
    conf = np.array([o[4] for o in obs], float)
    cls = np.array(['open' if o[3] == 0 else 'partly occluded' for o in obs])
    res = {'frames': n_frames, 'victim_views': {c: int(np.sum(cls == c))
                                                for c in ('open', 'partly occluded')}}
    for thr in THRESHOLDS[1:]:
        res[f'recall@{thr}'] = {c: float(np.mean(np.nan_to_num(conf[cls == c]) >= thr))
                                for c in ('open', 'partly occluded')}
    rep = ~np.isnan(conf)
    res['mean_conf_reported'] = {c: float(np.mean(conf[rep & (cls == c)]))
                                 for c in ('open', 'partly occluded')}
    fa = np.array([c for c, _ in fas if c is not None], float)
    for thr in THRESHOLDS[:2]:
        k = int(np.sum(fa >= thr))
        res[f'fa_per_frame@{thr}'] = k / n_frames
        res[f'fa_count@{thr}'] = k
        res[f'fa_per_frame@{thr}_ci95'] = poisson_ci(k, n_frames)
    res['fa_mean_conf'] = float(np.mean(fa)) if len(fa) else None
    res['debris_in_view_per_frame'] = sum(n for c, n in fas if c is None) / n_frames
    # predictions at buried victims (no full box, so counted among the false alarms above)
    hits = views = 0
    for f in fr:
        for _, (u, v) in f['buried']:
            views += 1
            hits += any(abs((p[0] + p[2]) / 2 - u) < 20 and abs((p[1] + p[3]) / 2 - v) < 20
                        for p in f['preds'])
    res['buried'] = {'views': views, 'reported': hits}
    return res


def sim_side(params: dict, fr: list, world_debris: int, draws: int, seed: int) -> dict:
    params = dict(params)
    if 'fp_per_debris_item' in params:        # fit output: fp_rate for this world's debris count
        params['fp_rate'] = params.pop('fp_per_debris_item') * world_debris
    det = DetectorModel(**params)
    rng = np.random.default_rng(seed)
    acc = {c: {thr: 0 for thr in THRESHOLDS} for c in ('open', 'partly occluded')}
    n_views = {c: 0 for c in ('open', 'partly occluded')}
    conf_sum = {c: 0.0 for c in ('open', 'partly occluded')}
    fa = {thr: 0 for thr in THRESHOLDS[:2]}
    fa_conf, buried_views, buried_rep = [], 0, 0
    for _ in range(draws):
        for f in fr:
            pad = np.full((max(0, world_debris - len(f['debris'])), 2), FAR)
            debris = np.vstack([f['debris'], pad])
            targets = [t for t, _ in f['victims']] + [
                Target(np.asarray(p, float), 1.0, 0.0, True) for p, _ in f['buried']]
            dets = det.look(targets, f['cam'], math.inf, rng, debris)
            got = {}
            for pos, c, k in dets:
                if k >= 0:
                    got[k] = max(c, got.get(k, 0.0))
                elif in_frame(f, pos):
                    fa_conf.append(c)
                    for thr in fa:
                        fa[thr] += c >= thr
            for k, (_, cl) in enumerate(f['victims']):
                n_views[cl] += 1
                c = got.get(k, 0.0)
                if c >= det.report_min:
                    conf_sum[cl] += c
                for thr in THRESHOLDS:
                    acc[cl][thr] += c >= thr
            for j in range(len(f['buried'])):
                buried_views += 1
                buried_rep += (len(f['victims']) + j) in got
    n_frames = len(fr) * draws
    res = {'world_debris': world_debris, 'fp_rate': det.fp_rate, 'draws': draws}
    for thr in THRESHOLDS[1:]:
        res[f'recall@{thr}'] = {c: acc[c][thr] / n_views[c] for c in n_views}
    res['mean_conf_reported'] = {c: conf_sum[c] / max(1, acc[c][0.25]) for c in n_views}
    for thr in fa:
        res[f'fa_per_frame@{thr}'] = fa[thr] / n_frames
    res['fa_mean_conf'] = float(np.mean(fa_conf)) if fa_conf else None
    res['buried'] = {'views_per_draw': buried_views // draws,
                     'reported_share': buried_rep / max(1, buried_views)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--tag', default='finetuned')
    ap.add_argument('--splits', nargs='+', default=['test', 'val'])
    ap.add_argument('--params', nargs='+', required=True, help='detector JSON files (simulator '
                    'parameters, or the fit output with fp_per_debris_item); "hand" for the hand-set '
                    'defaults')
    ap.add_argument('--world-debris', type=int, nargs='+', default=[480, 30],
                    help='debris items in the worlds the model runs in (400 m: 480, 100 m: 30)')
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {}
    for split in a.splits:
        fr = frames(a.root, a.tag, split)
        rep = {'real': real_side(a.root, a.tag, split, fr), 'sim': {}}
        for p in a.params:
            params = {} if p == 'hand' else json.load(open(p))
            name = 'hand' if p == 'hand' else os.path.splitext(os.path.basename(p))[0]
            for nd in a.world_debris:
                rep['sim'][f'{name}@{nd}'] = sim_side(params, fr, nd, a.draws, seed=1)
        report[split] = rep
        print(split, json.dumps(rep, indent=1))
    with open(os.path.join(a.out, 'check.json'), 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
