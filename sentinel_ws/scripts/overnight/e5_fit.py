#!/usr/bin/env python3
"""E5: fit the in-loop detector model (``sentinel_core.detector_model``) to detector outputs.

Data: the fine-tuned detector's predictions on the E4 val worlds (``preds_<tag>_val.jsonl``).
For every frame and every non-buried victim whose full box (``full_2d`` box camera) has its
centre in the image, the observation is the highest confidence of a prediction whose centre lies
in that box (4 px margin), or "not reported" if there is none at the 0.25 floor. Covariates come
from the rig pose and the world definition: slant range r from the camera to the victim and
view-dependent occlusion occ = clip(occ0 + kappa sin(theta) cos(az - block_dir), 0, 1).

Model (as in the simulator): c ~ clip(N(mu, sigma^2), 0, 1), reported if c >= 0.25, with
mu = c_max (1 - occ)^gamma exp(-max(0, r - r_ref) / r_scale). The parameters
(c_max, gamma, r_ref, r_scale, sigma, kappa) are fitted by maximum likelihood with the
non-reports as left-censored observations (Tobit). False alarms: predictions that match no
victim. Their confidences give fp_mu and fp_sigma (normal truncated at 0.25). The simulator draws
Poisson(``fp_rate``) false-alarm candidates per look, each at a debris item drawn from the whole
world (kept only if that item is in the footprint) and reported only if its confidence
N(fp_mu, fp_sigma) reaches 0.25. The fit therefore stores ``fp_per_debris_item``: the reported
false alarms per debris item in view (every item of the world that projects into the frame,
visible or not, as in the simulator) divided by P(c >= 0.25). ``make_fitted_configs.py`` turns it
into ``fp_rate`` for the number of debris items in each configuration's worlds.

Goodness of fit on the test worlds (``preds_<tag>_test.jsonl``): observed vs predicted
P(c >= 0.25), P(c >= 0.40) and P(c >= 0.70) per slant-range bin and occlusion class, for the
fitted and the hand-set parameters, and the test log-likelihood of both.

    python e5_fit.py --root <E4 folder> --tag finetuned --out <E5 folder>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import optimize, stats

from sentinel_core.detector_model import DetectorModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e4_capture import project  # noqa: E402

FLOOR = 0.25
FIT_KEYS = ('c_max', 'gamma', 'r_ref', 'r_scale', 'sigma', 'kappa')
BOUNDS = {'c_max': (0.05, 1.0), 'gamma': (0.1, 8.0), 'r_ref': (0.0, 60.0),
          'r_scale': (5.0, 1000.0), 'sigma': (0.01, 0.6), 'kappa': (0.0, 2.0)}
W_IMG, H_IMG = 640, 512
DEBRIS = 200


def _inside(pt, box, pad=4.0):
    return box[0] - pad <= pt[0] <= box[2] + pad and box[1] - pad <= pt[1] <= box[3] + pad


def observations(root: str, tag: str, split: str):
    """Rows (r, sin_theta, cos(az - block_dir), occ0, conf or nan, world, k, label) and
    false-alarm records (conf, None); one (None, n) record per frame gives n, the world's debris
    items that project into the frame."""
    obs, fas, frames = [], [], 0
    metas, truths = {}, {}
    for line in open(os.path.join(root, f'preds_{tag}_{split}.jsonl')):
        p = json.loads(line)
        w = p['world']
        if w not in metas:
            metas[w] = {json.loads(m)['k']: json.loads(m)
                        for m in open(os.path.join(root, 'meta', w + '.jsonl'))}
            truths[w] = json.load(open(os.path.join(root, 'truth', w + '.json')))
        meta, truth = metas[w][p['k']], truths[w]
        frames += 1
        preds = p['preds']
        centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in preds]
        used = set()
        cam = np.array([meta['x'], meta['y'], meta['z']])
        for lab, x0, y0, x1, y1 in meta['full']:
            if lab == DEBRIS:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            t = truth['targets'][lab - 1]
            if t['buried'] or not (0 <= cx < W_IMG and 0 <= cy < H_IMG):
                continue
            hits = [i for i, c in enumerate(centres) if _inside(c, (x0, y0, x1, y1))]
            used.update(hits)
            conf = max((preds[i][4] for i in hits), default=float('nan'))
            v = np.array([cam[0] - t['pos'][0], cam[1] - t['pos'][1]])
            horiz = float(np.linalg.norm(v))
            r = math.hypot(horiz, cam[2])
            sin_t = horiz / r
            az = math.atan2(v[1], v[0])
            obs.append((r, sin_t, math.cos(az - t['block_dir']), t['occ0'], conf, w, p['k'], lab))
        n_debris = debris_in_frame(meta, truth['clutter'])
        for i, b in enumerate(preds):
            if i in used:
                continue
            c = centres[i]
            if any(_inside(c, fb[1:]) for fb in meta['full'] if fb[0] != DEBRIS):
                continue
            fas.append((b[4], None))
        fas.append((None, n_debris))            # one marker per frame: debris items in view
    return obs, fas, frames


def debris_in_frame(meta: dict, clutter) -> int:
    """Debris items of the world whose position projects into the frame, whether rendered visible
    or hidden in rubble (the simulator draws false alarms at every debris item in its footprint)."""
    n = 0
    for c in clutter:
        u, v = project(meta['x'], meta['y'], meta['z'], meta['yaw'], c[0], c[1])
        n += 0 <= u < W_IMG and 0 <= v < H_IMG
    return n


def fp_per_debris_item(n_fa: int, debris_in_view: int, fp_mu: float, fp_sigma: float) -> float:
    """Simulator false-alarm candidates per look per debris item in view: the reported rate
    (confidence >= FLOOR) divided by the share of candidates the simulator reports."""
    return n_fa / max(1, debris_in_view) / float(stats.norm.sf(FLOOR, fp_mu, fp_sigma))


def _mu(theta: dict, r, sin_t, cosd, occ0):
    occ = np.clip(occ0 + theta['kappa'] * sin_t * cosd, 0, 1)
    return theta['c_max'] * (1 - occ) ** theta['gamma'] * np.exp(
        -np.maximum(0.0, r - theta['r_ref']) / theta['r_scale'])


def loglik(theta: dict, X: np.ndarray) -> float:
    r, sin_t, cosd, occ0, c = X.T
    mu = _mu(theta, r, sin_t, cosd, occ0)
    s = theta['sigma']
    rep = ~np.isnan(c)
    ll = np.where(rep, stats.norm.logpdf(np.nan_to_num(c, nan=0.0), mu, s),
                  stats.norm.logcdf(FLOOR, mu, s))
    # confidences reported as 1.0 can come from the clipped upper tail
    top = rep & (c >= 0.999)
    ll = np.where(top, stats.norm.logsf(0.999, mu, s), ll)
    return float(ll.sum())


def fit(X: np.ndarray, start: dict) -> tuple[dict, float]:
    keys = list(FIT_KEYS)
    lo = np.array([BOUNDS[k][0] for k in keys])
    hi = np.array([BOUNDS[k][1] for k in keys])

    def nll(z):
        return -loglik(dict(zip(keys, z)), X)

    best = None
    rng = np.random.default_rng(0)
    starts = [np.array([start[k] for k in keys])] + [lo + rng.random(len(keys)) * (hi - lo)
                                                    for _ in range(15)]
    for z0 in starts:
        res = optimize.minimize(nll, np.clip(z0, lo, hi), method='L-BFGS-B',
                                bounds=list(zip(lo, hi)))
        if best is None or res.fun < best.fun:
            best = res
    return dict(zip(keys, map(float, best.x))), float(-best.fun)


def p_at_least(theta: dict, X: np.ndarray, thr: float) -> np.ndarray:
    r, sin_t, cosd, occ0, _ = X.T
    return stats.norm.sf(thr, _mu(theta, r, sin_t, cosd, occ0), theta['sigma'])


def calibration(theta_fit, theta_hand, X, occ_class, bins):
    rows = []
    r = X[:, 0]
    for cls in ('open', 'partly occluded'):
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (occ_class == cls) & (r >= lo) & (r < hi)
            if not sel.any():
                continue
            c = X[sel, 4]
            row = {'class': cls, 'range_m': f'{lo:.0f}-{hi:.0f}', 'n': int(sel.sum())}
            for thr in (FLOOR, 0.40, 0.70):
                row[f'obs@{thr}'] = float(np.mean(np.nan_to_num(c, nan=0.0) >= thr))
                row[f'fit@{thr}'] = float(np.mean(p_at_least(theta_fit, X[sel], thr)))
                row[f'hand@{thr}'] = float(np.mean(p_at_least(theta_hand, X[sel], thr)))
            rows.append(row)
    return rows


def truncnorm_fit(c: np.ndarray) -> tuple[float, float]:
    def nll(z):
        mu, s = z
        return -np.sum(stats.norm.logpdf(c, mu, s) - stats.norm.logsf(FLOOR, mu, s))
    res = optimize.minimize(nll, [float(np.mean(c)), max(0.02, float(np.std(c)))],
                            method='L-BFGS-B', bounds=[(-1.0, 1.0), (0.01, 1.0)])
    return float(res.x[0]), float(res.x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--tag', default='finetuned')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hand = DetectorModel()
    theta_hand = {k: getattr(hand, k) for k in FIT_KEYS}
    obs_v, fas_v, frames_v = observations(a.root, a.tag, 'val')
    obs_t, _, frames_t = observations(a.root, a.tag, 'test')
    Xv = np.array([o[:5] for o in obs_v], float)
    Xt = np.array([o[:5] for o in obs_t], float)
    theta, ll_v = fit(Xv, theta_hand)
    fa_conf = np.array([c for c, _ in fas_v if c is not None])
    debris_in_view = sum(n for c, n in fas_v if c is None)
    per_debris = len(fa_conf) / max(1, debris_in_view)
    fp_mu, fp_sigma = truncnorm_fit(fa_conf) if len(fa_conf) >= 5 else (hand.fp_mu, hand.fp_sigma)
    fitted = {**theta, 'fp_per_debris_item': fp_per_debris_item(len(fa_conf), debris_in_view,
                                                                fp_mu, fp_sigma),
              'fp_mu': fp_mu, 'fp_sigma': fp_sigma}
    occ_t = np.array(['open' if o[3] == 0 else 'partly occluded' for o in obs_t])
    bins = [20.0, 27.0, 30.0, 32.0, 35.0, 45.0]
    at_bound = [k for k in FIT_KEYS
                if min(abs(theta[k] - BOUNDS[k][0]), abs(theta[k] - BOUNDS[k][1]))
                <= 1e-3 * (BOUNDS[k][1] - BOUNDS[k][0])]
    report = {
        'tag': a.tag, 'fitted': fitted, 'at_bound': at_bound, 'bounds': BOUNDS,
        'hand_set': {**theta_hand, 'fp_rate': hand.fp_rate, 'fp_mu': hand.fp_mu,
                     'fp_sigma': hand.fp_sigma},
        'val': {'frames': frames_v, 'victim_views': len(Xv),
                'reported': int(np.sum(~np.isnan(Xv[:, 4]))), 'loglik': ll_v,
                'false_alarms': int(len(fa_conf)), 'debris_in_view': int(debris_in_view),
                'fa_per_debris_in_view': per_debris,
                'p_reported_fa': float(stats.norm.sf(FLOOR, fp_mu, fp_sigma)),
                'slant_range_m': [float(Xv[:, 0].min()), float(Xv[:, 0].max())]},
        'test': {'frames': frames_t, 'victim_views': len(Xt),
                 'loglik_fitted': loglik(theta, Xt), 'loglik_hand_set': loglik(theta_hand, Xt),
                 'calibration': calibration(theta, theta_hand, Xt, occ_t, bins)},
    }
    with open(os.path.join(a.out, 'fit.json'), 'w') as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(a.out, 'detector_fitted.json'), 'w') as f:
        json.dump(fitted, f, indent=1)
    print(json.dumps(report['fitted']), json.dumps(report['val']))


if __name__ == '__main__':
    main()
