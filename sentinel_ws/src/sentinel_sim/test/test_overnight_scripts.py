"""Tests for the overnight E4/E5 scripts (rendering helpers, dataset/evaluation, detector fit)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

SCRIPTS = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                        'scripts', 'overnight'))
sys.path.insert(0, SCRIPTS)

import e4_capture as C  # noqa: E402
import e4_detector as D  # noqa: E402
import e4_render as R  # noqa: E402
import e5_fit as F  # noqa: E402
from sentinel_core.detector_model import DetectorModel, Target  # noqa: E402
from sentinel_sim import world as W  # noqa: E402


# ------------------------------------------------------------------ rendering helpers
def _world(seed=5):
    return W.generate(100.0, 20, np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[0]),
                      base=False)


def test_rig_world_labels_victims_and_debris():
    mw = R._make_world_module()
    w = _world()
    sdf, truth = mw.build(w, 'e4_test', np.random.default_rng(6))
    out = R.rig_world(sdf, 'e4_test')
    for k in range(len(truth['targets'])):
        assert f'<name>victim_{k}</name>' in out
        assert f'<label>{k + 1}</label>' in out
    assert out.count('<label>200</label>') == len(truth['clutter'])
    assert out.count('<model name="rig">') == 1 and 'gz-sim-sensors-system' in out
    assert out.count('type="boundingbox_camera"') == 2 and 'type="thermal"' in out


def test_viewpoints_avoid_buildings_and_stay_in_bounds():
    w = _world()
    vps = R.viewpoints(w, 5)
    assert len(vps) > 20
    xy = np.array([v[:2] for v in vps])
    assert not np.any(w.in_no_fly(xy))
    lo, hi = w.bounds
    assert np.all(xy >= lo) and np.all(xy <= hi)
    assert {v[2] for v in vps} <= {25.0, 30.0}
    assert vps == R.viewpoints(w, 5)                      # deterministic


def test_white_hot_matches_thermal_vision():
    l16 = np.array([[27000, 30000, 33000, 34000]], np.uint16)   # 270, 300, 330, 340 K
    assert R.to_white_hot(l16).tolist() == [[0, 127, 255, 255]]


def test_quaternion_nadir_pitch():
    w, x, y, z = R.quat(0.0, math.pi / 2, 0.0)
    assert np.allclose([w, x, y, z], [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])


def test_projection_nadir_camera():
    assert np.allclose(C.project(10.0, 20.0, 30.0, 0.7, 10.0, 20.0), (320.0, 256.0))
    u, v = C.project(0.0, 0.0, 30.0, 0.0, 3.0, 0.0)      # ahead (heading +x): image up
    assert u == pytest.approx(320.0) and v < 256.0
    u, v = C.project(0.0, 0.0, 30.0, 0.0, 0.0, 3.0)      # left of heading: image left
    assert u < 320.0 and v == pytest.approx(256.0)


def _boxes_at(pose, targets, clutter, half=10.0, jitter=3.0):
    out = []
    for k, t in enumerate(targets):
        u, v = C.project(*pose, *t['pos'])
        if 0 <= u < 640 and 0 <= v < 512:
            out.append([k + 1, u - half + jitter, v - half, u + half + jitter, v + half])
    for c in clutter:
        u, v = C.project(*pose, *c)
        if 0 <= u < 640 and 0 <= v < 512:
            out.append([200, u - half, v - half - jitter, u + half, v + half - jitter])
    return [[b[0], max(0.0, b[1]), max(0.0, b[2]), min(639.0, b[3]), min(511.0, b[4])] for b in out]


def test_pose_check_accepts_matching_and_rejects_stale_frames():
    targets = [{'pos': [50.0, 50.0]}, {'pos': [55.0, 47.0]}, {'pos': [80.0, 80.0]}]
    clutter = [[47.0, 53.0], [60.0, 60.0]]
    pose = (52.0, 50.0, 30.0, 0.4)
    full = _boxes_at(pose, targets, clutter)
    assert len(full) == 3 and C.pose_consistent(full, pose, targets, clutter)
    stale = _boxes_at((48.0, 52.0, 30.0, 0.4), targets, clutter)          # previous viewpoint
    assert not C.pose_consistent(stale, pose, targets, clutter)
    assert not C.pose_consistent(full, (52.0, 50.0, 30.0, 1.4), targets, clutter)
    assert C.pose_consistent([], pose, targets, clutter)                   # nothing to check
    assert not C.pose_consistent([[9, 0, 0, 10, 10]], pose, targets, clutter)
    # debris boxes alone also reveal a stale frame
    only_debris = [b for b in stale if b[0] == 200]
    assert only_debris and not C.pose_consistent(only_debris, pose, targets, clutter)


# ------------------------------------------------------------------ dataset and evaluation
def test_gt_boxes_drop_debris_and_slivers():
    row = {'vis': [[1, 0, 0, 10, 10], [2, 0, 0, 3, 40], [3, 0, 0, 5, 5], [200, 0, 0, 30, 30]]}
    assert [b[0] for b in D.gt_boxes(row)] == [1]


def test_split_seeds_must_be_disjoint():
    with pytest.raises(AssertionError):
        D.split_of(3, {'train': {1, 3}, 'test': {3}})
    assert D.split_of(3, {'train': {1}, 'test': {3}}) == 'test'


def test_match_respects_threshold_and_iou():
    gts = [(1, 0, 0, 10, 10), (2, 50, 50, 60, 60)]
    preds = [[0, 0, 10, 10, 0.9], [50, 50, 60, 61, 0.5], [100, 100, 110, 110, 0.8]]
    used, un = D.match(preds, gts, 0.4)
    assert used == {0, 1} and un == [2]
    used, un = D.match(preds, gts, 0.7)
    assert used == {0} and un == [2]


def test_frame_metrics_classify_false_alarms():
    truth = {'targets': [{'occ0': 0.0, 'buried': False}, {'occ0': 0.45, 'buried': False}]}
    row = {'vis': [[1, 0, 0, 10, 20], [200, 100, 100, 120, 120]],
           'full': [[1, 0, 0, 10, 20], [2, 200, 200, 220, 240], [200, 100, 100, 120, 120]]}
    preds = [[0, 0, 10, 20, 0.8],          # open victim
             [205, 205, 215, 230, 0.6],    # hidden part of victim 2: not a false alarm
             [102, 102, 118, 118, 0.5],    # warm debris
             [400, 400, 410, 410, 0.3]]    # elsewhere
    m = D.frame_metrics(row, truth, preds)
    assert m['gt'] == {'open': 1, 'partly occluded': 0}
    assert m['hit@0.7']['open'] == 1
    assert m['fa_debris@0.4'] == 1 and m['fa_other@0.4'] == 0
    assert m['fa_other@0.25'] == 1


# ------------------------------------------------------------------ detector-model fit
def _synthetic(theta: dict, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    dm = DetectorModel(**theta)
    X = []
    for _ in range(n):
        alt = rng.choice([25.0, 30.0])
        off = rng.random() * 14.0 * np.array([math.cos(a := rng.random() * 6.283), math.sin(a)])
        occ0 = float(rng.choice([0.0, 0.0, 0.2, 0.45, 0.6]))
        bd = float(rng.random() * 6.283)
        tgt = Target(np.zeros(2), occ0, bd, False)
        uav = np.r_[off, alt]
        c = float(np.clip(rng.normal(dm.mean_conf(tgt, uav), dm.sigma), 0, 1))
        horiz = float(np.linalg.norm(off))
        r = math.hypot(horiz, alt)
        az = math.atan2(off[1], off[0])
        X.append((r, horiz / r, math.cos(az - bd), occ0, c if c >= F.FLOOR else np.nan))
    return np.array(X, float)


def test_fit_recovers_identifiable_parameters():
    truth = dict(c_max=0.85, gamma=2.0, r_ref=40.0, r_scale=60.0, sigma=0.1, kappa=0.6)
    X = _synthetic(truth)
    theta, _ = F.fit(X, {k: getattr(DetectorModel(), k) for k in F.FIT_KEYS})
    for k in ('c_max', 'gamma', 'sigma', 'kappa'):
        assert theta[k] == pytest.approx(truth[k], rel=0.15), k
    assert F.loglik(theta, X) >= F.loglik(truth, X) - 1e-6


def test_truncated_normal_fit():
    rng = np.random.default_rng(1)
    c = rng.normal(0.3, 0.1, 20000)
    c = c[c >= F.FLOOR]
    mu, s = F.truncnorm_fit(c)
    assert mu == pytest.approx(0.3, abs=0.02) and s == pytest.approx(0.1, abs=0.02)
