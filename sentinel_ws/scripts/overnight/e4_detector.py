#!/usr/bin/env python3
"""E4: detector evaluated and fine-tuned on rendered thermal frames at 25-30 m.

Subcommands (all paths under ``--root``, the E4 results folder):

``build``    YOLO dataset from the rendered frames (``e4_render.py``). Images are the on-board
             detector input (8-bit white-hot -> ``ThermalDetector.preprocess``: CLAHE, BGR).
             Labels are the visible boxes of victims (``2d`` box camera) with an area of at least
             MIN_AREA px^2 and both sides >= MIN_SIDE px. Split by world seed (``--train``,
             ``--val``, ``--test``), so no world is in two splits.
``eval``     For one model on one split: mAP50 and mAP50-95 (Ultralytics ``val``); recall at the
             0.40 and 0.70 confidence thresholds for open and partly occluded victims (IoU >= 0.5,
             greedy by confidence, per threshold); false alarms per frame on warm debris. All
             predictions at the on-board floor (0.25) are saved for the detector-model fit (E5).
``finetune`` Fine-tunes the HIT-UAV checkpoint on the train worlds and validates on the val
             worlds with the original training arguments of that checkpoint (AdamW, lr0, batch,
             image size, seed, augmentation, 100 epochs).

A victim is "open" if its slab occlusion occ0 in the world definition is 0, else "partly
occluded" (including a buried victim of which some part is visible from an oblique view).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'src', 'sentinel_ai')))
from sentinel_ai.detector import ThermalDetector  # noqa: E402

VICTIM_LABEL0, DEBRIS_LABEL = 1, 200
MIN_AREA, MIN_SIDE = 40.0, 4.0
CONF_FLOOR = 0.25                    # thermal_vision conf_floor
THRESHOLDS = (0.40, 0.70)            # SA band / confirmation
W_IMG, H_IMG = 640, 512


def load_world(root: str, world: str):
    rows = [json.loads(line) for line in open(os.path.join(root, 'meta', world + '.jsonl'))]
    truth = json.load(open(os.path.join(root, 'truth', world + '.json')))
    return rows, truth


def victim_class(truth: dict, label: int) -> str:
    t = truth['targets'][label - VICTIM_LABEL0]
    return 'open' if (t['occ0'] == 0 and not t['buried']) else 'partly occluded'


def gt_boxes(row: dict) -> list:
    """Visible victim boxes kept as ground truth: [(label, x0, y0, x1, y1)]."""
    out = []
    for lab, x0, y0, x1, y1 in row['vis']:
        if lab == DEBRIS_LABEL:
            continue
        w, h = x1 - x0, y1 - y0
        if w * h >= MIN_AREA and min(w, h) >= MIN_SIDE:
            out.append((lab, x0, y0, x1, y1))
    return out


def worlds_for(seeds, root):
    return [os.path.basename(p)[:-len('.jsonl')] for p in sorted(glob.glob(
        os.path.join(root, 'meta', '*.jsonl'))) if int(p.split('_s')[-1].split('.')[0]) in seeds]


def split_of(seed: int, splits: dict) -> str | None:
    hits = [k for k, v in splits.items() if seed in v]
    assert len(hits) <= 1, f'world seed {seed} in two splits'
    return hits[0] if hits else None


# ----------------------------------------------------------------------------- build
def build(root: str, splits: dict) -> str:
    pre = ThermalDetector.__new__(ThermalDetector)        # preprocessing only, no model load
    pre.preprocess_mode, pre.invert = 'clahe', False
    pre._clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    ds = os.path.join(root, 'dataset')
    shutil.rmtree(ds, ignore_errors=True)
    counts = {}
    for name, seeds in splits.items():
        os.makedirs(os.path.join(ds, name, 'images'))
        os.makedirs(os.path.join(ds, name, 'labels'))
        nf = nb = 0
        for w in worlds_for(seeds, root):
            rows, _ = load_world(root, w)
            for r in rows:
                img = cv2.imread(os.path.join(root, 'frames', w, f'{r["k"]:04d}.png'),
                                 cv2.IMREAD_GRAYSCALE)
                stem = f'{w}_{r["k"]:04d}'
                cv2.imwrite(os.path.join(ds, name, 'images', stem + '.png'), pre.preprocess(img))
                lines = []
                for _, x0, y0, x1, y1 in gt_boxes(r):
                    lines.append(f'0 {(x0 + x1) / 2 / W_IMG:.6f} {(y0 + y1) / 2 / H_IMG:.6f} '
                                 f'{(x1 - x0) / W_IMG:.6f} {(y1 - y0) / H_IMG:.6f}')
                with open(os.path.join(ds, name, 'labels', stem + '.txt'), 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))
                nf += 1
                nb += len(lines)
        counts[name] = {'worlds': len(worlds_for(seeds, root)), 'frames': nf, 'boxes': nb,
                        'seeds': sorted(seeds)}
    with open(os.path.join(ds, 'data.yaml'), 'w') as f:
        f.write(f'path: {ds}\ntrain: train/images\nval: val/images\ntest: test/images\n'
                'names:\n  0: victim\n')
    with open(os.path.join(root, 'dataset_counts.json'), 'w') as f:
        json.dump(counts, f, indent=1)
    print(json.dumps(counts))
    return ds


# ----------------------------------------------------------------------------- eval
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr: float):
    """Greedy match of predictions (conf >= thr) to GT boxes at IoU >= 0.5.
    Returns (matched GT indices, unmatched prediction indices)."""
    order = sorted((i for i, p in enumerate(preds) if p[4] >= thr), key=lambda i: -preds[i][4])
    used, unmatched = set(), []
    for i in order:
        best, bj = 0.5, None
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(preds[i][:4], g[1:])
            if v >= best:
                best, bj = v, j
        if bj is None:
            unmatched.append(i)
        else:
            used.add(bj)
    return used, unmatched


def inside(pt, box, pad=4.0) -> bool:
    return box[0] - pad <= pt[0] <= box[2] + pad and box[1] - pad <= pt[1] <= box[3] + pad


def frame_metrics(r: dict, truth: dict, preds: list) -> dict:
    gts = gt_boxes(r)
    cls = [victim_class(truth, g[0]) for g in gts]
    full_victims = [b[1:] for b in r['full'] if b[0] != DEBRIS_LABEL]
    debris = [b[1:] for b in r['vis'] if b[0] == DEBRIS_LABEL]
    out = {'gt': {c: cls.count(c) for c in ('open', 'partly occluded')}}
    for thr in (CONF_FLOOR,) + THRESHOLDS:
        used, un = match(preds, gts, thr)
        out[f'hit@{thr}'] = {c: sum(1 for j in used if cls[j] == c)
                            for c in ('open', 'partly occluded')}
        fa_debris = fa_other = 0
        for i in un:
            p = preds[i]
            c = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
            if any(inside(c, b) for b in full_victims):
                continue                      # a (hidden or filtered) victim, not a false alarm
            if any(inside(c, b) for b in debris):
                fa_debris += 1
            else:
                fa_other += 1
        out[f'fa_debris@{thr}'], out[f'fa_other@{thr}'] = fa_debris, fa_other
    return out


def evaluate(root: str, model: str, split: str, splits: dict, tag: str, device: str = '0'):
    from ultralytics import YOLO
    ds = os.path.join(root, 'dataset')
    yolo = YOLO(model, task='detect')
    # workers=0: forked dataloader workers deadlocked once (2026-10-28); results are unchanged
    v = yolo.val(data=os.path.join(ds, 'data.yaml'), split=split, imgsz=640, batch=16,
                 device=device, verbose=False, plots=False, workers=0,
                 project=os.path.join(root, 'val_runs'), name=f'{tag}_{split}', exist_ok=True)
    res = {'model': model, 'tag': tag, 'split': split,
           'map50': float(v.box.map50), 'map50_95': float(v.box.map)}
    tot = {}
    pred_path = os.path.join(root, f'preds_{tag}_{split}.jsonl')
    with open(pred_path, 'w') as fp:
        for w in worlds_for(splits[split], root):
            rows, truth = load_world(root, w)
            for r in rows:
                img = cv2.imread(os.path.join(ds, split, 'images', f'{w}_{r["k"]:04d}.png'))
                p = yolo.predict(img, imgsz=640, conf=CONF_FLOOR, device=device, verbose=False)[0]
                preds = ([list(map(float, b)) + [float(c)] for b, c in
                          zip(p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy())]
                         if p.boxes is not None else [])
                fp.write(json.dumps({'world': w, 'k': r['k'], 'preds': preds}) + '\n')
                m = frame_metrics(r, truth, preds)
                for k, val in m.items():
                    if isinstance(val, dict):
                        d = tot.setdefault(k, {})
                        for c, x in val.items():
                            d[c] = d.get(c, 0) + x
                    else:
                        tot[k] = tot.get(k, 0) + val
                tot['frames'] = tot.get('frames', 0) + 1
    for thr in (CONF_FLOOR,) + THRESHOLDS:
        res[f'recall@{thr}'] = {c: tot[f'hit@{thr}'][c] / max(1, tot['gt'][c])
                                for c in ('open', 'partly occluded')}
        res[f'fa_debris_per_frame@{thr}'] = tot[f'fa_debris@{thr}'] / tot['frames']
        res[f'fa_other_per_frame@{thr}'] = tot[f'fa_other@{thr}'] / tot['frames']
    res['frames'], res['gt'] = tot['frames'], tot['gt']
    with open(os.path.join(root, f'eval_{tag}_{split}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))
    return res


# ----------------------------------------------------------------------------- finetune
def finetune(root: str, base_model: str, device: str = '0', epochs: int | None = None) -> str:
    import torch
    from ultralytics import YOLO
    torch.multiprocessing.set_start_method('spawn', force=True)   # no fork-after-threads deadlock
    ck = torch.load(base_model, map_location='cpu', weights_only=False)
    args = dict(ck['train_args'])
    keep = ['optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'batch', 'imgsz', 'seed',
            'deterministic', 'cos_lr', 'warmup_epochs', 'close_mosaic', 'mosaic', 'hsv_h',
            'hsv_s', 'hsv_v', 'fliplr', 'flipud', 'degrees', 'scale', 'translate', 'mixup',
            'amp', 'epochs']
    train = {k: args[k] for k in keep if k in args}
    if epochs is not None:
        train['epochs'] = epochs
    train['patience'] = train['epochs'] + 1            # fixed number of epochs
    with open(os.path.join(root, 'finetune_args.json'), 'w') as f:
        json.dump({'base_model': base_model, **train}, f, indent=1)
    YOLO(base_model).train(data=os.path.join(root, 'dataset', 'data.yaml'), device=device,
                           project=os.path.join(root, 'finetune'), name='yolo26n_rendered',
                           exist_ok=True, plots=False, verbose=False, **train)
    best = os.path.join(root, 'finetune', 'yolo26n_rendered', 'weights', 'best.pt')
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'eval', 'finetune'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--train', type=int, nargs='*', default=[])
    ap.add_argument('--val', type=int, nargs='*', default=[])
    ap.add_argument('--test', type=int, nargs='*', default=[])
    ap.add_argument('--model')
    ap.add_argument('--split', default='test')
    ap.add_argument('--tag')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--device', default='0')
    a = ap.parse_args()
    splits = {'train': set(a.train), 'val': set(a.val), 'test': set(a.test)}
    for s in set().union(*splits.values()):
        split_of(s, splits)
    if a.cmd == 'build':
        build(a.root, splits)
    elif a.cmd == 'eval':
        evaluate(a.root, a.model, a.split, splits, a.tag, a.device)
    else:
        print(finetune(a.root, a.model, a.device, a.epochs))


if __name__ == '__main__':
    main()
