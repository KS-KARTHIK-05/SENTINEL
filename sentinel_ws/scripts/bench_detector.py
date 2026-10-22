#!/usr/bin/env python3
"""Perception benchmark for the SENTINEL thermal detector (Phase 5 / R1.5).

For each model:
* accuracy: mAP50, mAP50-95, precision, recall. With ``--data`` they are
  measured on the given split; without it they are read from the checkpoint
  (validation metrics recorded by the training run) and labelled as such;
* size: parameters and GFLOPs at the input size;
* latency on this workstation: pre-process / inference / post-process and
  end-to-end p50/p95 for PyTorch FP32, PyTorch FP16, ONNX Runtime (CUDA) and
  TensorRT FP16 (engine built locally from the ONNX export, because TensorRT
  engines are specific to the GPU and TensorRT version that built them).

    python bench_detector.py --models a.pt b.pt --out results/perception \
        [--data hit_uav.yaml] [--frames-dir dir_with_pngs]

Exports are written inside ``--out`` (model files are copied first), so the
source model folders are never modified. Latency is desktop latency (RTX 3060),
not embedded (Jetson) latency.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import time

import numpy as np
import pandas as pd


def synthetic_frames(n, h=512, w=640, seed=0):
    """Thermal-like 8-bit frames: cool textured background, a few warm blobs."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for _ in range(n):
        img = rng.normal(60, 12, (h, w))
        for _ in range(rng.integers(1, 6)):
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
            s = rng.uniform(4, 14)
            img += rng.uniform(80, 170) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        g = np.clip(img, 0, 255).astype(np.uint8)
        out.append(np.dstack([g, g, g]))
    return out


def load_frames(args, n):
    import cv2
    if args.frames_dir:
        files = sorted(glob.glob(os.path.join(args.frames_dir, '*.png'))
                       + glob.glob(os.path.join(args.frames_dir, '*.jpg')))
        frames = [cv2.imread(f) for f in files]
        if frames:
            return [frames[k % len(frames)] for k in range(n)]
    return synthetic_frames(n)


def latency(model, frames, imgsz, half, warmup=30):
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, half=half, verbose=False)
    tot, pre, inf, post = [], [], [], []
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=imgsz, half=half, verbose=False)[0]
        tot.append((time.perf_counter() - t0) * 1e3)
        pre.append(r.speed['preprocess'])
        inf.append(r.speed['inference'])
        post.append(r.speed['postprocess'])
    q = np.percentile
    return {'pre_ms_p50': q(pre, 50), 'infer_ms_p50': q(inf, 50), 'infer_ms_p95': q(inf, 95),
            'post_ms_p50': q(post, 50), 'total_ms_p50': q(tot, 50), 'total_ms_p95': q(tot, 95),
            'fps_p50': 1000.0 / q(tot, 50)}


def accuracy(m, path, args):
    import torch
    if args.data:
        v = m.val(data=args.data, split=args.split, imgsz=args.imgsz, verbose=False, plots=False)
        return {'mAP50': float(v.box.map50), 'mAP50_95': float(v.box.map),
                'precision': float(v.box.mp), 'recall': float(v.box.mr),
                'acc_source': f'measured ({args.split} split)'}
    ck = torch.load(path, map_location='cpu', weights_only=False)
    tm = ck.get('train_metrics') or {}
    ta = ck.get('train_args') or {}
    return {'mAP50': tm.get('metrics/mAP50(B)'), 'mAP50_95': tm.get('metrics/mAP50-95(B)'),
            'precision': tm.get('metrics/precision(B)'), 'recall': tm.get('metrics/recall(B)'),
            'acc_source': f"checkpoint val metrics (data={ta.get('data')}, epochs={ta.get('epochs')}, "
                          f"optimizer={ta.get('optimizer')}, seed={ta.get('seed')})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--data', default='', help='dataset yaml; omit to use checkpoint metrics')
    ap.add_argument('--split', default='test')
    ap.add_argument('--frames-dir', default='')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--backends', nargs='+', default=['torch', 'torch_fp16', 'onnx', 'engine'])
    ap.add_argument('--out', default='results/perception')
    a = ap.parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    os.makedirs(a.out, exist_ok=True)
    frames = load_frames(a, a.frames)
    rows = []
    for src in a.models:
        name = os.path.splitext(os.path.basename(src))[0]
        work = os.path.join(a.out, 'models', name)
        os.makedirs(work, exist_ok=True)
        path = os.path.join(work, os.path.basename(src))
        shutil.copy2(src, path)
        m = YOLO(path)
        base = {'model': name, 'params_M': sum(p.numel() for p in m.model.parameters()) / 1e6,
                'gflops': get_flops(m.model, a.imgsz), **accuracy(m, path, a)}
        for be in a.backends:
            try:
                if be in ('torch', 'torch_fp16'):
                    lat = latency(m, frames, a.imgsz, half=be == 'torch_fp16')
                elif be == 'onnx':
                    onnx = m.export(format='onnx', imgsz=a.imgsz, dynamic=False, verbose=False)
                    lat = latency(YOLO(onnx, task='detect'), frames, a.imgsz, False)
                elif be == 'engine':
                    eng = m.export(format='engine', imgsz=a.imgsz, half=True, verbose=False)
                    lat = latency(YOLO(eng, task='detect'), frames, a.imgsz, True)
                else:
                    continue
                rows.append({**base, 'backend': be, **lat})
            except Exception as e:  # noqa: BLE001 - a missing backend must not stop the run
                rows.append({**base, 'backend': be, 'error': str(e).splitlines()[0][:200]})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'perception.csv'), index=False)
    cols = [c for c in ['model', 'backend', 'params_M', 'gflops', 'mAP50', 'mAP50_95', 'precision',
                        'recall', 'pre_ms_p50', 'infer_ms_p50', 'post_ms_p50', 'total_ms_p50',
                        'total_ms_p95', 'fps_p50', 'error'] if c in df]
    with open(os.path.join(a.out, 'perception.md'), 'w') as f:
        f.write(df[cols].round(3).to_markdown(index=False))
        f.write('\n\nAccuracy source: ' + '; '.join(sorted(set(df.acc_source))) + '\n')
        f.write('Frames: ' + (a.frames_dir or 'synthetic thermal-like 640x512') + f', n={a.frames}\n')
    print(df[cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
