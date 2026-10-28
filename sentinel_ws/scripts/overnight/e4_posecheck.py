#!/usr/bin/env python3
"""Audit of the rendered E4 frames: every frame's boxes must match its recorded camera pose
(``e4_capture.pose_consistent``). Exit 1 and list the frames if any world has a mismatch.

    python e4_posecheck.py --root <E4 folder> --seeds 100 101 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e4_capture import pose_consistent  # noqa: E402


def check_world(root: str, name: str) -> tuple[int, list[int]]:
    truth = json.load(open(os.path.join(root, 'truth', name + '.json')))
    n, bad = 0, []
    for line in open(os.path.join(root, 'meta', name + '.jsonl')):
        m = json.loads(line)
        n += 1
        if not pose_consistent(m['full'], (m['x'], m['y'], m['z'], m['yaw']), truth['targets'],
                               truth['clutter']):
            bad.append(m['k'])
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--size', type=int, default=100)
    a = ap.parse_args()
    total, bad_total = 0, 0
    for s in a.seeds:
        name = f'e4_g{a.size}_s{s}'
        n, bad = check_world(a.root, name)
        total += n
        bad_total += len(bad)
        if bad:
            print(f'{name}: {len(bad)} of {n} frames do not match their pose: {bad}')
    print(f'pose check: {total} frames, {bad_total} mismatched')
    sys.exit(1 if bad_total else 0)


if __name__ == '__main__':
    main()
