#!/usr/bin/python3 -s
"""E4 capture loop for one rendered world, run with the system Python (native protobuf 3.12, which
matches the gz.msgs10 bindings; the venv's pure-Python protobuf parser falls behind on three 10 Hz
streams under full CPU load and the in-process transport then stalls). Started by ``e4_render.py``:

    /usr/bin/python3 -s e4_capture.py --name W --sdf W.sdf --vps vps.json --out OUT --partition P

Starts ``gz sim`` on the rig world, captures every viewpoint (see ``e4_render.py``) and writes
``frames/W/<k>.png``, ``meta/W.jsonl`` and ``meta/W.jsonl.done``. Exit 0 on success.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

import cv2
import numpy as np

PITCH = 1.5707
T_MIN, T_MAX, K_PER_COUNT = 270.0, 330.0, 0.01
SETTLE_S = 0.3
HFOV, W_IMG, H_IMG = 0.8727, 640, 512
TOL_PX, BORDER_TOL_PX, DEBRIS = 15.0, 40.0, 200
RETAKES = 6


def to_white_hot(l16: np.ndarray) -> np.ndarray:
    temp = l16.astype(np.float32) * K_PER_COUNT
    return np.clip((temp - T_MIN) / (T_MAX - T_MIN) * 255.0, 0, 255).astype(np.uint8)


def quat(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)   # w, x, y, z


def project(x: float, y: float, z: float, yaw: float, px: float, py: float):
    """Pixel (u, v) of ground point (px, py) for the nadir camera at (x, y, z) with heading yaw
    (image up = heading, image left = heading + 90 deg)."""
    f = (W_IMG / 2) / math.tan(HFOV / 2)
    dx, dy = px - x, py - y
    a = dx * math.cos(yaw) + dy * math.sin(yaw)
    b = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return W_IMG / 2 - f * b / z, H_IMG / 2 - f * a / z


def pose_consistent(full: list, pose, targets: list, clutter: list) -> bool:
    """True if the frame shows the commanded pose: every box clear of the image border lies within
    TOL_PX of where its object projects from that pose (a victim by its label, a debris item by the
    nearest one), and a border-clipped victim box lies within BORDER_TOL_PX of its projection.
    Catches frames rendered before a pose change took effect, which carry the previous viewpoint's
    image and boxes. (Presence is not checked: ``full_2d`` omits victims hidden under a slab.)"""
    x, y, z, yaw = pose
    debris = [project(x, y, z, yaw, *c) for c in clutter]
    for lab, x0, y0, x1, y1 in full:
        lab = int(lab)
        clear = x0 > 1 and y0 > 1 and x1 < W_IMG - 2 and y1 < H_IMG - 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if lab == DEBRIS:
            if clear and min((math.hypot(u - cx, v - cy) for u, v in debris), default=1e9) > TOL_PX:
                return False
            continue
        if not 1 <= lab <= len(targets):
            return False
        u, v = project(x, y, z, yaw, *targets[lab - 1]['pos'])
        if clear:
            if math.hypot(u - cx, v - cy) > TOL_PX:
                return False
        elif not (x0 - BORDER_TOL_PX <= u <= x1 + BORDER_TOL_PX
                  and y0 - BORDER_TOL_PX <= v <= y1 + BORDER_TOL_PX):
            return False
    return True


def boxes(msg) -> list:
    return [[int(b.label), float(b.box.min_corner.x), float(b.box.min_corner.y),
             float(b.box.max_corner.x), float(b.box.max_corner.y)] for b in msg.annotated_box]


class Capture:
    def __init__(self, world_name: str):
        from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
        from gz.msgs10.image_pb2 import Image
        from gz.transport13 import Node
        self.node, self.world = Node(), world_name
        self.img, self.vis, self.full = {}, {}, {}
        self.latest = -1.0
        self.cli_fallbacks = 0
        self.use_cli, self.n_ok = False, 0

        def stamp(m):
            return m.header.stamp.sec + 1e-9 * m.header.stamp.nsec

        def on_img(m):
            s = stamp(m)
            self.img[s] = m
            self.latest = max(self.latest, s)
        self.node.subscribe(Image, '/rig/thermal', on_img)
        self.node.subscribe(AnnotatedAxisAligned2DBox_V, '/rig/bbox_vis',
                            lambda m: self.vis.__setitem__(stamp(m), m))
        self.node.subscribe(AnnotatedAxisAligned2DBox_V, '/rig/bbox_full',
                            lambda m: self.full.__setitem__(stamp(m), m))

    def wait_ready(self, timeout=90.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest > 0 and self.vis and self.full:
                return True
            time.sleep(0.2)
        return False

    def set_pose(self, x, y, z, yaw) -> bool:
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_pb2 import Pose
        p = Pose()
        p.name = 'rig'
        p.position.x, p.position.y, p.position.z = x, y, z
        w, qx, qy, qz = quat(0.0, PITCH, yaw)
        p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = w, qx, qy, qz
        tries = 0 if self.use_cli else (8 if self.n_ok == 0 else 2)   # discovery delay on the first call
        for _ in range(tries):
            ok, rep = self.node.request(f'/world/{self.world}/set_pose', p, Pose, Boolean, 3000)
            if ok and rep.data:
                self.n_ok += 1
                return True
            time.sleep(0.5)
        self.use_cli = True          # once the in-process request stalls it stays stalled: CLI from now on
        # the in-process request occasionally stops answering under heavy CPU load: same call via the CLI
        req = (f'name: "rig", position: {{x: {x}, y: {y}, z: {z}}}, '
               f'orientation: {{w: {w}, x: {qx}, y: {qy}, z: {qz}}}')
        for _ in range(5):
            out = subprocess.run(['gz', 'service', '-s', f'/world/{self.world}/set_pose', '--reqtype',
                                  'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean', '--timeout', '5000',
                                  '--req', req], capture_output=True, text=True, timeout=30)
            if 'data: true' in out.stdout:
                self.cli_fallbacks += 1
                return True
            time.sleep(1.0)
        return False

    def grab(self, timeout=10.0):
        s0 = self.latest
        t0 = time.time()
        while time.time() - t0 < timeout:
            ready = sorted(s for s in self.img if s >= s0 + SETTLE_S and s in self.vis
                           and s in self.full)
            if ready:
                s = ready[0]
                m = self.img[s]
                out = (s, m, self.vis[s], self.full[s])
                for d in (self.img, self.vis, self.full):      # drop old messages
                    for k in [k for k in d if k <= s]:
                        d.pop(k, None)
                return out
            time.sleep(0.02)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--sdf', required=True)
    ap.add_argument('--vps', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--partition', default='sentinel_e4')
    ap.add_argument('--truth', required=True)
    a = ap.parse_args()
    os.environ['GZ_PARTITION'] = a.partition
    vps = json.load(open(a.vps))
    truth = json.load(open(a.truth))
    targets, clutter = truth['targets'], truth['clutter']
    retaken = 0
    meta_path = os.path.join(a.out, 'meta', a.name + '.jsonl')
    log = open(os.path.join(a.out, 'worlds', a.name + '.gz.log'), 'w')
    gz = subprocess.Popen(['gz', 'sim', '-s', '-r', '--headless-rendering', a.sdf], env=os.environ,
                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    code = 1
    try:
        cap = Capture(a.name)
        if not cap.wait_ready():
            raise RuntimeError(f'{a.name}: sensors not publishing')
        with open(meta_path, 'w') as fm:
            for k, (x, y, z, yaw) in enumerate(vps):
                if not cap.set_pose(x, y, z, yaw):
                    raise RuntimeError(f'{a.name}: set_pose failed')
                for r in range(RETAKES + 1):
                    g = cap.grab()
                    if g is None:
                        raise RuntimeError(f'{a.name}: no synchronised frame for viewpoint {k}')
                    if pose_consistent(boxes(g[3]), (x, y, z, yaw), targets, clutter):
                        break
                    retaken += 1                 # stale frame (pose change not yet rendered)
                    if r == RETAKES // 2 and not cap.set_pose(x, y, z, yaw):
                        raise RuntimeError(f'{a.name}: set_pose failed')
                else:
                    raise RuntimeError(f'{a.name}: viewpoint {k} never matched its pose')
                s, img, vis, full = g
                arr = np.frombuffer(img.data, dtype=np.uint16).reshape(img.height, img.width)
                cv2.imwrite(os.path.join(a.out, 'frames', a.name, f'{k:04d}.png'), to_white_hot(arr))
                fm.write(json.dumps({'world': a.name, 'seed': int(a.name.split('_s')[-1]), 'k': k,
                                     'x': x, 'y': y, 'z': z, 'yaw': yaw, 'stamp': s,
                                     'vis': boxes(vis), 'full': boxes(full)}) + '\n')
        open(meta_path + '.done', 'w').close()
        print(f'{a.name}: {len(vps)} frames, {cap.cli_fallbacks} set_pose via CLI, '
              f'{retaken} stale frames retaken', flush=True)
        code = 0
    except Exception as e:  # noqa: BLE001
        print(f'{a.name}: {e}', flush=True)
    finally:
        try:
            os.killpg(gz.pid, signal.SIGTERM)
            gz.wait(timeout=20)
        except Exception:  # noqa: BLE001
            os.killpg(gz.pid, signal.SIGKILL)
        log.close()
    sys.stdout.flush()
    os._exit(code)            # skip gz-transport destructors (they can abort at interpreter exit)


if __name__ == '__main__':
    main()
