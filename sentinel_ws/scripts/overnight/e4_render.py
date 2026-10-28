#!/usr/bin/env python3
"""E4: render thermal frames with automatic ground truth at the operating altitude (25 and 30 m).

For every world seed a disaster world is generated with the same generator as the PX4/Gazebo
cross-check (``make_world.build``, 100 m, 20 victims). A static camera rig carries the on-board
thermal camera (same intrinsics as ``x500_thermal``: 640 x 512, 50 deg HFOV, L16, nadir) and two
bounding-box cameras with the same intrinsics: ``2d`` (visible part of each labelled object) and
``full_2d`` (full extent, occlusion ignored). Victims carry Gazebo labels 1..20 (victim id + 1),
warm debris label 200. The rig is moved with the ``set_pose`` service to viewpoints over each
victim and over random and debris locations; each saved frame is the first thermal image whose
simulation stamp is at least 0.3 s after the pose change, with the box messages of the same
stamp. Frames are converted to 8-bit white-hot exactly as ``thermal_vision`` does
(0.01 K per count, 270-330 K).

    python e4_render.py --seeds 100 101 --out ~/sentinel_ws/results/overnight_2026-10-28/e4_frames

Output per world: ``frames/<world>/<k>.png`` and ``meta/<world>.jsonl`` (camera pose, boxes) plus
``truth/<world>.json`` (victims with occ0 / block_dir / buried, debris, obstacles). Worlds that
already have a complete meta file are skipped (resumable).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

from e4_capture import quat, to_white_hot  # noqa: F401 (re-exported for tests)




HERE = os.path.dirname(os.path.abspath(__file__))
GZ_PKG = os.path.normpath(os.path.join(HERE, '..', '..', 'src', 'sentinel_gazebo'))
VICTIM_LABEL0, DEBRIS_LABEL = 1, 200
HFOV, W_IMG, H_IMG, PITCH = 0.8727, 640, 512, 1.5707     # identical to models/x500_thermal
T_MIN, T_MAX, K_PER_COUNT = 270.0, 330.0, 0.01            # identical to thermal_vision
SETTLE_S = 0.3
CAPTURE_PY = '/usr/bin/python3'


def _make_world_module():
    spec = importlib.util.spec_from_file_location(
        'make_world', os.path.join(GZ_PKG, 'scripts', 'make_world.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SYSTEMS = """    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
"""


def _camera(fmt: str = '', box: str = '') -> str:
    return (f'<camera>{f"<box_type>{box}</box_type>" if box else ""}'
            f'<horizontal_fov>{HFOV}</horizontal_fov><image><width>{W_IMG}</width>'
            f'<height>{H_IMG}</height>{f"<format>{fmt}</format>" if fmt else ""}</image>'
            '<clip><near>0.5</near><far>200</far></clip></camera>')


RIG = f"""    <model name="rig">
      <static>true</static>
      <pose>0 0 30 0 {PITCH} 0</pose>
      <link name="link">
        <sensor name="thermal" type="thermal"><topic>/rig/thermal</topic>{_camera(fmt='L16')}
          <always_on>1</always_on><update_rate>10</update_rate><visualize>false</visualize></sensor>
        <sensor name="bbox_vis" type="boundingbox_camera"><topic>/rig/bbox_vis</topic>{_camera(box='2d')}
          <always_on>1</always_on><update_rate>10</update_rate><visualize>false</visualize></sensor>
        <sensor name="bbox_full" type="boundingbox_camera"><topic>/rig/bbox_full</topic>{_camera(box='full_2d')}
          <always_on>1</always_on><update_rate>10</update_rate><visualize>false</visualize></sensor>
      </link>
    </model>
"""


def _label(text: str, prefix: str, label_of) -> str:
    def sub(m):
        lab = label_of(int(m.group(2)))
        return (f'{m.group(1)}<plugin filename="gz-sim-label-system" '
                f'name="gz::sim::systems::Label"><label>{lab}</label></plugin></include>')
    return re.sub(rf'(<include><name>{prefix}_(\d+)</name>.*?)</include>', sub, text)


def rig_world(sdf: str, name: str) -> str:
    """Add systems, labels and the camera rig to a ``make_world`` SDF."""
    sdf = sdf.replace(f'<world name="{name}">', f'<world name="{name}">\n{SYSTEMS}', 1)
    sdf = _label(sdf, 'victim', lambda k: VICTIM_LABEL0 + k)
    sdf = _label(sdf, 'warm_debris', lambda k: DEBRIS_LABEL)
    return sdf.replace('  </world>', RIG + '  </world>', 1)


def viewpoints(world, seed: int, alts=(25.0, 30.0), per_victim=3, n_random=10, n_debris=10,
               r_off=9.0):
    """Camera poses (x, y, z, yaw): over victims, debris and random spots; never in buildings."""
    rng = np.random.default_rng(seed + 12345)
    lo, hi = world.bounds

    def disk():
        r, a = r_off * math.sqrt(rng.random()), 2 * math.pi * rng.random()
        return np.array([r * math.cos(a), r * math.sin(a)])

    cand = []
    for t in world.targets:
        if t.buried:
            continue
        for z in alts:
            for _ in range(per_victim):
                cand.append((t.pos + disk(), z))
    clutter = np.asarray(world.clutter_xy).reshape(-1, 2)
    for _ in range(n_debris if len(clutter) else 0):
        cand.append((clutter[rng.integers(len(clutter))] + disk(), float(rng.choice(alts))))
    for _ in range(n_random):
        cand.append((lo + rng.random(2) * (hi - lo), float(rng.choice(alts))))
    out = []
    for xy, z in cand:
        xy = np.clip(xy, lo, hi)
        ring = xy + 2.0 * np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
        if np.any(world.in_no_fly(ring)):
            continue
        out.append((float(xy[0]), float(xy[1]), float(z), float(rng.uniform(-math.pi, math.pi))))
    return out


def render_world(mw, seed: int, out: str, size: float, n_targets: int, env: dict) -> int:
    from sentinel_sim import world as W
    name = f'e4_g{int(size)}_s{seed}'
    meta_path = os.path.join(out, 'meta', name + '.jsonl')
    done_flag = meta_path + '.done'
    if os.path.exists(done_flag):
        return -1
    r_world = np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[0])
    w = W.generate(size, n_targets, r_world, base=False)
    sdf, truth = mw.build(w, name, np.random.default_rng(seed + 1))
    for d in ('worlds', 'meta', 'truth', os.path.join('frames', name)):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    wpath = os.path.join(out, 'worlds', name + '.sdf')
    with open(wpath, 'w') as f:
        f.write(rig_world(sdf, name))
    with open(os.path.join(out, 'truth', name + '.json'), 'w') as f:
        json.dump(truth, f, indent=1)
    vps_path = os.path.join(out, 'worlds', name + '.vps.json')
    with open(vps_path, 'w') as f:
        json.dump(viewpoints(w, seed), f)
    # capture loop in the system Python (native protobuf matching gz.msgs10), see e4_capture.py
    r = subprocess.run([CAPTURE_PY, '-s', os.path.join(HERE, 'e4_capture.py'), '--name', name,
                        '--sdf', wpath, '--vps', vps_path, '--out', out,
                        '--partition', env['GZ_PARTITION'],
                        '--truth', os.path.join(out, 'truth', name + '.json')], env=env,
                       capture_output=True, text=True,
                       timeout=3600)
    print(r.stdout.strip(), flush=True)
    if r.returncode != 0 or not os.path.exists(done_flag):
        raise RuntimeError(f'{name}: capture failed ({r.stdout.strip()[-200:]} {r.stderr.strip()[-200:]})')
    return sum(1 for _ in open(meta_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=float, default=100.0)
    ap.add_argument('--targets', type=int, default=20)
    ap.add_argument('--partition', default='sentinel_e4')
    a = ap.parse_args()
    env = dict(os.environ)
    env['GZ_SIM_RESOURCE_PATH'] = os.path.join(GZ_PKG, 'models') + ':' + env.get(
        'GZ_SIM_RESOURCE_PATH', '')
    env['GZ_PARTITION'] = a.partition
    os.environ['GZ_PARTITION'] = a.partition          # for this process's transport node
    mw = _make_world_module()
    for s in a.seeds:
        t0 = time.time()
        for attempt in range(1, 4):     # transient transport timeouts: restart Gazebo, redo the world
            try:
                n = render_world(mw, s, a.out, a.size, a.targets, env)
                break
            except RuntimeError as e:
                print(f'world seed {s}: attempt {attempt} failed ({e})', flush=True)
                if attempt == 3:
                    raise
                time.sleep(5)
        msg = 'skipped (done)' if n < 0 else f'{n} frames in {time.time() - t0:.0f} s'
        print(f'world seed {s}: {msg}', flush=True)


if __name__ == '__main__':
    try:
        main()
        code = 0
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)          # skip gz-transport destructors (they can abort at interpreter exit)
