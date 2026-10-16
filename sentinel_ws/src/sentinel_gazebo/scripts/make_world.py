#!/usr/bin/env python3
"""Generate a Gazebo Harmonic disaster world from the fast-simulator generator.

The same (kind, size, seed) produces the same buildings, rubble, victims and
warm clutter as ``sentinel_sim.world``, so SITL runs cross-validate the fast
simulator on identical scenes (experiment E8). A ground-truth JSON is written
next to the SDF.

    python make_world.py --kind disaster_25 --seed 0 --out ../worlds
    python make_world.py --kind generate --size 100 --targets 20 --seed 3 --out ../worlds
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from sentinel_sim import world as W

# Thermal scene (K). Atmosphere/ambient ~288 K; victims 310.15 K via the heat
# signature of Rescue Randy; warm clutter 316 K (see models/warm_debris).
T_GROUND, T_BUILDING, T_RUBBLE, T_SLAB = 293.0, 291.0, 285.0, 289.0

HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9">
  <world name="{name}">
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"><temperature>288.15</temperature></atmosphere>
    <scene>
      <grid>false</grid>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>
    <light name="sunUTC" type="directional">
      <pose>0 0 500 0 0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>0.001 0.625 -0.78</direction>
      <diffuse>0.904 0.904 0.904 1</diffuse>
      <specular>0.271 0.271 0.271 1</specular>
    </light>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>488</elevation>
    </spherical_coordinates>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>{g} {g}</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>{g} {g}</size></plane></geometry>
          <material><ambient>0.5 0.5 0.45 1</ambient><diffuse>0.5 0.5 0.45 1</diffuse></material>
          {thermal_ground}
        </visual>
      </link>
    </model>
"""


def thermal(temp: float) -> str:
    return ('<plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal">'
            f'<temperature>{temp}</temperature></plugin>')


def box(name, x, y, z, sx, sy, sz, temp, rgb, yaw=0.0) -> str:
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry></collision>
        <visual name="v">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material>
          {thermal(temp)}
        </visual>
      </link>
    </model>
"""


def include(name, uri, x, y, z=0.0, yaw=0.0) -> str:
    return (f'    <include><name>{name}</name><uri>model://{uri}</uri>'
            f'<pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose></include>\n')


def build(w: W.World, name: str, rng: np.random.Generator) -> tuple[str, dict]:
    size = w.size[0]
    parts = [HEADER.format(name=name, g=int(size + 200), thermal_ground=thermal(T_GROUND))]
    obstacles = []
    for k, o in enumerate(w.obstacles):
        x0, y0, x1, y1 = o.poly.bounds
        cx, cy, sx, sy = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        tall = o.height > w.flight_alt
        parts.append(box(f'{"building" if tall else "rubble"}_{k}', cx, cy, o.height / 2, sx, sy,
                         o.height, T_BUILDING if tall else T_RUBBLE,
                         '0.55 0.55 0.55' if tall else '0.45 0.4 0.35'))
        obstacles.append({'center': [cx, cy], 'size': [sx, sy], 'height': o.height,
                          'moving': o.moving})
    targets = []
    for k, t in enumerate(w.targets):
        x, y = map(float, t.pos)
        yaw = float(rng.uniform(-math.pi, math.pi))
        parts.append(include(f'victim_{k}', 'rescue_randy_thermal', x, y, 0.0, yaw))
        if t.buried:
            # full concrete slab on four low supports: no line of sight from above
            parts.append(box(f'slab_{k}', x, y, 1.15, 3.0, 3.0, 0.3, T_SLAB, '0.6 0.6 0.6'))
        elif t.occ0 > 0:
            # overhanging slab displaced toward block_dir, covering ~occ0 of the body
            off = (1.0 - t.occ0) * 1.6
            parts.append(box(f'slab_{k}', x + off * math.cos(t.block_dir),
                             y + off * math.sin(t.block_dir), 1.2, 2.2, 2.2, 0.25, T_SLAB,
                             '0.6 0.6 0.6', t.block_dir))
        targets.append({'id': k, 'pos': [x, y], 'occ0': t.occ0, 'block_dir': t.block_dir,
                        'buried': t.buried})
    for k, c in enumerate(w.clutter_xy):
        parts.append(include(f'warm_debris_{k}', 'warm_debris', float(c[0]), float(c[1]), 0.15,
                             float(rng.uniform(-math.pi, math.pi))))
    parts.append('  </world>\n</sdf>\n')
    truth = {'world': name, 'size': list(w.size), 'flight_alt': w.flight_alt,
             'base_xy': None if w.base_xy is None else list(map(float, w.base_xy)),
             'targets': targets, 'obstacles': obstacles,
             'clutter': [list(map(float, c)) for c in w.clutter_xy]}
    return ''.join(parts), truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='generate', choices=['generate', 'disaster_25'])
    ap.add_argument('--size', type=float, default=100.0)
    ap.add_argument('--targets', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--base', action='store_true', help='include a responder node position')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'worlds'))
    a = ap.parse_args()
    # identical stream derivation to sentinel_sim.sim.Simulation (world = first child)
    r_world = np.random.default_rng(np.random.SeedSequence(a.seed).spawn(5)[0])
    if a.kind == 'disaster_25':
        w = W.disaster_25(r_world, base=a.base)
        name = f'sentinel_d25_s{a.seed}'
    else:
        w = W.generate(a.size, a.targets, r_world, base=a.base)
        name = f'sentinel_g{int(a.size)}_s{a.seed}'
    sdf, truth = build(w, name, np.random.default_rng(a.seed + 1))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, name + '.sdf'), 'w') as f:
        f.write(sdf)
    with open(os.path.join(a.out, name + '.json'), 'w') as f:
        json.dump(truth, f, indent=1)
    print(os.path.join(a.out, name + '.sdf'))


if __name__ == '__main__':
    main()
