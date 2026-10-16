"""Frame conventions.

* World: Gazebo ENU (x east, y north, z up); identical to the fast simulator.
* PX4 local: NED centred at the vehicle's spawn point (x north, y east, z down).
* Camera: nadir pinhole, image u to the right, v down. ``cam_axes`` maps the
  image axes to the body frame (forward, right) and is verified in SITL
  (scripts/check_geolocation.py) because it depends on the sensor mounting.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def ned_to_world(spawn_xy, ned_xyz) -> Tuple[float, float, float]:
    n, e, d = ned_xyz
    return spawn_xy[0] + e, spawn_xy[1] + n, -d


def world_to_ned(spawn_xy, x, y, alt) -> Tuple[float, float, float]:
    return y - spawn_xy[1], x - spawn_xy[0], -alt


def yaw_from_quat_ned(q) -> float:
    """Heading (rad, from north, clockwise) from a PX4 (w, x, y, z) quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pixel_to_world(u, v, width, height, hfov, uav_world_xyz, yaw_ned,
                   cam_axes=(-1.0, 1.0)) -> Tuple[float, float, float]:
    """Flat-ground geolocation of an image point seen by a level nadir camera.

    Returns (x, y, slant_range) in the world ENU frame. ``cam_axes`` = (sign of
    -v along body-forward, sign of u along body-right).
    """
    f = (width / 2.0) / math.tan(hfov / 2.0)
    h = max(uav_world_xyz[2], 0.1)
    fwd = cam_axes[0] * (v - height / 2.0) / f * h
    right = cam_axes[1] * (u - width / 2.0) / f * h
    # body (forward, right) -> NED (north, east) by heading, then NED -> ENU
    north = fwd * math.cos(yaw_ned) - right * math.sin(yaw_ned)
    east = fwd * math.sin(yaw_ned) + right * math.cos(yaw_ned)
    x = uav_world_xyz[0] + east
    y = uav_world_xyz[1] + north
    return x, y, float(np.sqrt(fwd * fwd + right * right + h * h))
