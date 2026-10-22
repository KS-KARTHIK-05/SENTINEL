"""2.5-D disaster world: buildings, rubble, moving debris, victims, warm clutter.

Tall structures (height above the flight altitude) obstruct UAV-to-UAV links;
low rubble only occludes ground targets (handled by the detector model).
Moving debris (e.g. drifting smoke/dust plumes or collapsing structures)
obstructs links while tall and fully occludes targets underneath it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import shapely
from shapely import STRtree
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from sentinel_core.detector_model import Target


@dataclass
class Obstacle:
    poly: Polygon
    height: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def moving(self) -> bool:
        return bool(np.any(self.velocity))


@dataclass
class World:
    size: tuple
    obstacles: List[Obstacle]
    targets: List[Target]
    clutter_xy: np.ndarray
    base_xy: Optional[np.ndarray]
    flight_alt: float = 30.0
    nofly_buffer: float = 3.0
    _tree: Optional[STRtree] = None
    _tall: Optional[list] = None
    _buffered: Optional[dict] = None

    def __post_init__(self):
        self._rebuild()

    def _rebuild(self):
        self._moving = [o for o in self.obstacles if o.moving]
        self._tall = [o.poly for o in self.obstacles if o.height > self.flight_alt]
        self._tree = STRtree(self._tall) if self._tall else None
        self._buffered = {}
        self._nofly = (shapely.union_all([p.buffer(self.nofly_buffer) for p in self._tall])
                       if self._tall else None)
        if self._nofly is not None:
            shapely.prepare(self._nofly)

    @property
    def bounds(self):
        return np.zeros(2), np.asarray(self.size, float)

    def step(self, dt: float) -> None:
        moved = False
        w, h = self.size
        for o in self._moving:
            o.poly = translate(o.poly, *(o.velocity * dt))
            x0, y0, x1, y1 = o.poly.bounds
            if x0 < 0 or x1 > w:
                o.velocity[0] *= -1
            if y0 < 0 or y1 > h:
                o.velocity[1] *= -1
            moved = True
        if moved:
            self._rebuild()

    def in_no_fly(self, pts: np.ndarray) -> np.ndarray:
        """True for points inside a (buffered) structure taller than the flight altitude."""
        pts = np.atleast_2d(np.asarray(pts, float))
        if self._nofly is None:
            return np.zeros(len(pts), dtype=bool)
        return shapely.contains_xy(self._nofly, pts[:, 0], pts[:, 1])

    def _tree_for(self, buffer: float):
        if buffer <= 0.0:
            return self._tree
        key = round(float(buffer), 3)
        if key not in self._buffered:
            self._buffered[key] = STRtree([p.buffer(buffer) for p in self._tall])
        return self._buffered[key]

    def obstructions(self, a: np.ndarray, b: np.ndarray, buffer: float = 0.0) -> np.ndarray:
        """Number of tall obstacles crossing each segment a[k]-b[k] (b may be one point).

        ``buffer`` inflates every obstacle, giving a conservative count that is
        robust to small motions of either endpoint.
        """
        a = np.atleast_2d(np.asarray(a, float))
        b = np.broadcast_to(np.asarray(b, float), a.shape)
        if self._tree is None or len(a) == 0:
            return np.zeros(len(a), dtype=int)
        lines = shapely.linestrings(np.stack([a, b], axis=1))
        idx = self._tree_for(buffer).query(lines, predicate='intersects')
        return np.bincount(idx[0], minlength=len(a)).astype(int)

    def obstruction_matrix(self, pos: np.ndarray) -> np.ndarray:
        n = len(pos)
        iu = np.triu_indices(n, 1)
        out = np.zeros((n, n), dtype=int)
        if n > 1:
            c = self.obstructions(pos[iu[0]], pos[iu[1]])
            out[iu] = c
            out[(iu[1], iu[0])] = c
        return out

    def target_hidden_by_debris(self, k: int) -> bool:
        if not self._moving:
            return False
        p = shapely.points(self.targets[k].pos)
        return any(o.poly.contains(p) for o in self._moving)


# --------------------------------------------------------------- generators
def _rand_rect(rng, cx, cy, wmin, wmax):
    w, h = rng.uniform(wmin, wmax, 2)
    return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def generate(size: float, n_targets: int, rng: np.random.Generator, n_buildings: int = None,
             n_rubble: int = None, n_moving: int = 0, buried: int = 1, base: bool = True,
             flight_alt: float = 30.0, clutter_per_rubble: int = 2) -> World:
    """Procedural disaster map. Densities scale with area unless given."""
    area_scale = (size / 100.0) ** 2
    # tall shells: absolute footprints of 10-25 m (about 12% of the area at the
    # default density of 4 per hectare); small maps scale the footprint down
    n_buildings = max(1, round(4 * area_scale)) if n_buildings is None else n_buildings
    b_min, b_max = min(10.0, 0.2 * size), min(25.0, 0.35 * size)
    n_rubble = max(3, round(15 * area_scale)) if n_rubble is None else n_rubble
    margin = 0.1 * size
    obstacles = []
    for _ in range(n_buildings):
        c = rng.uniform(margin, size - margin, 2)
        obstacles.append(Obstacle(_rand_rect(rng, *c, b_min, b_max),
                                  float(rng.uniform(flight_alt + 5, flight_alt + 30))))
    rubble = []
    for _ in range(n_rubble):
        c = rng.uniform(0.05 * size, 0.95 * size, 2)
        r = _rand_rect(rng, *c, 0.02 * size, 0.05 * size)
        rubble.append(r)
        obstacles.append(Obstacle(r, float(rng.uniform(1.0, 5.0))))
    for _ in range(n_moving):
        c = rng.uniform(margin, size - margin, 2)
        v = rng.normal(0, 1, 2)
        v = v / np.linalg.norm(v) * rng.uniform(0.5, 2.0)
        obstacles.append(Obstacle(_rand_rect(rng, *c, 0.05 * size, 0.1 * size),
                                  float(rng.uniform(flight_alt + 5, flight_alt + 20)), v))
    tall = [o.poly for o in obstacles if o.height > flight_alt]
    targets = []
    tries = 0
    while len(targets) < n_targets and tries < 10000:
        tries += 1
        p = rng.uniform(0.05 * size, 0.95 * size, 2)
        if any(t.contains(shapely.points(p)) for t in tall):
            continue
        if any(np.linalg.norm(p - t.pos) < max(4.0, 0.03 * size) for t in targets):
            continue
        is_buried = len(targets) < buried
        occ0 = 1.0 if is_buried else float(rng.choice([0.0, 0.0, 0.2, 0.45, 0.6]))
        targets.append(Target(p, occ0, float(rng.uniform(0, 2 * np.pi)), is_buried))
    clutter = []
    for r in rubble:
        c = np.asarray(r.centroid.coords[0])
        for _ in range(clutter_per_rubble):
            clutter.append(c + rng.normal(0, 1.0, 2))
    base_xy = np.array([0.04 * size, 0.04 * size]) if base else None
    return World((size, size), obstacles, targets, np.asarray(clutter).reshape(-1, 2),
                 base_xy, flight_alt)


def disaster_25(rng: np.random.Generator, base: bool = False) -> World:
    """Rebuild of the SENTINEL v1 25 x 25 m scenario: 10 victims, one under a slab."""
    return generate(25.0, 10, rng, n_buildings=2, n_rubble=6, buried=1, base=base,
                    clutter_per_rubble=1)


def from_truth(truth: dict) -> World:
    """Rebuild a World from the ground-truth JSON written by make_world.py."""
    obstacles = []
    for o in truth['obstacles']:
        (cx, cy), (sx, sy) = o['center'], o['size']
        obstacles.append(Obstacle(box(cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2),
                                  float(o['height'])))
    targets = [Target(np.asarray(t['pos'], float), float(t['occ0']), float(t['block_dir']),
                      bool(t['buried'])) for t in truth['targets']]
    base = None if truth.get('base_xy') is None else np.asarray(truth['base_xy'], float)
    return World(tuple(truth['size']), obstacles, targets,
                 np.asarray(truth.get('clutter', []), float).reshape(-1, 2), base,
                 float(truth.get('flight_alt', 30.0)))
