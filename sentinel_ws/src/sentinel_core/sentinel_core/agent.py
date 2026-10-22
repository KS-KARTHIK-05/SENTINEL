"""Decentralised SENTINEL agent (one instance per UAV).

The agent only uses information it owns or has received in beacons from
neighbours; there is no central coordinator. The same class is driven by the
fast simulator (``sentinel_sim``) and by the ROS 2 node ``emps_agent``.

Per control tick:
  1. refresh the neighbour table from received beacons (stale ones expire),
  2. select the spring/constraint set S_i (LMST, nearest neighbour or none),
  3. every planning period, choose a waypoint by PSO on the EM-PSO fitness
     (or follow an SA flank assignment),
  4. move toward the waypoint at v_max and, if enabled, scale the step back so
     that every link in S_i stays within R_hard (midpoint-ball projection).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import graph
from .channel import ChannelProfile, reliable_range
from .fitness import FitnessContext, SpringConfig, fitness
from .ledger import Entry, Ledger
from .maps import SearchMap
from .pso import PSOConfig, optimise
from .sa_trigger import plan_flank
from .safety import midpoint_balls, project_step, recovery_step


@dataclass
class AgentConfig:
    spring: SpringConfig = field(default_factory=SpringConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    neighbour_set: str = 'lmst'       # lmst | nn | all | none
    projection: bool = True           # hard connectivity constraint
    sa_enabled: bool = True
    fusion: bool = True               # multi-view log-odds fusion (else max view)
    fusion_rule: str = 'corroborative'  # corroborative (A3) | naive_bayes (ablation); set per run
    sa_offset_m: float = 15.0
    sa_timeout_s: float = 20.0
    v_max: float = 5.0                # m/s
    altitude: float = 30.0            # m
    footprint_r: float = 12.0         # m, thermal camera ground footprint radius
    plan_period_s: float = 1.0
    horizon_s: float = 4.0            # waypoint search radius = v_max * horizon
    beacon_stale_s: float = 3.0
    r_hard_eps: float = 0.05          # outage target defining R_hard
    r_hard_nlos: int = 1              # obstructions assumed for the nominal radius R_hard
    obstruction_aware: bool = True    # per-link radius R_eps(n_obs) from the building map
    obs_buffer_m: float = 0.0         # static obstacle inflation for the hard radius
    reconnect: bool = True            # root-heartbeat re-attachment (self-healing)
    root_timeout_s: float = 4.0
    r_hard_override: Optional[float] = None
    w_cov: float = 1.0
    w_det: float = 1.0
    w_tr: float = 0.05
    is_static: bool = False           # responder radio / base node
    # information-triggered intermittent connectivity (Amendment A1)
    connectivity: str = 'intermittent'  # intermittent | strict
    t_hold_s: float = 60.0            # max age of an unsecured detection before delivery
    explore_lam: float = 0.0          # soft outage spring while exploring (no projection)
    base_id: Optional[int] = None     # responder node id (None: no responder)
    base_xy: Optional[Tuple[float, float]] = None
    # exploration aids (not claimed as contributions)
    w_goal: float = 0.5               # pull toward the long-range frontier goal
    goal_len_m: float = 60.0          # distance discount of frontier goals
    sector_bias: float = 1.0          # preference for the UAV's own angular sector
    n_agents: int = 1                 # swarm size (for sector assignment)


@dataclass
class Beacon:
    sender: int
    t: float
    pos: np.ndarray
    waypoint: np.ndarray
    entries: List[Entry]
    looks: Optional[tuple] = None        # sender's SearchMap rows + versions (G-counter)
    acks: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    sa_assign: Optional[Tuple[int, np.ndarray, Tuple[int, int]]] = None  # (wingman, point, uid)
    vmax: float = 0.0                    # sender speed bound (0 for static nodes)
    root_t: float = -np.inf              # freshest root (responder) heartbeat known
    tether_until: float = -np.inf        # delivery tether stamp (propagated by max)


@dataclass
class NeighbourState:
    pos: np.ndarray
    waypoint: np.ndarray
    t: float
    vmax: float = 0.0


class SentinelAgent:
    def __init__(self, uid: int, pos, cfg: AgentConfig, profile: ChannelProfile,
                 search_map: SearchMap, bounds: Tuple[np.ndarray, np.ndarray],
                 rng: np.random.Generator,
                 obstructions: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                 no_fly: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.id = uid
        self.pos = np.asarray(pos, float)[:2].copy()
        self.cfg = cfg
        self.profile = profile
        self.map = search_map
        self.bounds = bounds
        self.rng = rng
        self.obstructions = obstructions
        self.no_fly = no_fly
        self.ledger = Ledger(uid)
        self.nbrs: Dict[int, NeighbourState] = {}
        self.waypoint = self.pos.copy()
        self._last_plan = -np.inf
        self._pending_acks: Dict[int, List[Tuple[int, int]]] = {}
        self._sa_task: Optional[Tuple[np.ndarray, float, Tuple[int, int]]] = None
        self._sa_out: Optional[Tuple[int, np.ndarray, Tuple[int, int]]] = None
        self._sa_out_expiry = -np.inf
        self._sa_served: set = set()
        self.sa_issued: List[np.ndarray] = []   # target positions of issued SA requests
        self.comm_cut = False
        self.alive = True
        self.r_hard = (cfg.r_hard_override if cfg.r_hard_override is not None
                       else reliable_range(profile, cfg.r_hard_eps, cfg.r_hard_nlos))
        # reliable range for 0..max obstructions (index = obstruction count)
        self.r_table = np.array([reliable_range(profile, cfg.r_hard_eps, k)
                                 for k in range(profile.max_obstructions + 1)])
        self.spring_set: List[int] = []
        self.root_t = -np.inf            # freshest heartbeat of the root (responder) node
        self.parent_pos: Optional[np.ndarray] = None
        self.reconnecting = False
        self.mode = 'explore'            # explore | deliver (intermittent connectivity)
        self.tethered_now = False        # constraints applied in the last step (for metrics)
        self.tether_until = -np.inf      # hold links while any connected UAV is delivering
        self.goal = None
        self.deliveries = 0
        self.compute_s = 0.0           # wall time of the last planning step

    # ------------------------------------------------------------------ comms
    def make_beacon(self, t: float, with_map: bool) -> Beacon:
        acks, self._pending_acks = self._pending_acks, {}
        sa = self._sa_out if t <= self._sa_out_expiry else None   # repeated for robustness
        return Beacon(self.id, t, self.pos.copy(), self.waypoint.copy(), self.ledger.digest(),
                      self.map.rows() if with_map else None, acks, sa,
                      0.0 if self.cfg.is_static else self.cfg.v_max,
                      t if self.cfg.is_static else self.root_t,
                      self.tether_until)

    def receive(self, b: Beacon) -> None:
        self.nbrs[b.sender] = NeighbourState(b.pos.copy(), b.waypoint.copy(), b.t, b.vmax)
        if b.tether_until > self.tether_until:
            self.tether_until = b.tether_until
        if b.root_t > self.root_t:
            self.root_t = b.root_t
            self.parent_pos = b.pos.copy()
        held = self.ledger.merge(b.entries)
        if held:
            self._pending_acks.setdefault(b.sender, []).extend(held)
        if self.id in b.acks:
            self.ledger.ack(b.acks[self.id], b.sender)
        if b.looks is not None:
            self.map.merge_rows(*b.looks)
        if (b.sa_assign is not None and b.sa_assign[0] == self.id and self._sa_task is None
                and b.sa_assign[2] not in self._sa_served):
            self._sa_served.add(b.sa_assign[2])
            self._sa_task = (np.asarray(b.sa_assign[1], float), b.t + self.cfg.sa_timeout_s,
                             b.sa_assign[2])

    def forget(self, uav: int) -> None:
        self.nbrs.pop(uav, None)
        self.ledger.forget_holder(uav)

    # -------------------------------------------------------------- sensing
    def sense(self, detections, t: float) -> None:
        xyz = np.r_[self.pos, self.cfg.altitude]
        self.map.record_look(self.id, self.pos, self.cfg.footprint_r)
        for pos, conf, _ in detections:
            e = self.ledger.observe(pos, conf, t, xyz)
            if not self.cfg.fusion and len(e.views) > 1:
                best = max(e.views.values())
                e.views = {k: v for k, v in e.views.items() if v == best}

    # ------------------------------------------------------------ topology
    def _fresh_neighbours(self, t: float) -> Dict[int, NeighbourState]:
        stale = [j for j, s in self.nbrs.items() if t - s.t > self.cfg.beacon_stale_s]
        for j in stale:
            # conservative: copies held by an unreachable UAV are not counted
            self.forget(j)
        return self.nbrs

    def link_radius(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Hard link radius for segments a[k]-b[k] (or a[k]-b)."""
        a = np.atleast_2d(a)
        if not self.cfg.obstruction_aware or self.obstructions is None \
                or self.cfg.r_hard_override is not None:
            return np.full(len(a), self.r_hard)
        k = np.minimum(self.obstructions(a, b, self.cfg.obs_buffer_m), len(self.r_table) - 1)
        return self.r_table[k]

    def _select_spring_set(self, nbrs: Dict[int, NeighbourState]) -> List[int]:
        mode = self.cfg.neighbour_set
        if mode == 'none' or not nbrs:
            return []
        ids = [self.id] + sorted(nbrs)
        pos = np.stack([self.pos] + [nbrs[j].pos for j in ids[1:]])
        dist = graph.pairwise_dist(pos)
        if mode == 'nn':
            return [ids[1 + int(np.argmin(dist[0, 1:]))]]
        iu = np.triu_indices(len(ids), 1)
        rad = np.zeros_like(dist)
        rad[iu] = self.link_radius(pos[iu[0]], pos[iu[1]])
        rad = rad + rad.T
        in_range = dist <= rad
        if mode == 'all':
            return [ids[k] for k in range(1, len(ids)) if in_range[0, k]]
        np.fill_diagonal(in_range, False)
        sel = graph.lmst_selected(0, in_range, dist)
        if not sel:   # isolated from every in-range neighbour: hold on to the nearest
            return [ids[1 + int(np.argmin(dist[0, 1:]))]]
        return [ids[k] for k in sorted(sel)]

    # ------------------------------------------------------------- planning
    def _sa_admissible(self, j: int, p: np.ndarray) -> bool:
        s = self.nbrs.get(j)
        return s is not None and np.linalg.norm(p - self.pos) <= float(self.link_radius(p, self.pos)[0])

    # ------------------------------------------------ intermittent connectivity
    def _secured(self, e: Entry) -> bool:
        if self.cfg.base_id is not None:
            return self.cfg.base_id in e.holders
        return len(e.holders) >= 2           # no responder: replicated to another UAV

    def unsecured(self, t: float) -> List[Entry]:
        """Confirmed detections this UAV must deliver (own, or orphaned copies)."""
        orphan = 3.0 * self.cfg.t_hold_s
        return [e for e in self.ledger.entries.values()
                if e.confirmed and not self._secured(e)
                and (e.uid[0] == self.id or t - e.t_first > orphan)]

    def connected_to_root(self, t: float) -> bool:
        return t - self.root_t <= self.cfg.root_timeout_s

    def _update_mode(self, t: float) -> None:
        uns = self.unsecured(t)
        if self.mode == 'explore' and any(t - e.t_first >= self.cfg.t_hold_s for e in uns):
            self.mode = 'deliver'
        elif self.mode == 'deliver' and not uns:
            self.mode = 'explore'
            self.deliveries += 1

    TETHER_HOLD_S = 2.0

    def _tethered(self, t: float) -> bool:
        """Whether the connectivity constraints (spring + projection) are active.

        A delivering UAV that is connected to the responder stamps
        ``tether_until = t + 2 s``; the stamp propagates through beacons (max), so
        every UAV of the connected component holds its links while a delivery is
        in progress and releases them 2 s after it ends (no mutual deadlock).
        """
        if self.cfg.connectivity == 'strict':
            return True
        if self.mode == 'deliver' and self.connected_to_root(t):
            self.tether_until = max(self.tether_until, t + self.TETHER_HOLD_S)
            return True
        return t <= self.tether_until and self.connected_to_root(t)

    def _plan(self, t: float, nbrs: Dict[int, NeighbourState]) -> None:
        cfg = self.cfg
        if cfg.connectivity == 'intermittent':
            self._update_mode(t)
            self.reconnecting = self.mode == 'deliver' and not self.connected_to_root(t)
            if self.reconnecting:
                # data mule: carry unsecured detections back toward the responder
                # (or, without one, toward the last known neighbour)
                if cfg.base_xy is not None:
                    self.waypoint = np.asarray(cfg.base_xy, float)
                elif self.parent_pos is not None:
                    self.waypoint = self.parent_pos.copy()
                elif nbrs:
                    self.waypoint = min((s.pos for s in nbrs.values()),
                                        key=lambda q: np.linalg.norm(q - self.pos)).copy()
                return
        else:
            self.reconnecting = (cfg.reconnect and self.parent_pos is not None
                                 and t - self.root_t > cfg.root_timeout_s)
        if cfg.connectivity == 'strict' and self.reconnecting:
            # cut off from the responder: return toward the last relay of a fresh
            # heartbeat (gradient of the heartbeat-propagation tree)
            self.waypoint = self.parent_pos.copy()
            return
        if self._sa_task is not None:
            point, expiry, _ = self._sa_task
            if t > expiry or np.linalg.norm(point - self.pos) < 1.0:
                self._sa_task = None
            else:
                self.waypoint = point
                return
        if cfg.sa_enabled:
            for e in self.ledger.tentative():
                if not e.sa_requested and e.uid[0] == self.id:
                    plan = plan_flank(self.id, self.pos, e.pos,
                                      {j: s.pos for j, s in nbrs.items()},
                                      self._sa_admissible, cfg.sa_offset_m)
                    e.sa_requested = True
                    if plan is not None:
                        self._sa_out = (plan[0], plan[1], e.uid)
                        self._sa_out_expiry = t + 5.0
                        self.sa_issued.append(e.pos.copy())
                        break
        exclude = [s.waypoint for s in nbrs.values()]
        field_ = self.map.gain_field(cfg.footprint_r, exclude)
        norm = max(float(self.map.disk_kernel(cfg.footprint_r).sum())
                   * self.map.p0 * (1 - self.map.q), 1e-9)
        horizon = cfg.v_max * cfg.horizon_s
        goal = self._frontier_goal(field_, nbrs) if cfg.w_goal > 0 else None
        ctx = FitnessContext(
            x_i=self.pos, horizon=horizon,
            gain=lambda p: self.map.sample(field_, p) / norm,
            neighbours=([nbrs[j].pos for j in self.spring_set if j in nbrs]
                        if self._tethered(t) or cfg.explore_lam > 0 else []),
            tentative=[(e.pos, e.confidence) for e in self.ledger.tentative()],
            unreplicated=self.ledger.unreplicated_count(),
            profile=self.profile, obstructions=self.obstructions,
            w_cov=cfg.w_cov, w_det=cfg.w_det, w_tr=cfg.w_tr)
        seeds = [self.waypoint] + [s.waypoint for s in nbrs.values()
                                   if np.linalg.norm(s.waypoint - self.pos) <= horizon]
        spring = cfg.spring
        if not self._tethered(t) and cfg.explore_lam > 0:
            # soft tether while exploring: keeps opportunistic contacts likely
            spring = replace(cfg.spring, mode='outage', lam0=cfg.explore_lam)
        f = lambda p: fitness(p, ctx, spring)  # noqa: E731
        if self.no_fly is not None:
            f = lambda p, f0=f: np.where(self.no_fly(p), -1e6, f0(p))  # noqa: E731
        if goal is not None:
            d0 = np.linalg.norm(goal - self.pos)
            seeds.append(self.pos + (goal - self.pos) / max(d0, 1e-9) * min(horizon, d0))
            f = lambda p, f0=f: f0(p) + cfg.w_goal * (  # noqa: E731
                d0 - np.linalg.norm(p - goal, axis=1)) / horizon
        self.goal = goal
        self.waypoint, _ = optimise(f, self.pos, horizon, self.bounds, self.rng, cfg.pso, seeds)

    def _frontier_goal(self, field_: np.ndarray, nbrs: Dict[int, NeighbourState]):
        """Distant cell with the best expected-detections / distance trade-off."""
        m = self.map
        score = field_ / (1.0 + np.hypot(m.cx - self.pos[0], m.cy - self.pos[1])
                          / self.cfg.goal_len_m)
        if self.cfg.sector_bias > 0 and self.cfg.n_agents > 1:
            if getattr(self, '_sector', None) is None:
                anchor = (np.asarray(self.cfg.base_xy, float) if self.cfg.base_xy is not None
                          else (self.bounds[0] + self.bounds[1]) / 2)
                ang = np.arctan2(m.cy - anchor[1], m.cx - anchor[0])
                lo, hi = float(ang.min()), float(ang.max())
                w = (hi - lo) / self.cfg.n_agents
                k = self.id % self.cfg.n_agents
                self._sector = 1.0 + self.cfg.sector_bias * (
                    (ang >= lo + k * w) & (ang <= lo + (k + 1) * w))
            score = score * self._sector
        for s in nbrs.values():                 # leave regions neighbours are heading to
            win = m._window(s.waypoint, 2 * self.cfg.footprint_r)
            if win is not None:
                score[win[0]][win[1]] = 0
        if score.max() <= 0:
            return None
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        return np.array([m.cx[i, j], m.cy[i, j]])

    # ---------------------------------------------------------------- step
    def _desired_step(self, t: float, dt: float, nbrs: Dict[int, NeighbourState]) -> np.ndarray:
        """Unconstrained displacement for this tick (overridden by baselines)."""
        if t - self._last_plan >= self.cfg.plan_period_s:
            t0 = time.perf_counter()
            self._plan(t, nbrs)
            self.compute_s = time.perf_counter() - t0
            self._last_plan = t
        d = self.waypoint - self.pos
        n = np.linalg.norm(d)
        return d if n <= self.cfg.v_max * dt else d / n * self.cfg.v_max * dt

    def _occlusion_backtrack(self, step, others, balls, radii_now, margins, vmax_age,
                             tries: int = 6):
        """Shrink the step until every preserved link stays within its radius.

        The radius of link j is evaluated at the *proposed* position against the
        neighbour's last known position with obstacles inflated by
        ``v_j (dt + age_j)``, the most j can move before the next check. That
        count upper-bounds the obstructions of the true future link, so each
        side's radius lower-bounds the true reliable range and
        ||x_i' - x_j'|| <= (r_i + r_j) / 2 <= R_eps(true). A move that is
        outside a ball is accepted only if it approaches the centre without
        shrinking that link's radius (re-attachment). Hovering is the fallback.
        """
        if not self.cfg.obstruction_aware or self.obstructions is None \
                or self.cfg.r_hard_override is not None:
            return step
        b = self.cfg.obs_buffer_m
        for _ in range(tries):
            new = self.pos + step
            ok = True
            for k, ((c, _), r_now, m, va) in enumerate(zip(balls, radii_now, margins, vmax_age)):
                cnt = self.obstructions(new[None], others[k], b + va)[0]
                r_new = self.r_table[min(cnt, len(self.r_table) - 1)]
                d_new = np.linalg.norm(new - c)
                inside = d_new <= 0.5 * r_new * (1 - 1e-7) - m
                approach = d_new <= np.linalg.norm(self.pos - c) and r_new >= r_now
                if not (inside or approach):
                    ok = False
                    break
            if ok:
                return step
            step = 0.5 * step
        return np.zeros_like(step)

    def _avoid(self, step: np.ndarray) -> np.ndarray:
        """Rotate the step away from no-fly structures (hover if no direction is free)."""
        if self.no_fly is None or not np.any(step):
            return step
        angles = np.deg2rad([0, 30, -30, 60, -60, 90, -90, 135, -135])
        c, s_ = np.cos(angles), np.sin(angles)
        cand = np.stack([c * step[0] - s_ * step[1], s_ * step[0] + c * step[1]], 1)
        # check the end point and the midpoint of each candidate step
        free = ~(self.no_fly(self.pos + cand) | self.no_fly(self.pos + 0.5 * cand))
        if self.no_fly(self.pos[None])[0]:
            return step                       # already inside (spawned/pushed): leave freely
        k = np.flatnonzero(free)
        return cand[k[0]] if len(k) else np.zeros_like(step)

    def step(self, t: float, dt: float) -> np.ndarray:
        """Advance one control tick; returns the new position."""
        if not self.alive or self.cfg.is_static:
            return self.pos
        nbrs = self._fresh_neighbours(t)
        self.spring_set = self._select_spring_set(nbrs)
        step = self._desired_step(t, dt, nbrs)
        n = np.linalg.norm(step)
        if n > self.cfg.v_max * dt:
            step = step / n * self.cfg.v_max * dt
        step = self._avoid(step)
        self.tethered_now = bool(self.cfg.projection and self.spring_set and self._tethered(t))
        if self.tethered_now:
            # Bounded-staleness margin: neighbour j may have moved v_j * age since
            # its beacon; half of that displacement shifts the common midpoint.
            others = np.stack([nbrs[j].pos for j in self.spring_set])
            margins = [0.5 * nbrs[j].vmax * max(0.0, t - nbrs[j].t) for j in self.spring_set]
            radii = self.link_radius(np.repeat(self.pos[None], len(others), 0), others)
            balls = [midpoint_balls(self.pos, [o], r, m)[0]
                     for o, r, m in zip(others, radii, margins)]
            proj = project_step(self.pos, step, balls)
            if (np.linalg.norm(proj) < 1e-9
                    and any(np.linalg.norm(self.pos - c) > r for c, r in balls)):
                # re-attach without worsening any other constraint (else hover)
                proj = project_step(self.pos, recovery_step(self.pos, balls, self.cfg.v_max * dt),
                                    balls)
            vmax_age = [nbrs[j].vmax * (dt + max(0.0, t - nbrs[j].t)) for j in self.spring_set]
            step = self._occlusion_backtrack(proj, others, balls, radii, margins, vmax_age)
        self.pos = np.clip(self.pos + step, self.bounds[0], self.bounds[1])
        return self.pos
