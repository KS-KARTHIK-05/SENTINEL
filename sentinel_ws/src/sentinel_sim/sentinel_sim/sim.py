"""Fast multi-UAV simulation of SENTINEL and baselines.

Random streams for the world, failures, channel and detector are derived from
the run seed independently of the method, so every method faces the same map,
targets and failure schedule for a given seed (common random numbers, which
makes paired statistical tests valid).
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from typing import Any, Dict, List

import numpy as np

from sentinel_core import failures as F
from sentinel_core import graph
from sentinel_core.agent import AgentConfig, SentinelAgent
from sentinel_core.baselines import Lambda2Agent, LawnmowerAgent
from sentinel_core.channel import get_profile, outage_table
from sentinel_core.detector_model import DetectorModel
from sentinel_core.maps import SearchMap
from sentinel_core.metrics import main_component, match_entries, recovery_times, time_to_reach

from . import world as W
from .linkfield import LinkField
from .methods import build_config

DEFAULTS: Dict[str, Any] = {
    'method': 'sentinel',
    'seed': 0,
    'n_uav': 5,
    'duration_s': 600.0,
    'dt': 0.5,
    'sense_period_s': 1.0,
    'map_gossip_period_s': 2.0,
    'record_period_s': 1.0,
    'map': {'kind': 'generate', 'size': 100.0, 'n_targets': 20, 'n_moving': 0, 'base': True},
    'channel': {'profile': 'urban_rubble'},
    'failures': {'kind': 'none'},
    'detector': {},
    'agent': {},
    'spring': {},
    'agent_attrs': {},
    'policy_path': None,   # B5: trained MAPPO actor (npz)
    'thresholds': {'low': 0.40, 'high': 0.70},   # SA band / confirmation (E6)     # class-level controller gains (B3/B4) and lawnmower lane factor
    'cell_m': 2.0,
    'eps_graph': 0.05,
}


def _merge(d: dict, u: dict) -> dict:
    out = copy.deepcopy(d)
    for k, v in (u or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _failure_schedule(spec: dict, n_uav: int, rng: np.random.Generator, T: float):
    kind = spec.get('kind', 'none')
    t = spec.get('t', 0.4 * T)
    if kind == 'none':
        return []
    if kind == 'single':
        return F.single(t, spec.get('uav', int(rng.integers(n_uav))), spec.get('mode', 'crash'))
    if kind == 'simultaneous':
        k = spec.get('k') or max(1, int(round(spec.get('fraction', 0.3) * n_uav)))
        return F.simultaneous(t, F.random_subset(n_uav, min(k, n_uav - 1), rng),
                              spec.get('mode', 'crash'))
    if kind == 'cascading':
        k = spec.get('k', max(2, n_uav // 3))
        return F.cascading(t, F.random_subset(n_uav, min(k, n_uav - 1), rng),
                           spec.get('gap', 20.0), spec.get('mode', 'crash'))
    if kind == 'poisson':
        return F.poisson(spec.get('rate', 1 / 120), T, n_uav, rng,
                         spec.get('max', n_uav - 1), spec.get('mode', 'crash'))
    if kind == 'brownout':
        return F.brownout(t, spec.get('uav', int(rng.integers(n_uav))), spec.get('duration', 60.0),
                          spec.get('drop_db', 20.0))
    raise ValueError(kind)


class Simulation:
    def __init__(self, cfg: dict):
        self.cfg = c = _merge(DEFAULTS, cfg)
        from sentinel_core.ledger import set_thresholds
        set_thresholds(c['thresholds']['low'], c['thresholds']['high'])
        ss = np.random.SeedSequence(c['seed'])
        r_world, r_fail, r_chan, r_det, r_agent = (np.random.default_rng(s) for s in ss.spawn(5))
        self.rng_det = r_det
        m = c['map']
        if m['kind'] == 'disaster_25':
            self.world = W.disaster_25(r_world, base=m.get('base', False))
        else:
            self.world = W.generate(m['size'], m['n_targets'], r_world,
                                    n_moving=m.get('n_moving', 0), base=m.get('base', True),
                                    buried=m.get('buried', 1))
        self.profile = get_profile(c['channel']['profile'],
                                   **{k: v for k, v in c['channel'].items() if k != 'profile'})
        self.N = c['n_uav']
        self.has_base = self.world.base_xy is not None
        self.base = self.N if self.has_base else None
        self.n_nodes = self.N + int(self.has_base)
        self.detector = DetectorModel(**c['detector'])
        self.events = sorted(_failure_schedule(c['failures'], self.N, r_fail, c['duration_s']),
                             key=lambda e: e.t)
        cls, acfg = build_config(c['method'], AgentConfig(), c['agent'], c['spring'])
        if self.has_base:
            acfg = replace(acfg, base_id=self.N, base_xy=tuple(map(float, self.world.base_xy)))
        acfg = replace(acfg, n_agents=self.N)
        self.acfg = acfg
        import sentinel_core.ledger as _ledger          # per run: never leaks between pooled runs
        _ledger.FUSION_RULE = acfg.fusion_rule
        lo, hi = self.world.bounds
        size = tuple(hi - lo)
        anchor = self.world.base_xy if self.has_base else (lo + hi) / 2
        spawn = self._spawn(anchor, self.N, 4.0)
        agent_rngs = [np.random.default_rng(s) for s in
                      np.random.SeedSequence(c['seed'] + 7919).spawn(self.n_nodes)]
        self.agents: List[SentinelAgent] = []
        for i in range(self.N):
            a = cls(i, spawn[i], acfg, self.profile,
                    SearchMap(size, c['cell_m'], self.n_nodes), (lo, hi), agent_rngs[i],
                    self.world.obstructions, self.world.in_no_fly)
            if isinstance(a, LawnmowerAgent):
                a.n_strips, a.strip = self.N, i
            if isinstance(a, Lambda2Agent):
                a.oracle = self._lambda2_oracle
            for k, v in c['agent_attrs'].items():
                setattr(a, k, v)
            if c['policy_path'] and hasattr(a, 'policy'):
                with np.load(c['policy_path']) as z:
                    a.policy = [z[f'arr_{q}'] for q in range(len(z.files))]
            self.agents.append(a)
        if self.has_base:
            bcfg = replace(acfg, is_static=True, sa_enabled=False)
            self.agents.append(SentinelAgent(self.N, self.world.base_xy, bcfg, self.profile,
                                             SearchMap(size, c['cell_m'], self.n_nodes),
                                             (lo, hi), agent_rngs[-1], self.world.obstructions))
        self.truth = SearchMap(size, c['cell_m'], 1)
        # coverage is measured over reachable cells (outside tall structures)
        self.reachable = ~self.world.in_no_fly(
            np.stack([self.truth.cx.ravel(), self.truth.cy.ravel()], 1)).reshape(self.truth.shape)
        self.links = LinkField(self.profile, self.n_nodes, r_chan)
        self.tx_offset = np.zeros(self.n_nodes)
        self.targets_xy = np.array([t.pos for t in self.world.targets]).reshape(-1, 2)
        self._l2_cache = (None, None)
        self.t_first = {}      # target -> first time confirmed in any alive ledger
        self.carriers = {}     # target -> UAVs that ever held it confirmed
        self.t_base = {}       # target -> first time confirmed at the responder

    @staticmethod
    def _spawn(anchor, n, spacing):
        k = int(np.ceil(np.sqrt(n)))
        pts = [anchor + spacing * np.array([1 + i % k, 1 + i // k]) for i in range(n)]
        return np.array(pts)

    # ------------------------------------------------------------- helpers
    def _positions(self):
        return np.stack([a.pos for a in self.agents])

    def _alive_nodes(self):
        return [a.id for a in self.agents if a.alive]

    def _lambda2_oracle(self):
        key = self._t
        if self._l2_cache[0] == key:
            return self._l2_cache[1]
        alive = self._alive_nodes()
        pos = {i: self.agents[i].pos.copy() for i in alive}
        P = np.stack([pos[i] for i in alive])
        R = self.agents[0].r_hard
        s2 = -R * R / (2.0 * np.log(0.01))
        d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
        A = np.where(d2 <= R * R, np.exp(-d2 / (2 * s2)), 0.0)
        np.fill_diagonal(A, 0.0)
        L = np.diag(A.sum(1)) - A
        w, v = np.linalg.eigh(L)
        res = (float(w[1]) if len(w) > 1 else 0.0, {i: float(v[k, 1]) for k, i in enumerate(alive)},
               pos)
        self._l2_cache = (key, res)
        return res

    # ----------------------------------------------------------------- run
    def run(self) -> Dict[str, Any]:
        c = self.cfg
        dt, T = c['dt'], c['duration_s']
        n_ticks = int(round(T / dt))
        sense_every = max(1, int(round(c['sense_period_s'] / dt)))
        map_every = max(1, int(round(c['map_gossip_period_s'] / dt)))
        rec_every = max(1, int(round(c['record_period_s'] / dt)))
        ev_i = 0
        brown = []
        ever_confirmed: set = set()
        series = {k: [] for k in ('t', 'nrr', 'connected', 'lambda2', 'coverage', 'tdr',
                                  'tdr_base', 'n_alive', 'lost', 'n_deliver')}
        pdr_num = pdr_den = 0
        link_ok = link_bad = link_all = 0
        path_len = np.zeros(self.N)
        compute = []
        wall0 = time.perf_counter()
        for k in range(n_ticks):
            t = k * dt
            self._t = t
            # --- failures
            while ev_i < len(self.events) and self.events[ev_i].t <= t:
                e = self.events[ev_i]
                ev_i += 1
                a = self.agents[e.uav]
                if e.kind == 'crash':
                    a.alive = False
                elif e.kind == 'link_cut':
                    a.comm_cut = True
                elif e.kind == 'brownout':
                    brown.append(e)
            for e in brown:
                self.tx_offset[e.uav] = -e.drop_db * min(1.0, (t - e.t) / max(e.duration, 1e-6))
            # --- sensing
            if k % sense_every == 0:
                for a in self.agents[:self.N]:
                    if not a.alive:
                        continue
                    visible = [tg for j, tg in enumerate(self.world.targets)
                               if not self.world.target_hidden_by_debris(j)]
                    dets = self.detector.look(visible, np.r_[a.pos, a.cfg.altitude],
                                              a.cfg.footprint_r, self.rng_det, self.world.clutter_xy)
                    a.sense(dets, t)
                    self.truth.record_look(0, a.pos, a.cfg.footprint_r)
            # --- communication
            pos = self._positions()
            obs = self.world.obstruction_matrix(pos)
            alive = np.array([a.alive for a in self.agents])
            can = alive & ~np.array([a.comm_cut for a in self.agents])
            D = self.links.deliver(pos, obs, can, can, self.tx_offset)
            with_map = k % map_every == 0
            for s in np.flatnonzero(can):
                rx = np.flatnonzero(D[s])
                if len(rx) == 0:
                    self.agents[s].make_beacon(t, False)     # drains pending acks
                    continue
                b = self.agents[s].make_beacon(t, with_map)
                for r in rx:
                    self.agents[r].receive(b)
            dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            near = (dist <= self.agents[0].r_hard) & can[:, None] & can[None, :]
            np.fill_diagonal(near, False)
            pdr_num += int((D & near).sum())
            pdr_den += int(near.sum())
            # --- motion
            pos_before = self._positions()
            for a in self.agents[:self.N]:
                if a.alive:
                    p0 = a.pos.copy()
                    a.step(t, dt)
                    path_len[a.id] += np.linalg.norm(a.pos - p0)
                    if a.compute_s:
                        compute.append(a.compute_s)
                        a.compute_s = 0.0
            # --- preserved-link check (Proposition 1): mutually selected, tethered pairs
            pairs = [(a.id, j) for a in self.agents[:self.N] if a.alive and a.tethered_now
                     for j in a.spring_set if j < len(self.agents) and self.agents[j].alive
                     and (j == self.base or (self.agents[j].tethered_now
                                             and a.id in self.agents[j].spring_set))
                     and (j == self.base or a.id < j)]
            if pairs:
                ai = np.array([p[0] for p in pairs])
                aj = np.array([p[1] for p in pairs])
                tab, eps = outage_table(self.profile), self.acfg.r_hard_eps + 1e-9

                def reliable(P):
                    return tab(np.linalg.norm(P[ai] - P[aj], axis=1),
                               self.world.obstructions(P[ai], P[aj])) <= eps
                was = reliable(pos_before)          # Proposition 1 hypothesis
                now = reliable(self._positions())
                link_all += len(pairs)
                link_ok += int((was & now).sum())
                link_bad += int((was & ~now).sum())
            self.world.step(dt)
            # --- metrics
            if k % rec_every == 0:
                self._record(series, t, pos, obs, alive, ever_confirmed)
        self._links = (link_ok, link_bad, link_all)
        return self._summary(series, pdr_num, pdr_den, path_len, compute,
                             time.perf_counter() - wall0)

    def _record(self, series, t, pos, obs, alive, ever_confirmed):
        G = self.links.reliable_graph(pos, obs, self.cfg['eps_graph'])
        alive_ids = list(np.flatnonzero(alive))
        comps = graph.components(G, alive_ids)
        main = main_component(comps, self.base)
        uav_alive = [i for i in alive_ids if i < self.N]
        nrr = sum(1 for i in main if i < self.N) / max(1, len(uav_alive))
        sub = G[np.ix_(alive_ids, alive_ids)]
        held = set()
        for i in alive_ids:
            m, _, _ = match_entries([e.pos for e in self.agents[i].ledger.confirmed()],
                                    self.targets_xy)
            held |= m
            for k in m:
                self.carriers.setdefault(k, set()).add(int(i))
        for k in held:
            self.t_first.setdefault(k, t)
        ever_confirmed |= held
        tdr_base = float('nan')
        if self.has_base:
            mb, _, _ = match_entries([e.pos for e in self.agents[self.base].ledger.confirmed()],
                                     self.targets_xy)
            tdr_base = len(mb) / max(1, len(self.targets_xy))
            for k in mb:
                self.t_base.setdefault(k, t)
        series['t'].append(t)
        series['nrr'].append(nrr)
        series['connected'].append(len(comps) == 1)
        series['lambda2'].append(graph.algebraic_connectivity(sub))
        series['coverage'].append(float((self.truth.looks[0] > 0)[self.reachable].mean()))
        series['tdr'].append(len(held) / max(1, len(self.targets_xy)))
        series['tdr_base'].append(tdr_base)
        series['n_alive'].append(len(uav_alive))
        series['lost'].append(len(ever_confirmed - held))
        series['n_deliver'].append(sum(1 for a in self.agents[:self.N]
                                       if a.alive and getattr(a, 'mode', '') == 'deliver'))

    def _summary(self, s, pdr_num, pdr_den, path_len, compute, wall):
        t = np.asarray(s['t'])
        conn = np.asarray(s['connected'])
        fail_t = [e.t for e in self.events if e.kind in ('crash', 'link_cut', 'brownout')]
        rec = recovery_times(t, conn, sorted(set(fail_t)))
        false_max = dup_max = 0
        for a in self.agents:
            if a.alive:
                _, fa, du = match_entries([e.pos for e in a.ledger.confirmed()], self.targets_xy)
                false_max, dup_max = max(false_max, fa), max(dup_max, du)
        sa_total = sum(len(a.sa_issued) for a in self.agents)
        sa_false = sum(match_entries(a.sa_issued, self.targets_xy)[1] for a in self.agents)
        lost_final = s['lost'][-1] if s['lost'] else 0
        held_final = s['tdr'][-1] * max(1, len(self.targets_xy)) if s['tdr'] else 0
        ever_n = held_final + lost_final
        # split the final loss: every carrier failed (failure-induced, bounded by
        # Proposition 3) vs. still-alive carriers whose entry merged with a nearby
        # victim's entry (association limit for victims closer than the gate)
        alive = {a.id for a in self.agents if a.alive}
        held_now = {k for k in self.carriers
                    if any(k in match_entries([e.pos for e in self.agents[i].ledger.confirmed()],
                                              self.targets_xy)[0] for i in alive)}
        gone = set(self.carriers) - held_now
        lost_fail = sum(1 for k in gone if not (self.carriers[k] & alive))
        lost_assoc = len(gone) - lost_fail
        after = t >= (min(fail_t) if fail_t else 0.0)
        out = {
            **{f'cfg_{k}': v for k, v in self._flat_cfg().items()},
            'n_targets': len(self.targets_xy),
            'r_hard': self.agents[0].r_hard,
            'tdr': s['tdr'][-1], 'tdr_base': s['tdr_base'][-1],
            'info_loss': lost_fail / ever_n if ever_n else 0.0,
            'assoc_loss': lost_assoc / ever_n if ever_n else 0.0,
            'lost_targets': lost_fail,
            'merged_targets': lost_assoc,
            'false_entries_max': false_max, 'dup_entries_max': dup_max,
            'nrr_final': s['nrr'][-1], 'nrr_min_after_fail': float(np.min(np.asarray(s['nrr'])[after])),
            'frac_connected': float(conn.mean()),
            'lambda2_mean': float(np.mean(s['lambda2'])),
            'coverage': s['coverage'][-1],
            't_cov50': time_to_reach(t, np.asarray(s['coverage']), 0.5),
            't_cov90': time_to_reach(t, np.asarray(s['coverage']), 0.9),
            'recovery_mean_s': float(np.nanmean(rec)) if rec and not np.all(np.isnan(rec)) else float('nan'),
            'recovery_unrecovered': int(np.sum(np.isnan(rec))) if rec else 0,
            'n_failures': len(fail_t),
            'pdr': pdr_num / pdr_den if pdr_den else float('nan'),
            # Proposition 1 check: tethered, mutually preserved links that were reliable
            # before a step and are not after it
            'preserved_link_steps': self._links[0] + self._links[1],
            'link_violation_rate': (self._links[1] / (self._links[0] + self._links[1])
                                    if self._links[0] + self._links[1] else float('nan')),
            # share of tethered link-steps that started reliable (feasibility in practice)
            'link_reliable_start_frac': ((self._links[0] + self._links[1]) / self._links[2]
                                         if self._links[2] else float('nan')),
            'compute_ms_mean': 1e3 * float(np.mean(compute)) if compute else 0.0,
            'compute_ms_p95': 1e3 * float(np.quantile(compute, 0.95)) if compute else 0.0,
            'path_len_mean': float(path_len.mean()),
            'sa_triggers': sa_total, 'sa_false': sa_false,
            'deliveries': sum(getattr(a, 'deliveries', 0) for a in self.agents[:self.N]),
            'frac_delivering': float(np.mean(s['n_deliver']) / max(1, self.N)),
            **self._latency(),
            'wall_s': wall,
        }
        return {'summary': out, 'series': s}

    def _latency(self):
        """Detection-to-responder reporting metrics.

        ``rep_within_D``: share of *all* true targets reported to the responder
        within D seconds of their first confirmation (denominator = all targets,
        so detecting little cannot score well). ``report_value``: time-discounted
        reports, mean over targets of exp(-t_report / 300 s) (0 if never reported).
        ``on_time_30s`` is kept for reference (denominator = detected targets).
        """
        nan = float('nan')
        if not self.has_base:
            return {'latency_mean_s': nan, 'latency_p90_s': nan, 'on_time_30s': nan,
                    'rep_within_60s': nan, 'rep_within_120s': nan, 'report_value': nan}
        lat = [self.t_base[k] - self.t_first[k] for k in self.t_base if k in self.t_first]
        n_t = max(1, len(self.targets_xy))
        return {'latency_mean_s': float(np.mean(lat)) if lat else nan,
                'latency_p90_s': float(np.quantile(lat, 0.9)) if lat else nan,
                'on_time_30s': sum(1 for x in lat if x <= 30.0) / max(1, len(self.t_first)),
                'rep_within_60s': sum(1 for x in lat if x <= 60.0) / n_t,
                'rep_within_120s': sum(1 for x in lat if x <= 120.0) / n_t,
                'report_value': float(sum(np.exp(-t / 300.0) for t in self.t_base.values()) / n_t)}

    def _flat_cfg(self):
        c = self.cfg
        return {'method': c['method'], 'seed': c['seed'], 'n_uav': c['n_uav'],
                'map_size': c['map'].get('size', 25.0), 'map_kind': c['map']['kind'],
                'n_moving': c['map'].get('n_moving', 0), 'profile': c['channel']['profile'],
                'failures': c['failures'].get('kind', 'none'),
                'det_sigma': self.detector.sigma, 'det_fp_rate': self.detector.fp_rate,
                'det_cmax': self.detector.c_max,
                'thr_low': c['thresholds']['low'], 'thr_high': c['thresholds']['high']}


def run(cfg: dict) -> Dict[str, Any]:
    return Simulation(cfg).run()
