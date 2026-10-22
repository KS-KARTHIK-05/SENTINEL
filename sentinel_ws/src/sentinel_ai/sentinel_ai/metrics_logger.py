"""Ground-truth metrics for SITL runs (same definitions as the fast simulator).

Writes ``series.csv`` (1 Hz) and ``summary.json`` into ``out_dir``; the summary
is refreshed every 10 s so an interrupted run still leaves results.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String

from sentinel_core import graph
from sentinel_core.channel import get_profile, outage_prob
from sentinel_core.maps import SearchMap
from sentinel_core.metrics import main_component, match_entries, recovery_times
from sentinel_interfaces.msg import DetectorStats, LedgerSnapshot
from sentinel_sim import world as W


class MetricsLogger(Node):
    def __init__(self):
        super().__init__('metrics_logger')
        p = self.declare_parameter
        self.n = p('n_uav', 3).value
        truth = json.load(open(p('truth_json', '').value))
        self.world = W.from_truth(truth)
        self.profile = get_profile(p('channel_profile', 'urban_rubble').value)
        self.eps = p('eps_graph', 0.05).value
        self.footprint = p('footprint_r', 12.0).value
        self.alt = p('altitude', 30.0).value
        self.out_dir = p('out_dir', os.path.expanduser('~/.ros/sentinel_sitl/metrics')).value
        os.makedirs(self.out_dir, exist_ok=True)
        self.targets = np.array([t['pos'] for t in truth['targets']]).reshape(-1, 2)
        self.base_id = self.n if truth.get('base_xy') is not None else None
        self.pos = {}
        if self.base_id is not None:
            self.pos[self.base_id] = np.asarray(truth['base_xy'], float)
        self.ledgers = {}
        self.alive = set(range(self.n)) | ({self.base_id} if self.base_id is not None else set())
        lo, hi = self.world.bounds
        self.truth_map = SearchMap(tuple(hi - lo), 2.0, 1)
        self.fail_t = []
        self.ever = set()
        self.t_first, self.t_base = {}, {}
        self.comm = {}
        self.det_stats = {}
        self.rows = []
        self.t0 = None
        for i in range(self.n):
            self.create_subscription(PointStamped, f'/px4_{i}/sentinel/pose',
                                     lambda m, i=i: self._on_pose(i, m), 10)
            self.create_subscription(LedgerSnapshot, f'/px4_{i}/sentinel/ledger',
                                     self._on_ledger, 10)
            self.create_subscription(DetectorStats, f'/px4_{i}/sentinel/detector_stats',
                                     self._on_det, 10)
        self.create_subscription(LedgerSnapshot, '/sentinel/base/ledger', self._on_ledger, 10)
        self.create_subscription(String, '/sentinel/fail', self._on_fail, 10)
        self.create_subscription(String, '/sentinel/comm_stats',
                                 lambda m: self.comm.update(json.loads(m.data)), 10)
        self.create_timer(1.0, self._record)
        self.create_timer(10.0, self._write)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_pose(self, i, m):
        self.pos[i] = np.array([m.point.x, m.point.y])
        if m.point.z > 0.8 * self.alt and i in self.alive:
            self.truth_map.record_look(0, self.pos[i], self.footprint)

    def _on_ledger(self, m: LedgerSnapshot):
        self.ledgers[m.uav_id] = [(np.array([x, y]), c) for x, y, c in
                                  zip(m.x, m.y, m.confirmed)]

    def _on_det(self, m: DetectorStats):
        self.det_stats[m.uav_id] = {'model': m.model, 'inference_ms_p50': m.inference_ms_p50,
                                    'inference_ms_p95': m.inference_ms_p95,
                                    'total_ms_p50': m.total_ms_p50, 'frames': m.frames}

    def _on_fail(self, m: String):
        ev = json.loads(m.data)
        self.fail_t.append(self._now() - (self.t0 or self._now()))
        if ev['kind'] == 'crash':
            self.alive.discard(int(ev['uav']))
            self.ledgers.pop(int(ev['uav']), None)

    def _confirmed(self, i):
        return [p for p, c in self.ledgers.get(i, []) if c]

    def _record(self):
        now = self._now()
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        ids = sorted(i for i in self.alive if i in self.pos)
        if len(ids) < 1:
            return
        P = np.stack([self.pos[i] for i in ids])
        obs = self.world.obstruction_matrix(P)
        d = graph.pairwise_dist(P)
        G = np.zeros_like(d, dtype=bool)
        for k in np.unique(obs):
            sel = obs == k
            G[sel] = outage_prob(self.profile, d[sel], int(k)) <= self.eps
        np.fill_diagonal(G, False)
        comps = [[ids[k] for k in c] for c in graph.components(G)]
        main = main_component(comps, self.base_id)
        uavs = [i for i in ids if i != self.base_id]
        nrr = sum(1 for i in main if i != self.base_id) / max(1, len(uavs))
        held = set()
        for i in ids:
            held |= match_entries(self._confirmed(i), self.targets)[0]
        self.ever |= held
        for k in held:
            self.t_first.setdefault(k, t)
        tdr_base = float('nan')
        if self.base_id is not None:
            mb = match_entries(self._confirmed(self.base_id), self.targets)[0]
            tdr_base = len(mb) / max(1, len(self.targets))
            for k in mb:
                self.t_base.setdefault(k, t)
        self.rows.append({'t': t, 'nrr': nrr, 'connected': len(comps) == 1,
                          'lambda2': graph.algebraic_connectivity(G),
                          'coverage': self.truth_map.coverage_fraction(),
                          'tdr': len(held) / max(1, len(self.targets)), 'tdr_base': tdr_base,
                          'lost': len(self.ever - held), 'n_alive': len(uavs),
                          'pdr': self.comm.get('pdr')})

    def _summary(self):
        if not self.rows:
            return {}
        t = np.array([r['t'] for r in self.rows])
        conn = np.array([r['connected'] for r in self.rows])
        last = self.rows[-1]
        rec = recovery_times(t, conn, sorted(set(self.fail_t)))
        false_max = dup_max = 0
        for i in self.alive:
            _, fa, du = match_entries(self._confirmed(i), self.targets)
            false_max, dup_max = max(false_max, fa), max(dup_max, du)
        lat = [self.t_base[k] - self.t_first[k] for k in self.t_base if k in self.t_first]
        ever_n = len(self.ever)
        return {'duration_s': float(t[-1]), 'n_targets': len(self.targets),
                'tdr': last['tdr'], 'tdr_base': last['tdr_base'],
                'info_loss': last['lost'] / ever_n if ever_n else 0.0,
                'false_entries_max': false_max, 'dup_entries_max': dup_max,
                'nrr_final': last['nrr'], 'frac_connected': float(conn.mean()),
                'lambda2_mean': float(np.mean([r['lambda2'] for r in self.rows])),
                'coverage': last['coverage'], 'recovery_s': rec,
                'latency_mean_s': float(np.mean(lat)) if lat else None,
                'comm': self.comm, 'detector': self.det_stats}

    def _write(self):
        if not self.rows:
            return
        with open(os.path.join(self.out_dir, 'series.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        with open(os.path.join(self.out_dir, 'summary.json'), 'w') as f:
            json.dump(self._summary(), f, indent=1, default=float)


def main(args=None):
    rclpy.init(args=args)
    node = MetricsLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._write()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
