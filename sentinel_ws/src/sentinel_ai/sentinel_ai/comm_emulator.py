"""Wireless channel emulator for SITL runs.

Every beacon published on ``/px4_i/sentinel/beacon_out`` is delivered to
``/px4_j/sentinel/beacon_in`` only if the sampled SINR of link i->j exceeds the
decoding threshold, using the channel model of ``sentinel_core.channel``
(path loss, building obstructions from the world geometry, correlated
shadowing, Rician/Rayleigh fading, interference from colliding transmitters).
The responder radio (static base node, if the world defines one) is hosted
here. Failure events arrive on ``/sentinel/fail``.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String

from sentinel_core.channel import LinkSampler, get_profile
from sentinel_interfaces.msg import Beacon, LedgerSnapshot
from sentinel_sim import world as W

from .emps_agent import load_tuned, make_agent


class CommEmulator(Node):
    def __init__(self):
        super().__init__('comm_emulator')
        p = self.declare_parameter
        self.n = p('n_uav', 3).value
        truth = json.load(open(p('truth_json', '').value))
        self.world = W.from_truth(truth)
        profile_name = p('channel_profile', 'urban_rubble').value
        self.profile = get_profile(profile_name)
        seed = p('seed', 0).value
        self.rng = np.random.default_rng(seed + 104729)
        self.sampler = LinkSampler(self.profile, self.rng)
        self.has_base = truth.get('base_xy') is not None
        self.base_id = self.n if self.has_base else None
        n_nodes = self.n + int(self.has_base)
        self.pos = {}
        self.alive = set(range(n_nodes))
        self.cut = set()
        self.tx_offset_db = {}
        self.base = None
        if self.has_base:
            method = p('method', 'sentinel').value
            tuned = load_tuned(p('tuned_yaml', '').value, method)
            self.base = make_agent(self.base_id, truth['base_xy'], truth,
                                   method,
                                   profile_name, n_nodes, seed,
                                   30.0, static=True, tuned=tuned)
            self.pos[self.base_id] = np.asarray(truth['base_xy'], float)
        self.out = {}
        for i in range(self.n):
            self.create_subscription(PointStamped, f'/px4_{i}/sentinel/pose',
                                     lambda m, i=i: self._on_pose(i, m), 10)
            self.create_subscription(Beacon, f'/px4_{i}/sentinel/beacon_out',
                                     lambda m: self._route(m.sender, m), 50)
            self.out[i] = self.create_publisher(Beacon, f'/px4_{i}/sentinel/beacon_in', 50)
        self.create_subscription(String, '/sentinel/fail', self._on_fail, 10)
        self.pub_stats = self.create_publisher(String, '/sentinel/comm_stats', 5)
        self.pub_base_ledger = self.create_publisher(LedgerSnapshot, '/sentinel/base/ledger', 5)
        self.sent = self.delivered = 0
        self.bytes = []
        self.create_timer(0.5, self._base_tick)
        self.create_timer(1.0, self._stats)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_pose(self, i, m):
        self.pos[i] = np.array([m.point.x, m.point.y])

    def _on_fail(self, m: String):
        ev = json.loads(m.data)
        u, kind = int(ev['uav']), ev['kind']
        if kind == 'crash':
            self.alive.discard(u)
        elif kind == 'link_cut':
            self.cut.add(u)
        elif kind == 'brownout':
            self.tx_offset_db[u] = -float(ev.get('drop_db', 20.0))
        self.get_logger().warn(f'failure event: {ev}')

    def _interference_mw(self, s, r):
        total = 0.0
        for k in self.alive:
            if k in (s, r) or k in self.cut or k not in self.pos or r not in self.pos:
                continue
            if self.rng.random() < self.profile.tx_activity:
                d = float(np.linalg.norm(self.pos[k] - self.pos[r]))
                obs = int(self.world.obstructions(self.pos[k], self.pos[r])[0])
                total += 10 ** ((self.profile.rx_power_dbm(d, obs)
                                 + self.tx_offset_db.get(k, 0.0)) / 10)
        return total

    def _link_ok(self, s, r) -> bool:
        if s not in self.pos or r not in self.pos:
            return False
        ps, pr = self.pos[s], self.pos[r]
        obs = int(self.world.obstructions(ps, pr)[0])
        sig = self.sampler.rx_power_mw(s, r, ps, pr, obs) * 10 ** (self.tx_offset_db.get(s, 0.0) / 10)
        return self.sampler.sinr_db(sig, self._interference_mw(s, r)) >= self.profile.gamma_th_db

    def _route(self, s, msg: Beacon):
        if s not in self.alive or s in self.cut:
            return
        self.bytes.append(len(msg.payload))
        for r in sorted(self.alive):
            if r == s or r in self.cut:
                continue
            self.sent += 1
            if not self._link_ok(s, r):
                continue
            self.delivered += 1
            if r == self.base_id:
                self.base.receive(pickle.loads(bytes(msg.payload)))
            else:
                out = Beacon()
                out.sender, out.receiver, out.stamp, out.payload = s, r, msg.stamp, msg.payload
                self.out[r].publish(out)

    def _base_tick(self):
        if self.base is None:
            return
        t = self._now()
        b = self.base.make_beacon(t, with_map=False)
        m = Beacon()
        m.sender, m.receiver, m.stamp = self.base_id, 255, t
        m.payload = list(pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL))
        self._route(self.base_id, m)
        led = LedgerSnapshot()
        led.uav_id, led.stamp = self.base_id, t
        es = list(self.base.ledger.entries.values())
        led.x = [float(e.pos[0]) for e in es]
        led.y = [float(e.pos[1]) for e in es]
        led.confidence = [float(e.confidence) for e in es]
        led.confirmed = [bool(e.confirmed) for e in es]
        self.pub_base_ledger.publish(led)

    def _stats(self):
        s = {'t': self._now(), 'sent': self.sent, 'delivered': self.delivered,
             'pdr': self.delivered / self.sent if self.sent else None,
             'payload_bytes_mean': float(np.mean(self.bytes)) if self.bytes else None,
             'payload_bytes_max': int(max(self.bytes)) if self.bytes else None}
        self.pub_stats.publish(String(data=json.dumps(s)))


def main(args=None):
    rclpy.init(args=args)
    node = CommEmulator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
