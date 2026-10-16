"""Decentralised SENTINEL agent node (one per UAV).

Wraps ``sentinel_core.agent.SentinelAgent`` (the exact code evaluated in the
fast simulator): consumes the UAV pose and geolocated detections, exchanges
beacons only through ``comm_emulator`` and publishes world-frame setpoints to
``flight_commander``. There is no central coordinator.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from std_msgs.msg import Empty

from sentinel_core.channel import get_profile
from sentinel_core.maps import SearchMap
from sentinel_interfaces.msg import Beacon, LedgerSnapshot, TargetReport
from sentinel_sim import world as W
from sentinel_sim.methods import build_config


def load_tuned(path: str, method: str) -> dict:
    """Tuned per-method values (same file and inheritance rules as the fast simulator)."""
    if not path:
        return {}
    import yaml
    from sentinel_sim.runner import _method_overrides
    with open(path) as f:
        return _method_overrides(method, yaml.safe_load(f) or {})


def make_agent(uid, pos, truth, method, profile_name, n_nodes, seed, altitude, static=False,
               agent_overrides=None, tuned=None):
    world = W.from_truth(truth)
    tuned = tuned or {}
    over = {'altitude': altitude, 'is_static': static, **tuned.get('agent', {}),
            **(agent_overrides or {})}
    base = truth.get('base_xy')
    n_uav = n_nodes - (base is not None)
    if base is not None:
        over.update(base_id=n_uav, base_xy=tuple(base))
    over.setdefault('n_agents', n_uav)
    cls, cfg = build_config(method, agent_overrides=over, spring_overrides=tuned.get('spring'))
    lo, hi = world.bounds
    agent = cls(uid, pos, cfg, get_profile(profile_name),
                SearchMap(tuple(hi - lo), 2.0, n_nodes), (lo, hi),
                np.random.default_rng(seed + 7919 + uid), world.obstructions, world.in_no_fly)
    for k, v in tuned.get('agent_attrs', {}).items():
        setattr(agent, k, v)
    return agent


class EmpsAgent(Node):
    def __init__(self):
        super().__init__('emps_agent')
        p = self.declare_parameter
        self.uav = p('uav_id', 0).value
        n_uav = p('n_uav', 3).value
        truth = json.load(open(p('truth_json', '').value))
        self.dt = p('dt', 0.5).value
        method = p('method', 'sentinel').value
        tuned = load_tuned(p('tuned_yaml', '').value, method)
        over = {}
        t_hold = p('t_hold_s', -1.0).value
        if t_hold >= 0:
            over['t_hold_s'] = t_hold
        self.agent = make_agent(self.uav, (p('spawn_x', 0.0).value, p('spawn_y', 0.0).value),
                                truth, method,
                                p('channel_profile', 'urban_rubble').value,
                                n_uav + (truth.get('base_xy') is not None), p('seed', 0).value,
                                p('altitude', 30.0).value, agent_overrides=over, tuned=tuned)
        self.pending = []
        self.have_pose = False
        self.alt = 0.0
        self.pub_sp = self.create_publisher(Point, 'sentinel/setpoint', 10)
        self.pub_beacon = self.create_publisher(Beacon, 'sentinel/beacon_out', 10)
        self.pub_ledger = self.create_publisher(LedgerSnapshot, 'sentinel/ledger', 5)
        self.create_subscription(PointStamped, 'sentinel/pose', self._on_pose, 10)
        self.create_subscription(TargetReport, 'sentinel/targets', self._on_target, 50)
        self.create_subscription(Beacon, 'sentinel/beacon_in', self._on_beacon, 50)
        self.create_subscription(Empty, 'sentinel/kill', self._on_kill, 10)
        self.create_timer(self.dt, self._tick)
        self.create_timer(1.0, self._publish_ledger)
        self._k = 0

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_pose(self, m: PointStamped):
        self.agent.pos = np.array([m.point.x, m.point.y])
        self.alt = m.point.z
        self.have_pose = True

    def _on_target(self, m: TargetReport):
        self.pending.append((np.array([m.x, m.y]), float(m.confidence), -1))

    def _on_beacon(self, m: Beacon):
        if self.agent.alive:
            self.agent.receive(pickle.loads(bytes(m.payload)))

    def _on_kill(self, _):
        self.agent.alive = False

    def _tick(self):
        if not self.agent.alive or not self.have_pose:
            return
        t = self._now()
        airborne = self.alt > 0.8 * self.agent.cfg.altitude
        if airborne:
            dets, self.pending = self.pending, []
            self.agent.sense(dets, t)
        b = self.agent.make_beacon(t, with_map=self._k % 4 == 0)
        self._k += 1
        out = Beacon()
        out.sender, out.receiver, out.stamp = self.uav, 255, t
        out.payload = list(pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL))
        self.pub_beacon.publish(out)
        if not airborne:
            return
        measured = self.agent.pos.copy()
        target = self.agent.step(t, self.dt)
        self.agent.pos = measured            # the autopilot, not the planner, owns the state
        sp = Point()
        sp.x, sp.y, sp.z = float(target[0]), float(target[1]), float(self.agent.cfg.altitude)
        self.pub_sp.publish(sp)

    def _publish_ledger(self):
        m = LedgerSnapshot()
        m.uav_id, m.stamp = self.uav, self._now()
        es = list(self.agent.ledger.entries.values())
        m.x = [float(e.pos[0]) for e in es]
        m.y = [float(e.pos[1]) for e in es]
        m.confidence = [float(e.confidence) for e in es]
        m.confirmed = [bool(e.confirmed) for e in es]
        m.unreplicated = int(self.agent.ledger.unreplicated_count())
        m.reconnecting = bool(self.agent.reconnecting)
        self.pub_ledger.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = EmpsAgent()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
