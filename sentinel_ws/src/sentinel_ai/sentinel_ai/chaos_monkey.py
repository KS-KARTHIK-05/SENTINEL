"""Failure injection for SITL runs (replaces v1's hard-coded kill of UAV 1).

``schedule`` is a JSON list of events {"t": s, "uav": i, "kind": crash|link_cut|brownout,
"drop_db": x}, with t relative to node start. A crash force-disarms the vehicle
(it falls) and removes it from the radio network; link_cut silences its radio
while it keeps flying; brownout lowers its transmit power.
"""
from __future__ import annotations

import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty, String


class ChaosMonkey(Node):
    def __init__(self):
        super().__init__('chaos_monkey')
        self.events = sorted(json.loads(self.declare_parameter('schedule', '[]').value),
                             key=lambda e: e['t'])
        self.pub_fail = self.create_publisher(String, '/sentinel/fail', 10)
        self.kill = {}
        self.t0 = None
        self.create_timer(0.1, self._tick)

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        while self.events and now - self.t0 >= self.events[0]['t']:
            ev = self.events.pop(0)
            self.get_logger().fatal(f'CHAOS: {ev}')
            self.pub_fail.publish(String(data=json.dumps({**ev, 't_abs': now})))
            if ev['kind'] == 'crash':
                u = int(ev['uav'])
                if u not in self.kill:
                    self.kill[u] = self.create_publisher(Empty, f'/px4_{u}/sentinel/kill', 10)
                self.kill[u].publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    node = ChaosMonkey()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
