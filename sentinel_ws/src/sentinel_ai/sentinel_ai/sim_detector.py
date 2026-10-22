"""Model-based perception for SITL cross-validation (experiment E8).

Publishes ``sentinel/targets`` from the ground-truth victim positions of the
generated world, passed through the same stochastic detector model as the fast
simulator (``sentinel_core.detector_model``): confidence depends on slant range
and view-dependent occlusion, with warm-clutter false positives. Used instead
of ``thermal_vision`` when the experiment compares SITL with the fast simulator
on identical perception, so that differences come from flight dynamics, PX4
control and ROS 2 timing only. The YOLO pipeline (``thermal_vision``) is the
deployment path; its accuracy is reported on real HIT-UAV data.
"""
from __future__ import annotations

import json

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from sentinel_core.detector_model import DetectorModel
from sentinel_interfaces.msg import TargetReport
from sentinel_sim import world as W


class SimDetector(Node):
    def __init__(self):
        super().__init__('sim_detector')
        p = self.declare_parameter
        self.uav = p('uav_id', 0).value
        self.world = W.from_truth(json.load(open(p('truth_json', '').value)))
        self.footprint = p('footprint_r', 12.0).value
        self.min_alt = p('min_altitude', 20.0).value
        self.det = DetectorModel()
        self.rng = np.random.default_rng(p('seed', 0).value * 1000 + self.uav)
        self.pose = None
        self.pub = self.create_publisher(TargetReport, 'sentinel/targets', 20)
        self.create_subscription(PointStamped, 'sentinel/pose', self._on_pose, 10)
        self.create_timer(1.0 / p('rate_hz', 1.0).value, self._tick)

    def _on_pose(self, m: PointStamped):
        self.pose = np.array([m.point.x, m.point.y, m.point.z])

    def _tick(self):
        if self.pose is None or self.pose[2] < self.min_alt:
            return
        for pos, conf, _ in self.det.look(self.world.targets, self.pose, self.footprint, self.rng,
                                          self.world.clutter_xy):
            r = TargetReport()
            r.header.stamp = self.get_clock().now().to_msg()
            r.header.frame_id = 'world'
            r.uav_id = self.uav
            r.x, r.y = float(pos[0]), float(pos[1])
            r.confidence = float(conf)
            r.slant_range = float(np.linalg.norm(np.r_[self.pose[:2] - pos, self.pose[2]]))
            r.uav_x, r.uav_y, r.uav_z = map(float, self.pose)
            self.pub.publish(r)


def main(args=None):
    rclpy.init(args=args)
    node = SimDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
