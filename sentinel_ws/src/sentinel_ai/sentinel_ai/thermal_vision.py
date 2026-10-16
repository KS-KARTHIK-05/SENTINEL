"""Onboard thermal perception for one UAV.

Runs the detector on the nadir thermal image and publishes every detection
geolocated to the world frame (``sentinel/targets``). v1 published a
camera-frame pixel offset scaled by a constant, which made the shared ledger
grow with every viewpoint (the "15 coordinates" inconsistency); here the
position comes from the camera model, vehicle pose and heading.
"""
from __future__ import annotations

import math
import os

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from px4_msgs.msg import VehicleAttitude
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from sentinel_interfaces.msg import DetectorStats, TargetReport

from .detector import ThermalDetector
from .flight_commander import PX4_QOS
from .frames import pixel_to_world, yaw_from_quat_ned


def _default_model() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('sentinel_ai'), 'models',
                            'yolo26n_thermal_best.pt')
    except Exception:  # noqa: BLE001
        return 'yolo26n_thermal_best.pt'


class ThermalVision(Node):
    def __init__(self):
        super().__init__('thermal_vision')
        p = self.declare_parameter
        self.uav = p('uav_id', 0).value
        model = p('model_path', '').value or _default_model()
        self.hfov = p('hfov', 0.8727).value
        self.cam_axes = tuple(p('cam_axes', [-1.0, 1.0]).value)
        self.min_alt = p('min_altitude', 10.0).value
        # radiometric L16 (0.01 K per count) -> 8-bit white-hot, as in HIT-UAV imagery
        self.t_min = p('t_min', 270.0).value
        self.t_max = p('t_max', 330.0).value
        self.k_per_count = p('kelvin_per_count', 0.01).value
        self.debug = p('publish_debug', False).value     # annotated frames for figures
        image_topic = p('image_topic', '').value or (
            f'/world/{p("world", "default").value}/model/x500_thermal_{self.uav}'
            '/link/thermal_link/sensor/thermal_camera/image')
        self.det = ThermalDetector(model, imgsz=p('imgsz', 640).value,
                                   conf_floor=p('conf_floor', 0.25).value,
                                   device=p('device', '').value,
                                   preprocess=p('preprocess', 'clahe').value,
                                   invert=p('invert', False).value)
        self.get_logger().info(f'UAV {self.uav}: detector {model} on {image_topic}')
        self.bridge = CvBridge()
        self.pose = None
        self.yaw = 0.0
        self.pub = self.create_publisher(TargetReport, 'sentinel/targets', 20)
        self.pub_stats = self.create_publisher(DetectorStats, 'sentinel/detector_stats', 5)
        self.pub_dbg = (self.create_publisher(Image, 'sentinel/thermal_annotated', 2)
                        if self.debug else None)
        self.create_subscription(PointStamped, 'sentinel/pose', self._on_pose, 10)
        self.create_subscription(VehicleAttitude, 'fmu/out/vehicle_attitude', self._on_att, PX4_QOS)
        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.create_timer(5.0, self._publish_stats)

    def _on_pose(self, m: PointStamped):
        self.pose = (m.point.x, m.point.y, m.point.z)

    def _on_att(self, m: VehicleAttitude):
        self.yaw = yaw_from_quat_ned(m.q)

    def _on_image(self, msg: Image):
        if self.pose is None or self.pose[2] < self.min_alt:
            return
        img = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))
        if img.dtype == np.uint16:
            temp = img.astype(np.float32) * self.k_per_count
            img = np.clip((temp - self.t_min) / (self.t_max - self.t_min) * 255.0,
                          0, 255).astype(np.uint8)
        pose, yaw = self.pose, self.yaw
        dets = self.det(img)
        if self.pub_dbg is not None:
            self._publish_debug(img, dets, msg.header)
        for d in dets:
            x, y, rng = pixel_to_world(d.u, d.v, msg.width, msg.height, self.hfov, pose, yaw,
                                       self.cam_axes)
            r = TargetReport()
            r.header = msg.header
            r.header.frame_id = 'world'
            r.uav_id = self.uav
            r.x, r.y = float(x), float(y)
            r.confidence = float(d.conf)
            r.slant_range = float(rng)
            r.uav_x, r.uav_y, r.uav_z = map(float, pose)
            self.pub.publish(r)

    def _publish_debug(self, img, dets, header):
        import cv2
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        for d in dets:
            x0, y0 = int(d.u - d.w / 2), int(d.v - d.h / 2)
            x1, y1 = int(d.u + d.w / 2), int(d.v + d.h / 2)
            colour = (0, 200, 0) if d.conf >= 0.70 else (0, 200, 255) if d.conf >= 0.40 else (0, 0, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), colour, 2)
            cv2.putText(vis, f'victim {d.conf:.2f}', (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        out.header = header
        self.pub_dbg.publish(out)

    def _publish_stats(self):
        s = self.det.stats()
        if not s['frames']:
            return
        m = DetectorStats()
        m.uav_id = self.uav
        m.model = os.path.basename(self.det.model_path)
        for k in ('preprocess_ms_p50', 'inference_ms_p50', 'inference_ms_p95', 'total_ms_p50',
                  'total_ms_p95'):
            setattr(m, k, float(s[k]) if math.isfinite(s[k]) else 0.0)
        m.frames = int(s['frames'])
        self.pub_stats.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = ThermalVision()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
