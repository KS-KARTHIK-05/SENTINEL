#!/usr/bin/env python3
"""Save raw thermal frames (no detector boxes) from a running Gazebo demo, for the paper figure.

Uses the same radiometric conversion as ``thermal_vision`` (L16, 0.01 K per count, 270-330 K
mapped to 8-bit white-hot), so the saved frames are exactly what the detector sees.

    python3 save_thermal_frames.py                      # UAV 0, world sentinel_g100_s0
    python3 save_thermal_frames.py --uav 2 --every 1.5 --count 60

Frames go to ~/sentinel_ws/results/thermal_frames/uav<i>/frame_<sim time>.png
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Saver(Node):
    def __init__(self, a):
        super().__init__('thermal_frame_saver')
        self.a, self.bridge, self.last, self.saved = a, CvBridge(), None, 0
        self.out = os.path.expanduser(f'~/sentinel_ws/results/thermal_frames/uav{a.uav}')
        os.makedirs(self.out, exist_ok=True)
        topic = (f'/world/{a.world}/model/x500_thermal_{a.uav}/link/thermal_link'
                 '/sensor/thermal_camera/image')
        self.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info(f'saving every {a.every} s from {topic} -> {self.out}')

    def _on_image(self, msg: Image):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last is not None and t - self.last < self.a.every:
            return
        self.last = t
        img = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))
        if img.dtype == np.uint16:      # same conversion as thermal_vision
            temp = img.astype(np.float32) * 0.01
            img = np.clip((temp - 270.0) / (330.0 - 270.0) * 255.0, 0, 255).astype(np.uint8)
        path = os.path.join(self.out, f'frame_{t:08.1f}.png')
        cv2.imwrite(path, img)
        self.saved += 1
        self.get_logger().info(f'{self.saved}/{self.a.count} {path}')
        if self.saved >= self.a.count:
            raise SystemExit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uav', type=int, default=0)
    ap.add_argument('--world', default='sentinel_g100_s0')
    ap.add_argument('--every', type=float, default=2.0, help='seconds of sim time between frames')
    ap.add_argument('--count', type=int, default=60)
    a = ap.parse_args()
    rclpy.init()
    node = Saver(a)
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
