"""PX4 offboard interface for one UAV.

Publishes the true world-frame pose (fixing v1, whose telemetry was always
(0, 0)) and flies to world-frame setpoints produced by ``emps_agent``.
Arming/offboard sequence follows v1: stream setpoints, switch to offboard,
arm. A message on ``sentinel/kill`` force-disarms the vehicle (chaos tests).
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, PointStamped
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                          VehicleLocalPosition, VehicleStatus)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty

from .frames import ned_to_world, world_to_ned

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


class FlightCommander(Node):
    def __init__(self):
        super().__init__('flight_commander')
        self.uav = self.declare_parameter('uav_id', 0).value
        self.spawn = (self.declare_parameter('spawn_x', 0.0).value,
                      self.declare_parameter('spawn_y', 0.0).value)
        self.alt = self.declare_parameter('altitude', 30.0).value
        lp_topic = self.declare_parameter('local_position_topic',
                                          'fmu/out/vehicle_local_position_v1').value
        st_topic = self.declare_parameter('status_topic', 'fmu/out/vehicle_status_v4').value
        self.sys_id = self.uav + 1

        self.pub_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', 10)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', 10)
        self.pub_cmd = self.create_publisher(VehicleCommand, 'fmu/in/vehicle_command', 10)
        self.pub_pose = self.create_publisher(PointStamped, 'sentinel/pose', 10)
        self.create_subscription(VehicleLocalPosition, lp_topic, self._on_lp, PX4_QOS)
        self.create_subscription(VehicleStatus, st_topic, self._on_status, PX4_QOS)
        self.create_subscription(Point, 'sentinel/setpoint', self._on_setpoint, 10)
        self.create_subscription(Empty, 'sentinel/kill', self._on_kill, 10)

        self.target_world = None          # (x, y) world ENU
        self.pose_world = None
        self.nav_state = None
        self.armed = False
        self.killed = False
        self.ticks = 0
        self.valid_since = None          # time from which PX4 has a valid position estimate
        self.create_timer(0.1, self._tick)

    # ------------------------------------------------------------- callbacks
    def _on_lp(self, m: VehicleLocalPosition):
        if not (m.xy_valid and m.z_valid):
            self.valid_since = None
            return
        if self.valid_since is None:
            self.valid_since = self.get_clock().now().nanoseconds * 1e-9
        x, y, z = ned_to_world(self.spawn, (m.x, m.y, m.z))
        self.pose_world = (x, y, z)
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.point.x, msg.point.y, msg.point.z = x, y, z
        self.pub_pose.publish(msg)

    def _on_status(self, m: VehicleStatus):
        self.nav_state = m.nav_state
        self.armed = m.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def _on_setpoint(self, m: Point):
        self.target_world = (m.x, m.y)

    def _on_kill(self, _):
        self.get_logger().warn(f'UAV {self.uav}: kill received, force disarm')
        self.killed = True
        # MAV_CMD_COMPONENT_ARM_DISARM, param2 = 21196 forces disarm in flight
        self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 21196.0)

    # ----------------------------------------------------------------- loop
    def _tick(self):
        if self.killed:
            return
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        mode = OffboardControlMode()
        mode.position = True
        mode.timestamp = now_us
        self.pub_mode.publish(mode)

        if self.pose_world is None:
            tx, ty = self.spawn                                  # no estimate yet
        elif self.pose_world[2] < 0.8 * self.alt:
            tx, ty = self.pose_world[0], self.pose_world[1]      # climb in place first
        elif self.target_world is None:
            tx, ty = self.pose_world[0], self.pose_world[1]      # hover until planned
        else:
            tx, ty = self.target_world
        n, e, d = world_to_ned(self.spawn, tx, ty, self.alt)
        sp = TrajectorySetpoint()
        sp.position = [float(n), float(e), float(d)]
        sp.yaw = 0.0
        sp.timestamp = now_us
        self.pub_sp.publish(sp)

        self.ticks += 1
        # Arm only after the estimator has delivered a valid position for 3 s: arming on a
        # fixed timer let a UAV take off without a position estimate under heavy CPU load
        # (PX4 "blind land" failsafe).
        now = self.get_clock().now().nanoseconds * 1e-9
        estimator_ready = self.valid_since is not None and now - self.valid_since >= 3.0
        if estimator_ready and self.ticks % 10 == 0 and (
                self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD or not self.armed):
            self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def _command(self, command, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.command = command
        m.param1, m.param2 = float(p1), float(p2)
        m.target_system = self.sys_id
        m.target_component = 1
        m.source_system = 1
        m.source_component = 1
        m.from_external = True
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_cmd.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = FlightCommander()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
