import sys
import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from geometry_msgs.msg import Point

class SentinelCommander(Node):
    def __init__(self, namespace=''):
        node_name = 'sentinel_commander' if not namespace else f'sentinel_commander_{namespace.strip("/")}'
        super().__init__(node_name)

        self.namespace = namespace
        
        # --- FIX 1: DYNAMIC MAVLINK SYSTEM ID ---
        drone_id_str = self.namespace.replace('/px4_', '') if self.namespace else '0'
        try:
            self.sys_id = int(drone_id_str) + 1
        except ValueError:
            self.sys_id = 1

        # --- FIX 2: PX4 NAMESPACE QUIRK ---
        px4_prefix = '' if self.namespace == '/px4_0' else self.namespace

        # Dynamically prefix the namespace to the PX4 topics
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, f'{px4_prefix}/fmu/in/offboard_control_mode', 10)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, f'{px4_prefix}/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, f'{px4_prefix}/fmu/in/vehicle_command', 10)

        clean_namespace = self.namespace.strip("/") if self.namespace else "px4_0"
        
        self.error_sub = self.create_subscription(
            Point, f'/{clean_namespace}/target_error', self.error_callback, 10)

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.nav_state = "INIT"
        self.takeoff_altitude = -30.0
        self.timer_count = 0

        self.target_x_pos = 0.0 
        self.target_y_pos = 0.0 
        self.err_x = 0.0
        self.err_y = 0.0
        self.last_target_time = self.get_clock().now()
        
        self.kp = 0.005      
        self.deadband = 15.0 
        
        telemetry_topic = f'{self.namespace}/telemetry' if self.namespace else '/px4_0/telemetry'
        self.telemetry_pub = self.create_publisher(Point, telemetry_topic, 10)
        self.heartbeat_timer = self.create_timer(1.0, self.publish_telemetry)

    def error_callback(self, msg):
        self.err_x = msg.x
        self.err_y = msg.y
        self.last_target_time = self.get_clock().now()

    def timer_callback(self):
        self.publish_offboard_control_heartbeat()
        
        # --- AGGRESSIVE STATE MACHINE ---
        if self.nav_state == "INIT":
            self.publish_trajectory_setpoint(0.0, 0.0, self.takeoff_altitude)
            
            # Tick 20 to 50: Aggressive ARM Spamming
            if 20 <= self.timer_count <= 50:
                if self.timer_count % 5 == 0: 
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                    self.get_logger().info(f"Aggressive Ignition Pulse sent to SYS_ID {self.sys_id}...")
                    
            if self.timer_count > 50:
                self.nav_state = "TRACKING"
                self.get_logger().info(f"SENTINEL {self.namespace} LOCKED IN TRACKING MODE.")
                
            self.timer_count += 1

        elif self.nav_state == "TRACKING":
            time_since_target = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
            
            if time_since_target < 0.5:
                vx = 0.0
                vy = 0.0
                
                if abs(self.err_x) > self.deadband:
                    vy = self.kp * self.err_x
                    
                if abs(self.err_y) > self.deadband:
                    vx = -self.kp * self.err_y 
                
                self.target_x_pos += vx * 0.1 
                self.target_y_pos += vy * 0.1
                
            self.publish_trajectory_setpoint(self.target_x_pos, self.target_y_pos, self.takeoff_altitude)

    def publish_offboard_control_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 0.0 
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        
        msg.target_system = self.sys_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)
        
    def publish_telemetry(self):
        msg = Point()
        msg.x = float(getattr(self, 'current_x', 0.0))
        msg.y = float(getattr(self, 'current_y', 0.0))
        msg.z = float(getattr(self, 'current_z', -30.0))
        self.telemetry_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    import sys
    
    namespace = ''
    for arg in sys.argv[1:]:
        if arg.startswith('/px4_'):
            namespace = arg
            break
            
    try:
        commander = SentinelCommander(namespace)
        rclpy.spin(commander)
    except Exception as e:
        print(f"Commander Error: {e}")
    finally:
        commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    
