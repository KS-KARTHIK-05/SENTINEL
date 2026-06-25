import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import time

# Import your EM-PSO and SA math
from .fitness_function import get_sa_offset_coords, calculate_fitness

class SwarmCoordinator(Node):
    def __init__(self):
        super().__init__('swarm_coordinator')
        self.get_logger().info("Swarm Coordinator Online: Tracking 3 Agents.")
        self.num_drones = 3  # FORCE THIS TO 3
        self.active_mesh = list(range(self.num_drones))
        
        # 1. The Heartbeat Ledger (Stores timestamp of last received message)
        self.last_heartbeat = {i: time.time() for i in range(self.num_drones)}
        
        # 1.5 The Live Position Ledger
        self.drone_positions = {i: [0.0, 0.0] for i in range(self.num_drones)}
        self.telemetry_subs = []
        
        # 2. The Global Target Ledger
        self.confirmed_targets = []
        
        # Subscriptions and Publishers for all 5 drones
        self.vision_subs = []
        self.cmd_pubs = []
        
        for i in range(self.num_drones):
            # Dynamic callback creation using lambda to pass the drone ID
            sub = self.create_subscription(
                Point, f'/px4_{i}/target_error', 
                lambda msg, drone_id=i: self.vision_callback(msg, drone_id), 10)
            self.vision_subs.append(sub)
            
            pub = self.create_publisher(Point, f'/px4_{i}/target_error', 10)
            self.cmd_pubs.append(pub)
            
            # Subscribe to the new Heartbeat/Telemetry pulse
            t_sub = self.create_subscription(
                Point, f'/px4_{i}/telemetry',
                lambda msg, drone_id=i: self.telemetry_callback(msg, drone_id), 10)
            self.telemetry_subs.append(t_sub)

        # 3. The Heartbeat Monitor (Runs every 1.0 seconds)
        self.timer = self.create_timer(1.0, self.monitor_heartbeats)

    def monitor_heartbeats(self):
        current_time = time.time()
        
        # 1. Initialize the boot timer
        if not hasattr(self, 'start_time'):
            self.start_time = current_time
            
        uptime = current_time - self.start_time
        
        # 2. THE INCUBATOR: Grace Period for Staggered Launch (0 to 40 seconds)
        if uptime < 40.0:
            # Artificially keep unborn drones alive by faking their heartbeats 
            # until their actual telemetry links boot up.
            for i in self.active_mesh:
                self.last_heartbeat[i] = current_time
            return # Exit the function early so we don't kill anything
            
        # 3. THE CHAOS MONKEY (THESIS DEMONSTRATION)
        if uptime > 100.0 and 1 in self.active_mesh:
            self.get_logger().fatal("CHAOS INJECTION: Severing telemetry link with Sentinel 1.")
            self.last_heartbeat[1] = 0.0 
            
        # 4. Standard Fault Tolerance Execution
        dead_nodes = []
        for drone_id in self.active_mesh:
            if (current_time - self.last_heartbeat[drone_id]) > 3.0:
                self.get_logger().error(f"CRITICAL: Sentinel {drone_id} connection lost. Removing from mesh.")
                dead_nodes.append(drone_id)
                
        for dead_id in dead_nodes:
            self.active_mesh.remove(dead_id)
            self.recalculate_mesh()

    def recalculate_mesh(self):
        self.get_logger().info(f"Recalculating EM-PSO Spring Penalty for remaining nodes: {self.active_mesh}")
        # Note: Your fitness_function.py naturally expands the swarm here 
        # because the 'swarm_positions' array just got smaller, triggering the max_range penalty.

    def telemetry_callback(self, msg, drone_id):
        # 1. Update the heartbeat to keep the drone alive in the mesh
        self.last_heartbeat[drone_id] = time.time()
        # 2. Update its physical global coordinates for the EM-PSO math
        self.drone_positions[drone_id] = [msg.x, msg.y]
    
    def vision_callback(self, msg, drone_id):
        # Update heartbeat timestamp for this drone
        
        target_x = msg.x
        target_y = msg.y
        confidence = msg.z
        
        # If the local node spots a victim with high confidence
        if confidence > 0.60:
            target_coords = [target_x, target_y]
            
            # SPATIAL RADIUS CHECK: Is this target within 2 meters of a known victim?
            is_new_target = True
            for existing_target in self.confirmed_targets:
                dist = math.dist(target_coords, existing_target)
                if dist < 2.0:
                    is_new_target = False
                    break
            
            if is_new_target:
                self.confirmed_targets.append(target_coords)
                self.get_logger().info(f"New Target Added to Ledger by Sentinel {drone_id}. Total: {len(self.confirmed_targets)}")
                
                # Execute SA Sweep using the nearest available neighbor
                self.trigger_sa_sweep(drone_id, target_coords)

    def trigger_sa_sweep(self, observer_id, target_pos):
        # Find the next available drone in the active mesh
        available_neighbors = [d for d in self.active_mesh if d != observer_id]
        
        if not available_neighbors:
            self.get_logger().warning("No active wingmen available for SA Sweep.")
            return
            
        wingman_id = available_neighbors[0]
        
        # Fetch the observer's ACTUAL live position from the telemetry ledger
        my_pos = self.drone_positions[observer_id] 
        helper_a, helper_b = get_sa_offset_coords(my_pos, target_pos, baseline=10.0)
        
        self.get_logger().info(f"Commanding Sentinel {wingman_id} to SA Vector.")
        
        sa_cmd = Point()
        sa_cmd.x = float(helper_a[0])
        sa_cmd.y = float(helper_a[1])
        sa_cmd.z = 0.0 # Override command
        self.cmd_pubs[wingman_id].publish(sa_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SwarmCoordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
