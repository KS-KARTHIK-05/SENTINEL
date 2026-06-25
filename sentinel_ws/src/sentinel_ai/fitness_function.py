import numpy as np

def calculate_fitness(drone_pos, swarm_positions, target_confidence, max_range=50.0):
    """
    Refined EM-PSO Fitness Function for Project SENTINEL.
    Calculates search reward vs signal penalty to maintain mesh integrity.
    """
    drone_pos = np.array(drone_pos)
    
    # 1. Search Reward (Attraction to victim)
    reward = target_confidence 
    
    # 2. Multi-Neighbor Spring Penalty
    # Ensures the drone is within range of at least ONE neighbor
    distances = [np.linalg.norm(drone_pos - np.array(p)) for p in swarm_positions]
    min_dist = min(distances) if distances else 0
    
    penalty = 0
    if min_dist > max_range:
        k = 0.8  # Elastic constant (stiffness)
        penalty = k * (min_dist - max_range)**2
        
    # Final Fitness: Reward for detection, penalty for isolation
    fitness = reward - penalty
    return fitness, min_dist

def get_sa_offset_coords(current_pos, target_pos, baseline=15.0):
    """
    Calculates coordinates for two neighboring drones to perform a 
    Synthetic Aperture (SA) sweep to 'see through' occlusions.
    
    baseline: The lateral distance (meters) between drones to create the 'virtual lens'.
    """
    curr = np.array(current_pos)
    tgt = np.array(target_pos)
    
    # 1. Get the direction vector to the suspected victim
    direction = tgt - curr
    direction_norm = direction / np.linalg.norm(direction)
    
    # 2. Calculate the 'Perpendicular' vector for lateral offsets
    # This creates the multi-angle 'Synthetic' baseline
    perpendicular = np.array([-direction_norm[1], direction_norm[0]])
    
    # 3. Generate two points (Left and Right) at 'baseline' distance
    helper_pos_a = curr + (perpendicular * baseline)
    helper_pos_b = curr - (perpendicular * baseline)
    
    return helper_pos_a.tolist(), helper_pos_b.tolist()

# --- Example Usage for Verification ---
my_pos = [0, 0]
suspected_victim = [100, 0] # 100m away
helper_a, helper_b = get_sa_offset_coords(my_pos, suspected_victim)

print(f"Drone A SA Position: {helper_a}") # Expected: [0, 15]
print(f"Drone B SA Position: {helper_b}") # Expected: [0, -15]