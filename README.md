# SENTINEL: Edge-AI Driven Elastic Mesh PSO for Low-Altitude UAV IoT Networks

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Gazebo](https://img.shields.io/badge/Simulator-Gazebo_Harmonic-orange.svg)](https://gazebosim.org/)
[![YOLOv8](https://img.shields.io/badge/Edge_AI-YOLOv8--Nano-yellow.svg)](https://github.com/ultralytics/ultralytics)

**SENTINEL** is a decentralized, zero-infrastructure UAV swarm architecture designed for post-disaster Search and Rescue (SAR). It solves the "Data-Loss Paradox" by strictly coupling Edge AI perception with RF-aware swarm mobility.

## 🏗️ System Architecture
* **Edge Perception Layer:** YOLOv8-Nano trained on the HIT-UAV thermal dataset, executing 1.2ms inference at the edge with radiometric normalization.
* **Swarm Mobility Layer:** Elastic-Mesh Particle Swarm Optimization (EM-PSO) algorithm that applies a mathematical quadratic penalty to maintain a 50m RF communication boundary.
* **Collaborative Edge Sensing (SA Trigger):** Autonomous lateral flanking ($\pm 15m$) triggered by medium-confidence thermal anomalies to bypass line-of-sight occlusions.

## ⚙️ Setup and Installation
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/SENTINEL-UAV-Swarm.git](https://github.com/YOUR_USERNAME/SENTINEL-UAV-Swarm.git)
cd SENTINEL-UAV-Swarm

# Install dependencies
pip install -r requirements.txt

# Build workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
\`\`\`
