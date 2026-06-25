# SENTINEL: Edge-AI Driven Elastic Mesh PSO

## Overview
SENTINEL is a decentralized, zero-infrastructure UAV swarm framework for post-disaster Search and Rescue (SAR). It bridges the gap between AI-driven target perception and mesh-network stability.

## 🧠 System Architecture
* **Perception:** YOLOv8-Nano (1.2ms latency) with radiometric normalization for high-fidelity TIR target detection.
* **Mobility:** Elastic-Mesh PSO (EM-PSO) using a quadratic Signal-Spring penalty to prevent swarm isolation.
* **Collaboration:** Decentralized Synthetic Aperture (SA) triggers for verifying occluded signatures.

## 🚀 Key Performance
| Metric | Baseline PSO | SENTINEL (Full) |
| :--- | :---: | :---: |
| Network Retention | 66% | 100% |
| Target Discovery | 60% | 90% |

## 📁 Repository Structure
```text
/
├── ros2_ws/src/
│   ├── sentinel_ai/          # Python perception & YOLO inference node
│   ├── sentinel_mobility/    # C++ EM-PSO coordination logic
│   ├── sentinel_gazebo/      # Simulation launch files & disaster world
│   └── sentinel_interfaces/  # Custom ROS 2 msg definitions
├── scripts/
│   ├── chaos_monkey.sh       # Resilience testing protocol
│   └── plot_metrics.py       # Ablation study visualization
└── requirements.txt          # AI dependencies
