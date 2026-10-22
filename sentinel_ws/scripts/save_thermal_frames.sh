#!/usr/bin/env bash
# Save raw white-hot thermal frames (no YOLO boxes) from the running demo, for screenshot S4.
# Usage: ~/sentinel_ws/save_thermal_frames.sh [--uav 0] [--every 2] [--count 60]
source /opt/ros/humble/setup.bash
source ~/sentinel_ws/install/setup.bash
source ~/sentinel_venv/bin/activate
exec python3 ~/sentinel_ws/src/SENTINEL/sentinel_ws/scripts/save_thermal_frames.py "$@"
