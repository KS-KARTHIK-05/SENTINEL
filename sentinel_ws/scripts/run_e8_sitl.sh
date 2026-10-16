#!/usr/bin/env bash
# E8 SITL cross-validation: PX4 + Gazebo + ROS 2 stack with model-based perception on the
# same worlds/seeds as configs/experiments/e8_fastsim.yaml.
#   run_e8_sitl.sh [SIM_SECONDS=300] [N=3]
set -o pipefail
T=${1:-300}; N=${2:-3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GZS=$ROOT/src/sentinel_gazebo/scripts
TUNED=$ROOT/src/sentinel_sim/configs/tuned.yaml
OUT=$HOME/sentinel_ws/results/e8_sitl
mkdir -p "$OUT"
source /opt/ros/humble/setup.bash; source "$HOME/sentinel_ws/install/setup.bash"; source "$HOME/sentinel_venv/bin/activate"
export OMP_NUM_THREADS=2

cleanup_group() { [ -n "$1" ] && kill -TERM -- -"$1" 2>/dev/null; sleep 5; [ -n "$1" ] && kill -KILL -- -"$1" 2>/dev/null; }

for s in 0 1 2; do for m in sentinel B0_pso; do
  W=sentinel_g100_s$s; R=$OUT/${W}_${m}
  [ -f "$R/summary.json" ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$R/summary.json')).get('duration_s',0)>=$T-5 else 1)" && { echo "skip $W $m"; continue; }
  rm -rf "$R"; mkdir -p "$R"
  export SENTINEL_RUN_DIR=$HOME/.ros/sentinel_e8
  setsid "$GZS/launch_sitl.sh" "$N" "$W" > "$R/sitl.log" 2>&1 &
  SG=$!
  for i in $(seq 120); do grep -q "SITL running" "$R/sitl.log" && break; sleep 2; done
  setsid ros2 launch sentinel_ai swarm_launch.py n_uav:=$N world:=$W method:=$m seed:=$s perception:=model \
     tuned_yaml:=$TUNED out_dir:=$R failures:='[{"t":150,"uav":1,"kind":"crash"}]' run_dir:=$SENTINEL_RUN_DIR > "$R/launch.log" 2>&1 &
  LG=$!
  echo "$(date +%T) running $W $m (sim ${T}s)"
  for i in $(seq 360); do
    sleep 5
    d=$(python3 -c "import json;print(json.load(open('$R/summary.json')).get('duration_s',0))" 2>/dev/null || echo 0)
    python3 -c "import sys;sys.exit(0 if float('$d')>=$T else 1)" && break
  done
  cleanup_group "$LG"; cleanup_group "$SG"
  echo "$(date +%T) done $W $m duration=$d"
done; done
echo "E8 SITL DONE"
