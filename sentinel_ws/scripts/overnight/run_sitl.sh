#!/usr/bin/env bash
# PX4/Gazebo cross-check runs (overnight 2026-10-28): E6 (10 worlds, YOLO26n in the loop) and the
# re-run of the original 3-world check (E8r, parameterised detector model) on the current code.
# Same protocol as scripts/run_e8_sitl.sh: 3 PX4 x500 UAVs, 100 m world, crash of UAV 1 at
# t = 150 s, 300 s of simulated time, SENTINEL and B0, tuned values from configs/tuned.yaml.
#
#   run_sitl.sh OUT PERCEPTION MODEL_PATH SEEDS...      (PERCEPTION: yolo | model)
# A run is skipped if its summary.json already covers the full duration (resumable).
set -o pipefail
OUT=$1; PERC=$2; MODEL=$3; shift 3
T=${SITL_T:-300}; N=3
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
GZS=$ROOT/src/sentinel_gazebo/scripts
TUNED=$ROOT/src/sentinel_sim/configs/tuned.yaml
mkdir -p "$OUT"
source /opt/ros/humble/setup.bash; source "$HOME/sentinel_ws/install/setup.bash"; source "$HOME/sentinel_venv/bin/activate"
export OMP_NUM_THREADS=2

cleanup_group() { [ -n "$1" ] && kill -TERM -- -"$1" 2>/dev/null; sleep 5; [ -n "$1" ] && kill -KILL -- -"$1" 2>/dev/null; }

for s in "$@"; do
  W=sentinel_g100_s$s
  if [ ! -f "$GZS/../worlds/$W.sdf" ]; then
    (cd "$GZS" && python3 make_world.py --kind generate --size 100 --targets 20 --seed "$s" --base \
       --out "$GZS/../worlds") || { echo "$(date +%T) WORLD FAILED $W"; continue; }
  fi
  for m in sentinel B0_pso; do
    R=$OUT/${W}_${m}
    [ -f "$R/summary.json" ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$R/summary.json')).get('duration_s',0)>=$T-5 else 1)" && { echo "skip $W $m"; continue; }
    rm -rf "$R"; mkdir -p "$R"
    export SENTINEL_RUN_DIR=$HOME/.ros/sentinel_overnight
    setsid "$GZS/launch_sitl.sh" "$N" "$W" > "$R/sitl.log" 2>&1 &
    SG=$!
    for i in $(seq 120); do grep -q "SITL running" "$R/sitl.log" && break; sleep 2; done
    setsid ros2 launch sentinel_ai swarm_launch.py n_uav:=$N world:=$W method:=$m seed:=$s \
       perception:=$PERC model_path:=$MODEL tuned_yaml:=$TUNED out_dir:=$R \
       failures:='[{"t":150,"uav":1,"kind":"crash"}]' run_dir:=$SENTINEL_RUN_DIR > "$R/launch.log" 2>&1 &
    LG=$!
    echo "$(date +%T) running $W $m ($PERC, sim ${T}s)"
    d=0
    for i in $(seq 480); do
      sleep 5
      d=$(python3 -c "import json;print(json.load(open('$R/summary.json')).get('duration_s',0))" 2>/dev/null || echo 0)
      python3 -c "import sys;sys.exit(0 if float('$d')>=$T else 1)" && break
    done
    cleanup_group "$LG"; cleanup_group "$SG"
    echo "$(date +%T) done $W $m duration=$d"
  done
done
echo "SITL DONE"
