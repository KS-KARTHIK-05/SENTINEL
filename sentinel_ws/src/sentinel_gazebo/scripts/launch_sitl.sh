#!/usr/bin/env bash
# Launch Gazebo Harmonic + N PX4 SITL x500_thermal instances + Micro XRCE-DDS agent.
#
#   launch_sitl.sh N WORLD [--gui]
#   e.g. launch_sitl.sh 3 sentinel_d25_s0 --gui
#
# - The PX4 tree is not modified: each PX4 instance runs from its own directory
#   under $SENTINEL_RUN_DIR, so px4-rc.gzsim does not re-source PX4's gz_env.sh
#   and our PX4_GZ_MODELS (containing x500_thermal) stays in effect.
# - ROS 2 namespaces: /px4_0 ... /px4_{N-1}; Gazebo models x500_thermal_<i>
#   (SENTINEL_UAV_MODEL=x500_plain: no camera, for model-based perception).
# - Spawn poses (Gazebo ENU world frame) are written to $SENTINEL_RUN_DIR/spawn.csv;
#   each PX4 local NED frame is centred at its own spawn point.
set -eo pipefail   # no -u: PX4 gz_env.sh appends to possibly-unset variables
N=${1:-3}
WORLD=${2:-sentinel_d25_s0}
GUI=${3:-}
PX4_DIR=${PX4_DIR:-$HOME/PX4-Autopilot}
BUILD=$PX4_DIR/build/px4_sitl_default
PKG=$(cd "$(dirname "$0")/.." && pwd)
RUN=${SENTINEL_RUN_DIR:-$HOME/.ros/sentinel_sitl}
WORLD_SDF=$PKG/worlds/$WORLD.sdf
TRUTH=$PKG/worlds/$WORLD.json
[ -f "$WORLD_SDF" ] || { echo "world $WORLD_SDF not found (run scripts/make_world.py)"; exit 1; }

# shellcheck disable=SC1091
source "$BUILD/gz_env.sh"
export PX4_GZ_MODELS=$PKG/models
export GZ_SIM_RESOURCE_PATH=$PKG/models:$PKG/worlds:$GZ_SIM_RESOURCE_PATH
rm -rf "$RUN" && mkdir -p "$RUN"

pids=()
cleanup() {
  # Stop only what this script started: its own process group (gz, agent, PX4
  # instances and their children). Never pattern-match other processes.
  echo "stopping SITL..."
  trap - EXIT INT TERM
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  PG=$(ps -o pgid= -p $$ | tr -d ' ')
  kill -TERM -- -"$PG" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

gz sim -s -r --headless-rendering "$WORLD_SDF" > "$RUN/gz.log" 2>&1 &
pids+=($!)
for _ in $(seq 60); do
  gz topic -l 2>/dev/null | grep -q "/world/$WORLD/clock" && break
  sleep 1
done
echo "gazebo world $WORLD up"
if [ "$GUI" = "--gui" ]; then gz sim -g > "$RUN/gz_gui.log" 2>&1 & pids+=($!); fi

MicroXRCEAgent udp4 -p 8888 > "$RUN/agent.log" 2>&1 &
pids+=($!)

python3 "$PKG/scripts/spawn_poses.py" "$TRUTH" "$N" > "$RUN/spawn.csv"
i=0
while IFS=, read -r x y; do
  mkdir -p "$RUN/px4_$i"
  (cd "$RUN/px4_$i" && env \
      PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4014 PX4_SIM_MODEL=gz_${SENTINEL_UAV_MODEL:-x500_thermal} \
      PX4_GZ_WORLD="$WORLD" PX4_GZ_MODEL_POSE="$x,$y,0.25,0,0,0" PX4_GZ_NO_FOLLOW=1 \
      PX4_UXRCE_DDS_NS="px4_$i" \
      PX4_PARAM_NAV_DLL_ACT=0 PX4_PARAM_NAV_RCL_ACT=0 PX4_PARAM_COM_RCL_EXCEPT=4 \
      PX4_PARAM_COM_RC_IN_MODE=4 PX4_PARAM_COM_OF_LOSS_T=2.0 \
      "$BUILD/bin/px4" -i "$i" -d "$BUILD/etc" > "$RUN/px4_$i.log" 2>&1) &
  pids+=($!)
  echo "PX4 instance $i at ($x, $y)"
  i=$((i + 1))
  sleep 2
done < "$RUN/spawn.csv"
echo "SITL running: $N vehicles in $WORLD. Logs in $RUN. Ctrl-C to stop."
wait
