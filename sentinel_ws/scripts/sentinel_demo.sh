#!/usr/bin/env bash
# One-command SENTINEL demo: Gazebo GUI + N PX4 drones + full SENTINEL stack + viewers.
#
#   ~/sentinel_ws/run_sentinel_demo.sh                  # defaults: 6 drones, world sentinel_g100_s0, YOLO26n
#   ~/sentinel_ws/run_sentinel_demo.sh -n 3 -w sentinel_d25_s0 -p model -t 400 --no-crash
#
# Options:
#   -n N          number of drones (default 6; this PC handles up to 9)
#   -w WORLD      sentinel_g100_s0 | sentinel_g100_s1 | sentinel_g100_s2 | sentinel_g200_s0 | sentinel_d25_s0
#   -p MODE       yolo (YOLO26n on the Gazebo thermal camera, default) | model (calibrated detector model)
#   -m METHOD     sentinel (default) | B0_pso | B6_lawnmower | ... (any method name)
#   -t SECONDS    stop automatically after this many seconds (default: run until Ctrl-C)
#   --no-crash    do not crash UAV 1 at t = 150 s
#   --headless    no Gazebo GUI / viewer windows (for remote or automated runs)
# Everything stops cleanly on Ctrl-C. Metrics: ~/sentinel_ws/results/demo_<timestamp>/
set -o pipefail
N=6; WORLD=sentinel_g100_s0; PERC=yolo; METHOD=sentinel; DUR=0; CRASH=1; GUI=1
while [ $# -gt 0 ]; do
  case "$1" in
    -n) N=$2; shift 2 ;; -w) WORLD=$2; shift 2 ;; -p) PERC=$2; shift 2 ;;
    -m) METHOD=$2; shift 2 ;; -t) DUR=$2; shift 2 ;; --no-crash) CRASH=0; shift ;;
    --headless) GUI=0; shift ;; -h|--help) sed -n 2,15p "$0"; exit 0 ;;
    *) echo "unknown option $1 (use -h)"; exit 1 ;;
  esac
done

WS=$HOME/sentinel_ws
SCRIPTS=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
GZS=$SCRIPTS/../src/sentinel_gazebo/scripts
TUNED=$SCRIPTS/../src/sentinel_sim/configs/tuned.yaml
OUT=$WS/results/demo_$(date +%Y%m%d_%H%M%S)
export SENTINEL_RUN_DIR=$HOME/.ros/sentinel_demo
mkdir -p "$OUT"

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
# shellcheck disable=SC1091
source "$HOME/sentinel_venv/bin/activate"

GROUPS_TO_KILL=()
cleanup() {
  echo; echo "[sentinel] stopping..."
  for g in "${GROUPS_TO_KILL[@]}"; do kill -INT -- -"$g" 2>/dev/null; done
  sleep 4
  for g in "${GROUPS_TO_KILL[@]}"; do kill -TERM -- -"$g" 2>/dev/null; done
  sleep 3
  for g in "${GROUPS_TO_KILL[@]}"; do kill -KILL -- -"$g" 2>/dev/null; done
  echo "[sentinel] stopped. Metrics and logs: $OUT"
  exit 0
}
trap cleanup INT TERM

# model-based perception does not use the camera: fly camera-less drones (much cheaper)
[ "$PERC" = model ] && export SENTINEL_UAV_MODEL=x500_plain
echo "[sentinel] 1/3 Gazebo + $N PX4 drones in $WORLD ..."
GUIFLAG=""; [ "$GUI" = 1 ] && GUIFLAG="--gui"
setsid "$GZS/launch_sitl.sh" "$N" "$WORLD" $GUIFLAG > "$OUT/sitl.log" 2>&1 &
GROUPS_TO_KILL+=($!)
for _ in $(seq 120); do grep -q "SITL running" "$OUT/sitl.log" && break; sleep 2; done
if ! grep -q "SITL running" "$OUT/sitl.log"; then
  echo "[sentinel] Gazebo/PX4 did not start, see $OUT/sitl.log"; cleanup
fi

FAIL='[]'; [ "$CRASH" = 1 ] && FAIL='[{"t":150,"uav":1,"kind":"crash"}]'
echo "[sentinel] 2/3 SENTINEL stack (method=$METHOD, perception=$PERC) ..."
setsid ros2 launch sentinel_ai swarm_launch.py n_uav:="$N" world:="$WORLD" method:="$METHOD" \
    perception:="$PERC" tuned_yaml:="$TUNED" failures:="$FAIL" out_dir:="$OUT" \
    run_dir:="$SENTINEL_RUN_DIR" > "$OUT/stack.log" 2>&1 &
GROUPS_TO_KILL+=($!)

if [ "$GUI" = 1 ]; then
  echo "[sentinel] 3/3 viewers ..."
  sleep 8
  if [ "$PERC" = yolo ]; then
    setsid ros2 run rqt_image_view rqt_image_view /px4_0/sentinel/thermal_annotated > "$OUT/viewer.log" 2>&1 &
  else
    setsid ros2 run rqt_image_view rqt_image_view \
      "/world/$WORLD/model/x500_thermal_0/link/thermal_link/sensor/thermal_camera/image" > "$OUT/viewer.log" 2>&1 &
  fi
  GROUPS_TO_KILL+=($!)
  setsid ros2 run rqt_graph rqt_graph > "$OUT/graph.log" 2>&1 &
  GROUPS_TO_KILL+=($!)
fi

echo "[sentinel] running. Drones take off in ~20 s, then search. Ctrl-C to stop."
echo "[sentinel] live status every 10 s (from $OUT/summary.json):"
T0=$(date +%s)
while true; do
  sleep 10
  if [ -f "$OUT/summary.json" ]; then
    python3 - "$OUT/summary.json" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get('comm', {}) or {}
print(f"  t={d.get('duration_s', 0):6.0f}s  victims found {d.get('tdr', 0):.2f}  reported {d.get('tdr_base', 0):.2f}"
      f"  coverage {d.get('coverage', 0):.2f}  connected {d.get('frac_connected', 0):.2f}"
      f"  PDR {(c.get('pdr') or 0):.2f}", flush=True)
PY
  fi
  if [ "$DUR" -gt 0 ] && [ $(( $(date +%s) - T0 )) -ge "$DUR" ]; then cleanup; fi
done
