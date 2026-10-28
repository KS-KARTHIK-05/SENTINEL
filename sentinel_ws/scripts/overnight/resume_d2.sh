#!/usr/bin/env bash
# Re-run D2 (E6: 10 PX4 worlds, fine-tuned YOLO26n in the loop) on its own after it was stopped at
# 02:23 on 29 Oct: the ROS nodes read each world's truth file from the colcon install, which had only
# worlds 0-2, so on world 3 they exited at start-up (run_sitl.sh now passes world_dir, the folder Gazebo
# loads the worlds from, and stops a run when a node dies). Same job name and done marker; finished
# runs (worlds 0-2) are kept.
set -o pipefail
RES=$HOME/sentinel_ws/results/overnight_2026-10-28
ROOT=$HOME/sentinel_ws/src/SENTINEL/sentinel_ws
OV=$ROOT/scripts/overnight
PY=$HOME/sentinel_venv/bin/python
FT=$RES/e4/finetune/yolo26n_rendered/weights/best.pt
LOG=$RES/progress.log
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
run_job() {
  local name=$1; shift
  if [ -f "$RES/done/$name" ]; then log "skip $name (already done)"; return 0; fi
  local att rc t0
  for att in 1 2; do
    t0=$(date +%s)
    echo "=== $(date '+%F %T') start $name (attempt $att, resume_d2)" >> "$RES/logs/$name.log"
    "$@" >> "$RES/logs/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch "$RES/done/$name"; log "DONE $name ($(( ($(date +%s) - t0) / 60 )) min, attempt $att, resume_d2)"; return 0
    fi
    log "FAILED $name attempt $att (exit $rc, resume_d2, see logs/$name.log)"
  done
  return 1
}
source /opt/ros/humble/setup.bash 2>/dev/null
[ -f "$FT" ] || { log "FAILED d2_e6_sitl: no fine-tuned model"; exit 1; }
log "D2 RESUME (pid $$)"
run_job d2_e6_sitl bash -c "'$OV/run_sitl.sh' '$RES/e6_sitl' yolo '$FT' $(seq -s ' ' 0 9) && '$PY' '$OV/check_sitl.py' '$RES/e6_sitl' 300 $(seq -s ' ' 0 9)"
log "D2 END"
