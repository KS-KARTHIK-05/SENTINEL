#!/usr/bin/env bash
# Resume the rest of the overnight queue (plan: results/overnight_2026-10-28/PLAN.md, deviations §9)
# with the same job names and done markers as run_overnight.sh.
#
#   setsid nohup bash resume_queue.sh > .../logs/resume_queue.out 2>&1 < /dev/null &
#
# Lane B: E4 rendering (worlds whose done flag is missing; every frame is checked against its
#   pose while capturing and audited afterwards), dataset, evaluation, fine-tuning, E5 fit, configs.
# PX4 lane: D1 (E8r) now; D2 (E6) once lane B has finished (fine-tuned model).
# Simulator lane: C1, C2 once lane B has written the fitted configs; nice 19, so the PX4 runs keep
#   CPU priority (fast-simulator results do not depend on load or worker count).
# Stopped at 00:47 (D1 launch argument, lane B dataloader deadlock) and at 01:09 on 29 Oct (stale
# rendered frames in worlds 107, 110-115; see PLAN.md §9).
set -o pipefail
RES=$HOME/sentinel_ws/results/overnight_2026-10-28
ROOT=$HOME/sentinel_ws/src/SENTINEL/sentinel_ws
SIM=$ROOT/src/sentinel_sim
OV=$ROOT/scripts/overnight
CFG=configs/experiments/overnight
PY=$HOME/sentinel_venv/bin/python
HIT=$ROOT/src/sentinel_ai/models/yolo26n_thermal_best.pt
FT=$RES/e4/finetune/yolo26n_rendered/weights/best.pt
SPLITS="--train $(seq -s ' ' 100 113) --val $(seq -s ' ' 114 118) --test $(seq -s ' ' 119 123)"
LOG=$RES/progress.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SIM" || exit 1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
run_job() {
  local name=$1; shift
  if [ -f "$RES/done/$name" ]; then log "skip $name (already done)"; return 0; fi
  local att rc t0
  for att in 1 2; do
    t0=$(date +%s)
    echo "=== $(date '+%F %T') start $name (attempt $att, resume_queue)" >> "$RES/logs/$name.log"
    "$@" >> "$RES/logs/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch "$RES/done/$name"; log "DONE $name ($(( ($(date +%s) - t0) / 60 )) min, attempt $att, resume_queue)"; return 0
    fi
    log "FAILED $name attempt $att (exit $rc, resume_queue, see logs/$name.log)"
  done
  return 1
}
sim() {
  "$PY" -m sentinel_sim.runner "$CFG/$1.yaml" --out "$RES/$1" --jobs "$2" --no-series \
    && "$PY" "$OV/check_complete.py" "$CFG/$1.yaml" "$RES/$1"
}
e4_render() {
  "$PY" "$OV/e4_render.py" --seeds $(seq 100 123) --out "$RES/e4" \
    && "$PY" "$OV/e4_posecheck.py" --root "$RES/e4" --seeds $(seq 100 123)
}
e4_eval() {
  "$PY" "$OV/e4_detector.py" eval --root "$RES/e4" $SPLITS --split test --model "$1" --tag "$2" \
    && "$PY" "$OV/e4_detector.py" eval --root "$RES/e4" $SPLITS --split val --model "$1" --tag "$2"
}

lane_b() {
  run_job b1_e4_render e4_render || return 1
  run_job b2_e4_build "$PY" "$OV/e4_detector.py" build --root "$RES/e4" $SPLITS || return 1
  run_job b3_e4_eval_hituav e4_eval "$HIT" hituav || return 1
  run_job b4_e4_finetune "$PY" "$OV/e4_detector.py" finetune --root "$RES/e4" $SPLITS --model "$HIT" || return 1
  run_job b5_e4_eval_finetuned e4_eval "$FT" finetuned || return 1
  run_job b6_e5_fit "$PY" "$OV/e5_fit.py" --root "$RES/e4" --tag finetuned --out "$RES/e5" || return 1
  run_job b7_e5_configs "$PY" "$OV/make_fitted_configs.py" "$RES/e5/detector_fitted.json" "$CFG"
}

sitl_lane() {
  source /opt/ros/humble/setup.bash 2>/dev/null
  run_job d1_e8r_sitl bash -c "'$OV/run_sitl.sh' '$RES/e8r_sitl' model '' 0 1 2 && '$PY' '$OV/check_sitl.py' '$RES/e8r_sitl' 300 0 1 2"
  while kill -0 "$PB" 2>/dev/null; do sleep 60; done
  if [ -f "$RES/done/b4_e4_finetune" ] && [ -f "$FT" ]; then E6M=$FT; else E6M=$HIT; log "NOTE e6 uses the HIT-UAV-only model (no fine-tuned model)"; fi
  run_job d2_e6_sitl bash -c "'$OV/run_sitl.sh' '$RES/e6_sitl' yolo '$E6M' $(seq -s ' ' 0 9) && '$PY' '$OV/check_sitl.py' '$RES/e6_sitl' 300 $(seq -s ' ' 0 9)"
}

sim_lane() {
  renice -n 19 -p $BASHPID > /dev/null
  while kill -0 "$PB" 2>/dev/null; do sleep 60; done
  if [ -f "$RES/done/b7_e5_configs" ] && [ -f "$RES/done/a1_tune_b7b8" ]; then
    run_job c1_e5_robust sim robust_fitted 18
    run_job c2_e6_fastsim sim e6_fastsim 18
  else
    log "SKIPPED c1_e5_robust, c2_e6_fastsim (needs a1_tune_b7b8 and b7_e5_configs)"
  fi
}

log "RESUME QUEUE START (pid $$)"
lane_b > "$RES/logs/lane_b3.out" 2>&1 &
PB=$!
sitl_lane > "$RES/logs/sitl_lane.out" 2>&1 &
PS=$!
sim_lane > "$RES/logs/sim_lane.out" 2>&1 &
PC=$!
wait $PB $PS $PC
log "QUEUE END"
