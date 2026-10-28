#!/usr/bin/env bash
# Overnight queue 2026-10-28 (plan: ~/sentinel_ws/results/overnight_2026-10-28/PLAN.md).
#
#   setsid nohup bash run_overnight.sh > ~/sentinel_ws/results/overnight_2026-10-28/queue.log 2>&1 < /dev/null &
#
# Resumable: a finished job leaves results/overnight_2026-10-28/done/<job>; the runner skips
# finished missions, rendering skips finished worlds and PX4 runs skip finished folders, so after
# a restart the same command continues where it stopped. A failed job is retried once; if it
# fails again it is logged as FAILED and the queue moves on (dependants are skipped).
# progress.log: one line per finished (or failed / skipped) job with a timestamp.
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
mkdir -p "$RES/done" "$RES/logs"
cd "$SIM" || exit 1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

# run_job NAME CMD... : skip if done, else up to two attempts; output to logs/NAME.log
run_job() {
  local name=$1; shift
  if [ -f "$RES/done/$name" ]; then log "skip $name (already done)"; return 0; fi
  local att rc t0
  for att in 1 2; do
    t0=$(date +%s)
    echo "=== $(date '+%F %T') start $name (attempt $att)" >> "$RES/logs/$name.log"
    "$@" >> "$RES/logs/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch "$RES/done/$name"
      log "DONE $name ($(( ($(date +%s) - t0) / 60 )) min, attempt $att)"
      return 0
    fi
    log "FAILED $name attempt $att (exit $rc, see logs/$name.log)"
  done
  return 1
}

sim() {   # sim EXP JOBS : run one experiment spec to completion, then verify it is complete
  "$PY" -m sentinel_sim.runner "$CFG/$1.yaml" --out "$RES/$1" --jobs "$2" --no-series \
    && "$PY" "$OV/check_complete.py" "$CFG/$1.yaml" "$RES/$1"
}

tune_b7b8() {
  local m
  for m in B7_wall B8_relay; do
    if [ -f configs/tuned_b7b8.yaml ] && grep -q "^$m:" configs/tuned_b7b8.yaml; then continue; fi
    "$PY" -m sentinel_sim.tune --methods "$m" --trials 30 --jobs 16 --out configs/tuned_b7b8.yaml || return 1
  done
  "$PY" "$OV/merge_tuned.py" configs/tuned.yaml configs/tuned_b7b8.yaml configs/tuned_v3.yaml
}

lane_a() {   # fast simulator, 16 workers
  if run_job a1_tune_b7b8 tune_b7b8; then
    run_job a2_b7b8_failures sim b7b8_failures 16
    run_job a3_b7b8_scalability sim b7b8_scalability 16
    run_job a4_b7b8_channel sim b7b8_channel 16
    run_job a5_b7b8_attribution sim b7b8_attribution 16
    run_job a6_b7b8_dense_fusion sim b7b8_dense_fusion 16
  else
    log "SKIPPED a2-a6 (B7/B8 tuning failed)"
  fi
  run_job a7_sensitivity sim sensitivity 16
}

e4_eval() {   # e4_eval MODEL TAG : evaluate on val and test
  "$PY" "$OV/e4_detector.py" eval --root "$RES/e4" $SPLITS --split test --model "$1" --tag "$2" \
    && "$PY" "$OV/e4_detector.py" eval --root "$RES/e4" $SPLITS --split val --model "$1" --tag "$2"
}

lane_b() {   # GPU: E4 rendering, evaluation, fine-tuning; E5 fit
  run_job b1_e4_render "$PY" "$OV/e4_render.py" --seeds $(seq 100 123) --out "$RES/e4" || return 1
  run_job b2_e4_build "$PY" "$OV/e4_detector.py" build --root "$RES/e4" $SPLITS || return 1
  run_job b3_e4_eval_hituav e4_eval "$HIT" hituav || return 1
  run_job b4_e4_finetune "$PY" "$OV/e4_detector.py" finetune --root "$RES/e4" $SPLITS --model "$HIT" || return 1
  run_job b5_e4_eval_finetuned e4_eval "$FT" finetuned || return 1
  run_job b6_e5_fit "$PY" "$OV/e5_fit.py" --root "$RES/e4" --tag finetuned --out "$RES/e5" || return 1
  run_job b7_e5_configs "$PY" "$OV/make_fitted_configs.py" "$RES/e5/detector_fitted.json" "$CFG"
}

log "QUEUE START (pid $$)"
lane_a > "$RES/logs/lane_a.out" 2>&1 &
PA=$!
lane_b > "$RES/logs/lane_b.out" 2>&1 &
PB=$!
wait $PA $PB
log "lanes A and B finished"

if [ -f "$RES/done/b7_e5_configs" ] && [ -f "$RES/done/a1_tune_b7b8" ]; then
  run_job c1_e5_robust sim robust_fitted 18
else
  log "SKIPPED c1_e5_robust (needs a1_tune_b7b8 and b7_e5_configs)"
fi
[ -f "$RES/done/b7_e5_configs" ] && run_job c2_e6_fastsim sim e6_fastsim 18

source /opt/ros/humble/setup.bash 2>/dev/null
run_job d1_e8r_sitl bash -c "'$OV/run_sitl.sh' '$RES/e8r_sitl' model '' 0 1 2 && '$PY' '$OV/check_sitl.py' '$RES/e8r_sitl' 300 0 1 2"
if [ -f "$FT" ]; then E6M=$FT; else E6M=$HIT; log "NOTE e6 uses the HIT-UAV-only model (no fine-tuned model)"; fi
run_job d2_e6_sitl bash -c "'$OV/run_sitl.sh' '$RES/e6_sitl' yolo '$E6M' $(seq -s ' ' 0 9) && '$PY' '$OV/check_sitl.py' '$RES/e6_sitl' 300 $(seq -s ' ' 0 9)"
log "QUEUE END"
