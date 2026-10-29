#!/usr/bin/env bash
# Baseline re-run (PLAN.md section 9, 29 Oct): re-tune B0-B4 and B6 on the current code (same
# Optuna budget, objective and tuning seeds), then re-run every row of B0-B4, B6 and their fusion
# variants in every experiment that contains them. Same job protocol as resume_queue.sh (two
# attempts, done markers, completeness check, resumable); waits for the C1 queue to finish.
#
#   setsid nohup bash run_baseline_rerun.sh <C1 queue pid> > .../baseline_rerun_2026-10-29/logs/queue.out 2>&1 < /dev/null &
set -o pipefail
WAIT_PID=${1:-}
RES=$HOME/sentinel_ws/results/baseline_rerun_2026-10-29
ROOT=$HOME/sentinel_ws/src/SENTINEL/sentinel_ws
SIM=$ROOT/src/sentinel_sim
OV=$ROOT/scripts/overnight
CFG=configs/experiments/rerun_2026-10-29
PY=$HOME/sentinel_venv/bin/python
LOG=$RES/progress.log
JOBS=${JOBS:-19}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$RES/logs" "$RES/done"
cd "$SIM" || exit 1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
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
      touch "$RES/done/$name"; log "DONE $name ($(( ($(date +%s) - t0) / 60 )) min, attempt $att)"; return 0
    fi
    log "FAILED $name attempt $att (exit $rc, see logs/$name.log)"
  done
  return 1
}
retune() {
  # one method at a time, as for B7/B8 (run_overnight.sh): 30 trials, 16 workers
  local m
  for m in B0_pso B1_v1 B2_commpso B3_lambda2 B4_potential B6_lawnmower; do
    "$PY" -m sentinel_sim.tune --methods "$m" --trials 30 --jobs 16 --out configs/tuned_baselines_v4.yaml || return 1
  done
}
sim() {
  "$PY" -m sentinel_sim.runner "$CFG/$1.yaml" --out "$RES/$1" --jobs "$JOBS" --no-series \
    && "$PY" "$OV/check_complete.py" "$CFG/$1.yaml" "$RES/$1"
}

log "BASELINE RERUN QUEUE START (pid $$, $JOBS workers)"
if [ -n "$WAIT_PID" ]; then
  log "waiting for pid $WAIT_PID (C1 queue)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "pid $WAIT_PID finished"
fi
run_job r0_retune retune || { log "STOP: re-tune failed"; exit 1; }
run_job r0_merge "$PY" "$OV/merge_tuned_v4.py" configs/tuned_v3.yaml configs/tuned_baselines_v4.yaml configs/tuned_v4.yaml \
  || { log "STOP: merge failed"; exit 1; }
run_job r0_configs "$PY" "$OV/make_rerun_configs.py" --tuned configs/tuned_v4.yaml --out "$CFG" \
  || { log "STOP: configs failed"; exit 1; }
# main tables first (Tables 2-3, channel sentence), then Table 4, robustness, extended results
for e in e3_failures e2_channel e1_scalability e12_attribution e11b_dense_fusion robust_fitted_v2 \
         e6_fastsim_v2 e8_fastsim e9_v1_rerun e4_dynamic e6_detection e6c_false_alarms e11_dense_diag \
         e7_ablation; do
  run_job "r_$e" sim "$e"
done
log "BASELINE RERUN QUEUE END"
