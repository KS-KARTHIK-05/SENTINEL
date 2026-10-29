#!/usr/bin/env bash
# Re-run C1 (E5 robustness) and C2 (E6 fast-simulator side) with the corrected false-alarm
# conversion (results/overnight_2026-10-28/PLAN.md §9, 29 Oct 13:50). Same job protocol as
# resume_queue.sh (two attempts, done markers, completeness check); new results folder.
#
#   setsid nohup bash run_e5_recheck.sh > .../e5_recheck_2026-10-29/logs/queue.out 2>&1 < /dev/null &
set -o pipefail
RES=$HOME/sentinel_ws/results/e5_recheck_2026-10-29
ROOT=$HOME/sentinel_ws/src/SENTINEL/sentinel_ws
SIM=$ROOT/src/sentinel_sim
OV=$ROOT/scripts/overnight
CFG=configs/experiments/overnight
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
sim() {
  "$PY" -m sentinel_sim.runner "$CFG/$1.yaml" --out "$RES/$1" --jobs "$JOBS" --no-series \
    && "$PY" "$OV/check_complete.py" "$CFG/$1.yaml" "$RES/$1"
}

log "E5 RECHECK START (pid $$, $JOBS workers)"
run_job c2_e6_fastsim_v2 sim e6_fastsim_v2
run_job c1_e5_robust_v2 sim robust_fitted_v2
log "E5 RECHECK END"
