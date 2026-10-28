#!/usr/bin/env bash
# Re-run lane B of run_overnight.sh (E4 rendering, evaluation, fine-tuning, E5 fit) on its own, with the
# same job names and done markers, after its render job failed twice in the main queue (transient
# gz-transport set_pose timeouts; renderer now retries per world and falls back to the gz CLI).
# The main queue picks up done/b7_e5_configs when lane A finishes.
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
    echo "=== $(date '+%F %T') start $name (attempt $att, lane_b_resume)" >> "$RES/logs/$name.log"
    "$@" >> "$RES/logs/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch "$RES/done/$name"; log "DONE $name ($(( ($(date +%s) - t0) / 60 )) min, attempt $att, lane_b_resume)"; return 0
    fi
    log "FAILED $name attempt $att (exit $rc, lane_b_resume, see logs/$name.log)"
  done
  return 1
}
e4_eval() {
  "$PY" "$OV/e4_detector.py" eval --root "$RES/e4" $SPLITS --split test --model "$1" --tag "$2" \
    && "$PY" "$OV/e4_detector.py" eval --root "$RES/e4" $SPLITS --split val --model "$1" --tag "$2"
}
log "LANE B RESUME (pid $$)"
run_job b1_e4_render "$PY" "$OV/e4_render.py" --seeds $(seq 100 123) --out "$RES/e4" || exit 1
run_job b2_e4_build "$PY" "$OV/e4_detector.py" build --root "$RES/e4" $SPLITS || exit 1
run_job b3_e4_eval_hituav e4_eval "$HIT" hituav || exit 1
run_job b4_e4_finetune "$PY" "$OV/e4_detector.py" finetune --root "$RES/e4" $SPLITS --model "$HIT" || exit 1
run_job b5_e4_eval_finetuned e4_eval "$FT" finetuned || exit 1
run_job b6_e5_fit "$PY" "$OV/e5_fit.py" --root "$RES/e4" --tag finetuned --out "$RES/e5" || exit 1
run_job b7_e5_configs "$PY" "$OV/make_fitted_configs.py" "$RES/e5/detector_fitted.json" "$CFG"
log "LANE B RESUME END"
