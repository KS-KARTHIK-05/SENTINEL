#!/usr/bin/env bash
# Run the evaluation suite (REWORK_PLAN §6, Amendment A1). Resumable: the runner skips
# completed runs. Usage: run_experiments.sh JOBS [extra runner args, e.g. --skip-methods B5_mappo]
set -eo pipefail
JOBS=${1:-18}; shift || true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$HOME/sentinel_venv/bin/activate"
SIM=$(cd "$(dirname "$0")/../src/sentinel_sim" && pwd)
OUT=${RESULTS_DIR:-$HOME/sentinel_ws/results}
cd "$SIM"
for e in e9_v1_rerun e10_thold e7_ablation e2_channel e3_failures e4_dynamic e6_detection e6b_thresholds e1_scalability; do
  echo "=== $e $(date +%T)"
  python -m sentinel_sim.runner "configs/experiments/$e.yaml" --out "$OUT/$e" --jobs "$JOBS" --no-series "$@" 2>&1 | grep -E "runs to do|done|ERROR|failed" | tail -3
done
echo "ALL DONE $(date +%T)"
