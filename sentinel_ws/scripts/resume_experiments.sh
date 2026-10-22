#!/usr/bin/env bash
# Resume the A3 re-run: run every experiment; the runner only computes missing rows.
# (No purge, no re-tune: configs/tuned.yaml already holds the final SENTINEL values.)
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$HOME/sentinel_venv/bin/activate"
cd "$(dirname "$0")/../src/sentinel_sim"
RES=$HOME/sentinel_ws/results
for e in e11_dense_diag e9_v1_rerun e10_thold e7_ablation e2_channel e3_failures e4_dynamic e6_detection e6b_thresholds e6c_false_alarms e8_fastsim e1_scalability; do
  echo "=== $e $(date +%T)"
  python -m sentinel_sim.runner configs/experiments/$e.yaml --out $RES/$e --jobs 18 --no-series 2>&1 | grep --line-buffered -E "runs to do|ERROR|failed"
done
echo "ALL DONE $(date +%T)"
