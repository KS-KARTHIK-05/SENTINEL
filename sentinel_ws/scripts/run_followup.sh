#!/usr/bin/env bash
# Follow-up runs: E9/E10 with the split loss metric, and B5 (MAPPO) for E1-E4.
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$HOME/sentinel_venv/bin/activate"
cd "$(dirname "$0")/../src/sentinel_sim"
OUT=$HOME/sentinel_ws/results
for e in e9_v1_rerun e10_thold; do
  echo "=== $e $(date +%T)"; python -m sentinel_sim.runner configs/experiments/$e.yaml --out $OUT/$e --jobs 18 --no-series 2>&1 | grep -E "runs to do|ERROR|failed" ; done
for e in e2_channel e3_failures e4_dynamic e1_scalability; do
  echo "=== $e B5 $(date +%T)"; python -m sentinel_sim.runner configs/experiments/$e.yaml --out $OUT/$e --jobs 18 --no-series --only-methods B5_mappo 2>&1 | grep -E "runs to do|ERROR|failed"; done
echo "ALL DONE $(date +%T)"
