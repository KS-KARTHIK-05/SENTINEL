#!/usr/bin/env bash
# Experiments for the internal-review issues 3 and 6 (2026-10-22), plus E9/E10 re-run with the
# tuned parameters. Each step is skipped-if-done by the runner; analysis after each experiment.
set -o pipefail
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/sentinel_venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/sentinel_ws/src/SENTINEL/sentinel_ws/src/sentinel_sim
for e in e12_attribution e10_thold e9_v1_rerun e11b_dense_fusion; do
  echo "=== $e $(date +%H:%M:%S)"
  python -m sentinel_sim.runner configs/experiments/$e.yaml --out ~/sentinel_ws/results/$e --jobs 18 \
    || { echo "!!! $e FAILED $(date +%H:%M:%S)"; continue; }
  python -m sentinel_sim.analyze ~/sentinel_ws/results/$e > ~/sentinel_ws/results/$e/analyze.log 2>&1 \
    || echo "!!! analyze $e FAILED"
  echo "=== $e done $(date +%H:%M:%S)"
done
echo "=== ALL DONE $(date +%H:%M:%S)"
