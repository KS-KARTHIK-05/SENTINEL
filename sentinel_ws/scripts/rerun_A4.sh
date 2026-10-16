#!/usr/bin/env bash
# Amendment A4: re-tune SENTINEL on the design's own parameter space (no explore_lam),
# re-run SENTINEL-family rows of every experiment, re-run the E8 SITL cross-check,
# regenerate results. Detached-safe and resumable.
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$HOME/sentinel_venv/bin/activate"
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT/src/sentinel_sim"
RES=$HOME/sentinel_ws/results
if [ ! -f configs/tuned_sentinel_A3_explore.yaml ]; then
  echo "=== purge $(date +%T)"
  python - <<'PY'
import glob, os, yaml, pandas as pd
t = yaml.safe_load(open('configs/tuned.yaml'))
yaml.safe_dump({'sentinel': t.pop('sentinel')}, open('configs/tuned_sentinel_A3_explore.yaml', 'w'))
yaml.safe_dump(t, open('configs/tuned.yaml', 'w'), sort_keys=True)
for f in glob.glob(os.path.expanduser('~/sentinel_ws/results/e*/summary.parquet')):
    if 'oldmetric' in f: continue
    df = pd.read_parquet(f)
    df.to_parquet(f.replace('summary.parquet', 'summary_A3explore.parquet'), index=False)
    keep = ~(df.cfg_method.str.startswith('sentinel') | df.cfg_method.str.startswith('abl_'))
    df[keep].to_parquet(f, index=False)
    print(f'{f.split("/")[-2]:18s} removed {(~keep).sum():5d}')
PY
fi
if ! grep -q "^sentinel:" configs/tuned.yaml; then
  echo "=== retune sentinel $(date +%T)"
  python -u -m sentinel_sim.tune --methods sentinel --trials 30 --out configs/tuned.yaml 2>&1 | grep -v Warning
fi
for e in e11_dense_diag e9_v1_rerun e10_thold e7_ablation e2_channel e3_failures e4_dynamic e6_detection e6b_thresholds e6c_false_alarms e8_fastsim e1_scalability; do
  echo "=== $e $(date +%T)"
  python -m sentinel_sim.runner configs/experiments/$e.yaml --out $RES/$e --jobs 18 --no-series 2>&1 | grep --line-buffered -E "runs to do|ERROR|failed"
done
echo "=== E8 SITL $(date +%T)"
[ -d $RES/e8_sitl ] && [ ! -d $RES/e8_sitl_A3explore ] && mv $RES/e8_sitl $RES/e8_sitl_A3explore
"$ROOT/scripts/run_e8_sitl.sh" 300 3
echo "=== make_results $(date +%T)"
OMP_NUM_THREADS=2 python "$ROOT/scripts/make_results.py" 2>&1 | tail -1
echo "ALL DONE $(date +%T)"
