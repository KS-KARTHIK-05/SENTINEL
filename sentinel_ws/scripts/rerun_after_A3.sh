#!/usr/bin/env bash
# Amendment A3 protocol: purge SENTINEL-family rows (old fusion), re-tune SENTINEL with the
# equal 30-trial budget, re-run every experiment (runner only computes missing rows).
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$HOME/sentinel_venv/bin/activate"
SIM=$(cd "$(dirname "$0")/../src/sentinel_sim" && pwd); cd "$SIM"
RES=$HOME/sentinel_ws/results
echo "=== purge $(date +%T)"
python - <<'PY'
import glob, os, pandas as pd
for f in glob.glob(os.path.expanduser('~/sentinel_ws/results/e*/summary.parquet')):
    if 'oldmetric' in f: continue
    bak = f.replace('summary.parquet', 'summary_preA3.parquet')
    df = pd.read_parquet(f)
    if not os.path.exists(bak): df.to_parquet(bak, index=False)
    keep = ~(df.cfg_method.str.startswith('sentinel') | df.cfg_method.str.startswith('abl_'))
    df[keep].to_parquet(f, index=False)
    print(f'{f.split("/")[-2]:18s} kept {keep.sum():5d} removed {(~keep).sum():5d}')
PY
echo "=== retune sentinel $(date +%T)"
python - <<'PY'
import yaml
t = yaml.safe_load(open('configs/tuned.yaml'))
if 'sentinel' in t:
    yaml.safe_dump({'sentinel': t['sentinel']}, open('configs/tuned_sentinel_preA3.yaml', 'w'))
    t.pop('sentinel'); yaml.safe_dump(t, open('configs/tuned.yaml', 'w'), sort_keys=True)
PY
python -u -m sentinel_sim.tune --methods sentinel --trials 30 --out configs/tuned.yaml 2>&1 | grep -v Warning
for e in e11_dense_diag e9_v1_rerun e10_thold e7_ablation e2_channel e3_failures e4_dynamic e6_detection e6b_thresholds e6c_false_alarms e8_fastsim e1_scalability; do
  echo "=== $e $(date +%T)"
  python -m sentinel_sim.runner configs/experiments/$e.yaml --out $RES/$e --jobs 18 --no-series 2>&1 | grep -E "runs to do|ERROR|failed"
done
echo "ALL DONE $(date +%T)"
