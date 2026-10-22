#!/usr/bin/env bash
# Long MAPPO (B5) training, then selection on tuning seeds (same rule as all methods);
# if the new policy wins, replace configs/mappo_policy.npz and re-run only B5 rows of E1-E4.
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$HOME/sentinel_venv/bin/activate"
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT/src/sentinel_sim"
RES=$HOME/sentinel_ws/results
echo "=== train $(date +%T)"
python -u -m sentinel_sim.marl --iters 600 --episodes 16 --jobs 14 --seed0 20000 \
    --out configs/mappo_policy_long.npz 2>&1 | grep --line-buffered -E "^iter|Error|Traceback"
echo "=== select $(date +%T)"
python - <<'PY'
import copy, os, shutil, numpy as np
from concurrent.futures import ProcessPoolExecutor
from sentinel_sim.tune import TUNE_SCENARIOS, TUNE_BASE, TUNE_SEEDS
def ev(a):
    path, sc, seed = a
    from sentinel_sim.sim import run
    c = copy.deepcopy(TUNE_BASE); c.update(copy.deepcopy(sc)); c.update(method='B5_mappo', seed=seed, policy_path=path)
    return run(c)['summary']['report_value']
cands = {'current': os.path.abspath('configs/mappo_policy.npz'), 'long': os.path.abspath('configs/mappo_policy_long.npz')}
jobs = [(n, (p, sc, s)) for n, p in cands.items() for sc in TUNE_SCENARIOS for s in TUNE_SEEDS]
with ProcessPoolExecutor(12) as ex: res = list(ex.map(ev, [j[1] for j in jobs]))
score = {n: float(np.mean([r for (m, _), r in zip(jobs, res) if m == n])) for n in cands}
print('tuning-seed report_value', score)
with open('configs/mappo_selection.md', 'a') as f:
    f.write(f'\n## Long run (600 it x 16 episodes, obs 28), selection {score}\n')
    if score['long'] > score['current']:
        shutil.copy('configs/mappo_policy.npz', 'configs/mappo_policy_prev_selected.npz')
        shutil.copy('configs/mappo_policy_long.npz', 'configs/mappo_policy.npz')
        f.write('Selected: long run -> mappo_policy.npz (previous kept as mappo_policy_prev_selected.npz)\n')
        open('/tmp/mappo_long_selected', 'w').write('1')
    else:
        f.write('Selected: previous policy kept (long run did not improve on tuning seeds)\n')
PY
if [ -f /tmp/mappo_long_selected ]; then
  rm -f /tmp/mappo_long_selected
  echo "=== re-run B5 rows $(date +%T)"
  python - <<'PY'
import glob, os, pandas as pd
for e in ['e1_scalability', 'e2_channel', 'e3_failures', 'e4_dynamic']:
    f = os.path.expanduser(f'~/sentinel_ws/results/{e}/summary.parquet')
    df = pd.read_parquet(f); df.to_parquet(f.replace('summary.parquet', 'summary_preB5long.parquet'), index=False)
    df[df.cfg_method != 'B5_mappo'].to_parquet(f, index=False)
PY
  for e in e2_channel e3_failures e4_dynamic e1_scalability; do
    python -m sentinel_sim.runner configs/experiments/$e.yaml --out $RES/$e --jobs 18 --no-series --only-methods B5_mappo 2>&1 | grep -E "runs to do|ERROR"
  done
  OMP_NUM_THREADS=2 python "$ROOT/scripts/make_results.py" 2>&1 | tail -1
fi
echo "MAPPO LONG DONE $(date +%T)"
