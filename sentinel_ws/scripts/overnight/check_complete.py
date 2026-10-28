#!/usr/bin/env python3
"""Exit 0 if every run of an experiment spec is in its summary.parquet (queue completion check)."""
import sys

import pandas as pd
import yaml

from sentinel_sim.runner import expand, run_id

spec_path, out = sys.argv[1], sys.argv[2]
with open(spec_path) as f:
    spec = yaml.safe_load(f)
want = {run_id(c) for c in expand(spec)}
try:
    have = set(pd.read_parquet(f'{out}/summary.parquet')['run_id'])
except FileNotFoundError:
    have = set()
missing = want - have
print(f'{spec_path}: {len(want) - len(missing)}/{len(want)} runs complete')
sys.exit(1 if missing else 0)
