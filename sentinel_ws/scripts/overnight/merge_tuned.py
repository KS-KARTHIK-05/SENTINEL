#!/usr/bin/env python3
"""tuned_v3.yaml = configs/tuned.yaml (B0-B6, SENTINEL unchanged) + the B7/B8 tuning result."""
import sys

import yaml

base, new, out = sys.argv[1:4]
t = yaml.safe_load(open(base))
n = yaml.safe_load(open(new))
for m in ('B7_wall', 'B8_relay'):
    assert m in n, f'{m} not tuned'
    assert m not in t
    t[m] = n[m]
yaml.safe_dump(t, open(out, 'w'), sort_keys=True)
chk = yaml.safe_load(open(out))
assert all(chk[m] == v for m, v in yaml.safe_load(open(base)).items()), 'existing values changed'
print('wrote', out, {m: chk[m] for m in ('B7_wall', 'B8_relay')})
