#!/usr/bin/env python3
"""tuned_v4.yaml = tuned_v3.yaml with B0-B4 and B6 replaced by their re-tune on the current code
(29 Oct, PLAN.md section 9). SENTINEL, B7 and B8 keep their values (checked)."""
import sys

import yaml

RETUNED = ('B0_pso', 'B1_v1', 'B2_commpso', 'B3_lambda2', 'B4_potential', 'B6_lawnmower')

base, new, out = sys.argv[1:4]
t = yaml.safe_load(open(base))
n = yaml.safe_load(open(new))
for m in RETUNED:
    assert m in n, f'{m} not re-tuned'
    t[m] = n[m]
yaml.safe_dump(t, open(out, 'w'), sort_keys=True)
chk = yaml.safe_load(open(out))
old = yaml.safe_load(open(base))
assert all(chk[m] == v for m, v in old.items() if m not in RETUNED), 'other values changed'
assert set(chk) == set(old)
for m in RETUNED:
    print(f'{m}: {old[m]} -> {chk[m]}')
print('wrote', out)
