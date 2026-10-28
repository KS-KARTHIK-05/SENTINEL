#!/usr/bin/env python3
"""Exit 0 if every PX4 run folder for the given seeds has a full-length summary.json."""
import json
import os
import sys

out, T = sys.argv[1], float(sys.argv[2])
missing = []
for s in sys.argv[3:]:
    for m in ('sentinel', 'B0_pso'):
        p = os.path.join(out, f'sentinel_g100_s{s}_{m}', 'summary.json')
        try:
            ok = json.load(open(p)).get('duration_s', 0) >= T - 5
        except (OSError, ValueError):
            ok = False
        if not ok:
            missing.append(f's{s} {m}')
print(f'{out}: {2 * (len(sys.argv) - 3) - len(missing)} complete; missing {missing}')
sys.exit(1 if missing else 0)
