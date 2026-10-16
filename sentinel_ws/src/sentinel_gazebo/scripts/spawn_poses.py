#!/usr/bin/env python3
"""Print spawn poses (x,y per line) for N UAVs in a generated world.

Uses the same grid as sentinel_sim.Simulation._spawn: anchor = responder
position if present, else the map centre; 4 m spacing.
"""
import json
import math
import sys

truth = json.load(open(sys.argv[1]))
n = int(sys.argv[2])
anchor = truth['base_xy'] or [truth['size'][0] / 2, truth['size'][1] / 2]
k = int(math.ceil(math.sqrt(n)))
for i in range(n):
    print(f"{anchor[0] + 4.0 * (1 + i % k):.2f},{anchor[1] + 4.0 * (1 + i // k):.2f}")
