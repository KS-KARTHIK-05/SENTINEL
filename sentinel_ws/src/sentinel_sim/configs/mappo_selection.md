# B5 (MAPPO) policy selection
Both policies were evaluated deterministically on the tuning seeds (1000-1002) in the two
tuning scenarios (urban N=8, severe N=4; 400 m; Poisson failures), like every other method.

| policy | observation | training | tuning-seed report_value |
|---|---|---|---|
| v1 | 26 features | 120 it x 16 episodes | **0.146** |
| v2 | 28 features (+frontier direction) | 250 it x 8 episodes | 0.119 |

Selected: v1 -> `mappo_policy.npz` (copy of `mappo_policy_v1_obs26.npz`).
Training curves: `mappo_policy_v1_obs26_curve.npy`, `mappo_policy_v2_obs28_curve.npy`.
For reference, tuned heuristic methods reach about 0.29 on the same seeds (configs/tuned.yaml).

## Long run (600 it x 16 episodes, obs 28), selection {'current': 0.15259931782605488, 'long': 0.2331862303641418}
Selected: long run -> mappo_policy.npz (previous kept as mappo_policy_prev_selected.npz)
