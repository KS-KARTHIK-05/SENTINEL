# Overnight experiments 2026-10-28: plan and analysis plan (fixed before any result)

> **File history.** First committed as `docs/overnight_2026-10-28_PLAN.md`. Sections 0–8 are the plan as fixed before any evaluation run, and section 9 records every later deviation with its date and reason. Claims quoted in the plan (for example "23–29 %") are those of the draft at the time; the current results are in the extended results. Edits for the public repository (30 Oct 2026): this note, and two wording changes ("the priority order of the task" → "the priority order fixed beforehand" in section 1; "Decision (user" → "Decision (authors" in section 9).

**Written:** 20:40, 28 Oct 2026, before any evaluation run.
- **Pilots used other seeds only:** fast simulator 900–901 with untuned B7/B8; render worlds 990–992; PX4 world 0 for 60 s. None of these is evaluation data.
- **Nothing below changes after results are seen.** Any deviation is added as a dated note at the end, with its reason.

Results go to `~/sentinel_ws/results/overnight_2026-10-28/` (one folder per job). Code is in:
- `sentinel_ws/scripts/overnight/`;
- `sentinel_core/baselines/{hard_wall,relay_chain}.py`;
- `sentinel_sim/methods.py`, `tune.py`;
- `configs/experiments/overnight/`.

## 0. Common protocol (unchanged from the paper)

- **Seeds:** evaluation seeds 0–29, with common random numbers; tuning seeds 1000–1002.
- **Primary metric:** `rep_within_60s`, the share of all victims reported to the responder within 60 s of first confirmation.
- **Statistics:**
  - mean ± half-width of the 95 % bootstrap CI (`sentinel_core.metrics.bootstrap_ci`, 5000 resamples, seed 0);
  - paired two-sided Wilcoxon signed-rank tests against SENTINEL, Holm-corrected (`paired_wilcoxon_holm`);
  - α = 0.05.
- **Holm family:** in Tables 2 and 3, one family per table column (one setting), covering every baseline in the table. With B7 and B8 that is 9 tests per column, and the stars of B0–B6 are recomputed in the new family. Table 4 keeps the existing rule: one family per column, over all rows compared with SENTINEL.
- **SENTINEL and B0–B6 rows** are taken from the existing result files of the same configurations. Runs are deterministic (re-runs of stored runs were checked identical on 28 Oct: sentinel, B0/B3/B4/B5/B6 + fusion, abl_no_projection). New jobs therefore run only the new methods, except E3 and E5, which re-run SENTINEL as their own reference.

## 1. Jobs, order and time estimate

Pilot wall times per mission with 18 parallel workers:

| Mission | Wall time |
|---|---|
| B7/B8, 8 UAVs | 27–39 s |
| B7/B8, 50 UAVs, 400 m | 360–570 s |
| Sensitivity variants, 8 UAVs | 39–45 s |
| Rendering | 0.4 s per frame |
| Fine-tuning | about 5 s per epoch on 82 frames |
| PX4 run | 5.3 min (300 s simulated) |

**Lane A (fast simulator, 16 workers):**

| # | Job | Runs | Estimate |
|---|---|---|---|
| A1 | Tune B7, B8: Optuna TPE, 30 trials × 6 missions each, same space rule, objective J and tuning seeds as B0–B6 (the full budget fits) | 360 | 15 min |
| A2 | B7, B8 in the Table 2 settings (E3 grid: urban rubble and severe × six failure settings, 400 m, N = 8) | 720 | 25 min |
| A3 | B7, B8 in the Table 3 settings (E1 grid: N ∈ {3, 5, 10, 20, 50} × 400 m and 200 m) | 600 | 75 min |
| A4 | B7, B8 in the four channel profiles (E2 grid) | 240 | 10 min |
| A5 | B7 + fusion, B8 + fusion in the E12 cells (urban, severe × N ∈ {4, 8}) | 240 | 5 min |
| A6 | B7 + fusion, B8 + fusion in the dense swarm (E11b: 200 m, N = 50) | 60 | 55 min |
| A7 | E3 sensitivity (§4); SENTINEL variants only, so it uses the original `configs/tuned.yaml` and does not depend on A1 | 600 | 25 min |

**Lane B (GPU, in parallel with lane A):** E4 rendering → dataset → evaluation of the HIT-UAV model → fine-tuning → evaluation of the fine-tuned model → E5 fit, about 60 min.

**Afterwards (sequential):**

| # | Job | Runs | Estimate |
|---|---|---|---|
| C1 | E5 robustness | 1800 | 60 min |
| C2 | Fast-simulator side of E6 and E8r | 40 | 5 min |
| D1 | E8r: re-run of the original 3-world PX4 check on the current code (§7) | 6 PX4 runs | 35 min |
| D2 | E6: 10 worlds with YOLO26n in the loop | 20 PX4 runs | 1 h 50 min |

**Total:** about 6.5 h. The whole list fits in the 10 h window, so **nothing is cut**.
- **If a job fails twice:** it is reported as not completed.
- **If time runs short:** the priority order fixed beforehand applies (B7/B8 tuning, Table 2, Table 3, channels, E3, E4, E5, E6), and E8r goes with E6.

## 2. E1: B7 hard distance wall

- **Rule:**
  - B0 constriction PSO, in which a candidate waypoint is infeasible if it is farther than R = 50 m from the UAV's nearest fresh neighbour (last beacon positions, responder included);
  - infeasible candidates get fitness −∞;
  - if the best candidate is infeasible, the UAV holds position;
  - the point toward the nearest neighbour is added to the particle seeds;
  - with no fresh neighbour the wall is vacuous.
- **Tuned:** `agent.w_goal` and `agent.sector_bias`, as for B0. R is fixed.

## 3. E2: B8 relay chain

- **Relays:**
  - m = round(ρN) relays (at least 1 when N ≥ 3, at most N − 1), taken as the highest UAV ids;
  - they sit on the ray from the responder to the centroid of the alive searchers, evenly spaced at s · R₀.₀₅(1) of the active profile (D/m if the centroid is closer);
  - the chain is re-planned every 1 s planning period and re-formed from the alive relays.
- **Searchers:** B0 PSO, with sectors over the searchers only.
- **Idealisation:** the chain uses the true positions of the alive UAVs, which favours the baseline, as B3's exact λ₂ does.
- **Tuned:** ρ ∈ [0, 0.5] and s ∈ [0.5, 1.0], plus w_goal and sector_bias.

**Claims re-checked with B7 and B8 included (the decision rules are fixed now):**

- **"23–29 %":** recomputed as min–max over the six urban settings of (SENTINEL − best baseline) / best baseline, with the best baseline over B0–B8. The paper states the new range.
- **"never significantly worse than any baseline in its original form":**
  - The claim stays only if no B0–B8 row (without fusion) has a Holm-significant higher mean than SENTINEL in any cell of Tables 2 and 3 (400 m and 200 m), the channel profiles or the severe-channel failure table.
  - Otherwise it is replaced by a statement naming the method and setting.
  - The E5 robustness run is reported separately and does not enter this claim, but any significant loss there is stated in its own sentence.
- **"significantly better at every swarm size from 3 to 50 on the 400 m area":** stays only if every B0–B8 star in the 400 m column set holds. Otherwise the exceptions are named.
- **Channel profiles:** "significantly against every baseline except in the severe one" is re-evaluated with B7 and B8 included.
- **Table 4:**
  - the best-baseline rows are recomputed over B0–B8 (best by mean in the E2 cells and the 200 m, N = 50 cell of E1);
  - the "+ fusion" rows are recomputed over B0–B8 + fusion (E12 cells and E11b);
  - "all seven" becomes "all nine" only if all nine stay significantly behind;
  - the dense-swarm sentence is updated with the new range of baselines + fusion.

## 4. E3: sensitivity of the spring parameters

- **Setting:** 400 m, N = 8, Poisson failures 0.01 s⁻¹, urban rubble and severe, seeds 0–29.
- **Variants:** one parameter at a time around the tuned SENTINEL (λ₀ = 1.20, β = 1.09, ε = 0.05); all other tuned values are inherited:
  - λ₀ ∈ {0.3, 4.8};
  - β ∈ {0, 4};
  - ε ∈ {0.01, 0.20}, set for both the spring target and the preserved-link radius R_ε;
  - distance spring (the existing ablation: nearest neighbour, λ = 0.5, β = 0) with R_max ∈ {30, 50, 70} m.
- **Reference:** tuned SENTINEL, re-run in the same job.
- **Per variant and channel, report:**
  - on-time share: mean ± CI;
  - link-violation rate: Σ violations / Σ preserved link-steps, counted against the variant's own ε;
  - 90th-percentile latency: mean over runs of the per-run p90, as for the paper's T_h table.
- **Tests:** Wilcoxon on the on-time share, variant vs tuned SENTINEL. Holm within each parameter, over that parameter's variants and both channels: λ₀ 4 tests, β 4, ε 4, R_max 6. The violation rate and latency are descriptive.
- **Paper sentence:** the range of the on-time share and violation rate across all variants, and which variants differ significantly.
- **Paper's "over all experiments" violation count:**
  - It is extended with the new SENTINEL-family runs that use the default ε = 0.05: E3 variants with λ₀, β and distance spring; E5 robustness SENTINEL. The E3 reference SENTINEL is excluded as a duplicate of E7.
  - The ε = 0.01 / 0.20 variants are reported only in the sensitivity table.

## 5. E4: detector at the operating altitude (rendered imagery)

- **Worlds:**
  - `make_world.build` (same generator as the PX4 check), 100 m, 20 victims, world seeds 100–123;
  - train 100–113 (14 worlds), val 114–118 (5), test 119–123 (5);
  - no world is in two splits (asserted).
- **Frames:**
  - Viewpoints at 25 m and 30 m: 3 per altitude over each non-buried victim (offset uniform in a 9 m disk), 10 over warm debris and 10 random, never above or within 2 m of a building.
  - Static rig with the on-board camera model (640 × 512, 50° HFOV, L16, nadir) and two bounding-box cameras with the same intrinsics: visible `2d` and `full_2d`.
  - Each frame is captured ≥ 0.3 s of simulated time after the pose change. It is converted exactly as in `thermal_vision`: 270–330 K white-hot, then CLAHE.
- **Ground truth:** visible victim boxes with area ≥ 40 px² and both sides ≥ 4 px.
  - "Open" means occ0 = 0 in the world definition; anything else is "partly occluded" (including a buried victim of which a part is visible from an oblique view).
- **Metrics on the test worlds, per model:**
  - mAP50 and mAP50-95 (Ultralytics `val`);
  - recall at confidence ≥ 0.40 and ≥ 0.70 for open and for partly occluded victims (IoU ≥ 0.5, greedy by confidence, per threshold);
  - false alarms per frame on warm debris: predictions at the threshold whose centre lies in a debris box and in no victim box.
- **Models:**
  - (a) the deployed YOLO26n (HIT-UAV only);
  - (b) the same model fine-tuned on the train worlds with its original training arguments, read from the checkpoint: AdamW, lr0 0.002, batch 16, 640 px, seed 0, same augmentation, 100 epochs, `best.pt` by val fitness.
- **Report:** frames and boxes per split, plus the caveat "rendered imagery, not real 25–30 m footage".

## 6. E5: detector model fitted to detector outputs

- **Data:** the fine-tuned detector's predictions on the **val** worlds.
- **Observation:** one per frame and in-image, non-buried victim. It is the highest prediction confidence with its centre in the victim's full box (4 px margin), or "not reported" (below 0.25).
- **Likelihood (Tobit):**
  - c ~ clip(N(μ, σ²), 0, 1) with μ = c_max (1 − occ)^γ exp(−max(0, r − r_ref)/r_s) and occ = clip(occ0 + κ sinθ cos(az − block_dir), 0, 1);
  - non-reports are left-censored at 0.25;
  - L-BFGS-B within fixed bounds, 16 starts.
- **Identifiability:**
  - The rendered slant ranges are about 25–35 m. If r_ref or r_s end at a bound, they are reported as not identifiable in that range.
  - The fitted values are still used: in the 400 m simulations the slant range is 30–32 m, inside the fitted range.
- **False alarms:**
  - λ_d = all false alarms (conf ≥ 0.25, no victim box) / warm-debris items in view, on the val frames;
  - `fp_rate` = λ_d × mean number of debris items in the 400 m evaluation worlds (seeds 0–29). In the simulator each draw lands on a random debris item and counts only if it is in the footprint, so the expected false alarms per look per debris item in view equal λ_d;
  - fp_mu and fp_sigma come from a normal truncated at 0.25.
- **Goodness of fit (test worlds):**
  - observed vs predicted P(c ≥ 0.25 / 0.40 / 0.70) per slant-range bin (20–27, 27–30, 30–32, 32–35, 35–45 m) and occlusion class;
  - test log-likelihood of the fitted and the hand-set parameters.
- **Robustness re-run:**
  - SENTINEL, B0–B8, urban rubble, 400 m, N = 8, six failure settings, seeds 0–29, with the fitted detector model;
  - every method keeps its tuned values, with no re-tuning;
  - same statistics as Table 2 (9 tests per column);
  - reported as "with the detector model fitted to detector outputs".

## 7. E6 and E8r: PX4/Gazebo

**Finding before launch (20:38).** The ROS 2 install (`~/sentinel_ws/install`, built 14 Oct 22:15) was older than the source.
- Its ledger defaulted to naive-Bayes fusion and `emps_agent` does not set the rule.
- The paper's PX4 check (E8, 15 Oct) therefore ran SENTINEL with naive-Bayes fusion in Gazebo, against corroborative fusion in the fast simulator.
- The install was rebuilt from the current source (`colcon build --packages-select sentinel_core sentinel_ai`), and every module now matches the source.

**E8r:** the original check (worlds 0–2, parameterised detector model) re-run on the current code; same protocol, new folder. The paper's PX4 sentence is regenerated from E8r. The old E8 files stay untouched.

**E6:**
- 10 worlds (seeds 0–9, 100 m, 20 victims, responder; worlds 3–9 generated with the same generator), 3 PX4 x500 UAVs, crash of UAV 1 at 150 s, 300 s, SENTINEL and B0.
- **YOLO26n in the loop:** the fine-tuned model from E4. The HIT-UAV-only model detects few rendered victims (pilot world, not evaluation data), which would make the comparison uninformative. If E4 fails, the HIT-UAV model is used and this is stated.
- **Metric:** victims reported at the end (`tdr_base`). Paired two-sided Wilcoxon SENTINEL vs B0 over the 10 worlds; mean paired difference with its bootstrap CI.
- **Fast-simulator counterpart:** the same 10 worlds with the E5-fitted detector model, for agreement. The fast side of E8r is the existing `results/e8_fastsim` (unchanged code, deterministic runs).

## 8. Reporting rules

- Every outcome is reported, including losses of SENTINEL and wins of B7/B8.
- No seed, setting or variant is dropped.
- Failed jobs are re-run once; if they fail again, they are reported as not completed.

## 9. Deviations (dated, added after launch)

**28 Oct, 20:45–21:28 — E4 rendering (lane B), capture method only.**
- The render job failed twice in the main queue: gz-transport `set_pose` timeouts under CPU load, then an abort in the gz-transport destructor at exit.
- Lane B was re-run on its own with `lane_b_resume.sh` (same job names and done markers). The renderer gained three changes:
  - a per-world retry;
  - a sticky fallback to the `gz service` CLI for `set_pose`;
  - from 21:24, the capture loop runs in a system-Python subprocess (`e4_capture.py`, native protobuf) after in-process subscriptions stalled.
- Unchanged: worlds, rig, sensors, viewpoints, the ≥ 0.3 s settle time, the frame-stamp check and the conversion. 19 of 24 worlds were captured before the switch and 5 after.
- No world or frame was dropped. All 24 worlds are complete.

**29 Oct, 00:47 — main queue stopped and resumed (`resume_queue.sh`).**
- **b3 (HIT-UAV evaluation):**
  - It deadlocked from 21:28 in Ultralytics' forked dataloader workers, with no progress for 3 h 15 min.
  - Fix: `val` now runs with `workers=0`, which does not change the metrics. The fine-tune's dataloader workers use the `spawn` start method; the training arguments are unchanged.
- **D1 (E8r):**
  - Every launch failed immediately: `model_path:=` with an empty value is not a valid launch argument. `run_sitl.sh` now passes `model_path` only when it is set and stops waiting when the launch exits.
  - Both PX4 paths were piloted for 30 s on world 0 into a scratch folder, not evaluation data.
  - No PX4 evaluation run had completed before the fix, so nothing was discarded.
- **C1 and C2 did not run** (skipped by the dependency check because lane B had not finished). They now run as soon as lane B writes the fitted configs.
- **Order change:**
  - C1 and C2 now run at nice 19 in parallel with the PX4 runs, instead of before them. Fast-simulator results do not depend on load or worker count.
  - D1 runs in parallel with lane B's fine-tuning (GPU).

**29 Oct, 01:09 — stale rendered frames found; seven worlds re-rendered, E4/E5 re-run.**
- **What was found.** The fitted slant range reached 114 m, which is impossible at 25–30 m altitude with a 50° camera.
  - A reprojection audit compared each frame's boxes with the victims and debris projected from its recorded pose. A matching frame has a median residual of 1.5 px.
  - 91 frames failed, all in worlds 107 and 110–115, which were rendered before the 21:24 capture switch. Each one shows an earlier viewpoint, because the pose change had not been rendered yet. The 0.3 s settle time was measured from the newest image received, and under load the image stream lagged the simulation.
  - The 17 other worlds, including all five test worlds, have no mismatched frame.
- **Effect on the first pass:**
  - Images and labels in the stale frames are consistent with each other, so they are valid training and evaluation frames, only duplicated.
  - Their recorded pose is wrong, which corrupts the E5 covariates (val world 115).
  - E4 test metrics come from clean worlds, but the model was trained on the duplicated frames.
- **Fix:**
  - The capture now checks every frame against its pose: every box clear of the border must lie within 15 px of its object's projection. A failing frame is retaken, at most 6 times, with the pose re-sent halfway.
  - The render job then audits all frames of all 24 worlds (`e4_posecheck.py`) and fails on any mismatch.
  - The check flags exactly these 91 frames and none in the clean worlds.
- **Re-run:**
  - worlds 107 and 110–115 re-rendered with the same viewpoints;
  - dataset rebuilt (test frames unchanged);
  - both models evaluated, fine-tune, E5 fit and configs re-run;
  - C1 and C2 run on the new fit (C1 had started on the first-pass fit at 01:07 and was stopped with no finished mission);
  - E6 waits for the new fine-tuned model.
- **First pass archived** in `firstpass_stale_frames/` (outputs, configs and done markers) and reported as superseded:
  - fine-tuned test mAP50 0.992, recall at 0.40 of 0.97 open and 0.94 partly occluded;
  - fit with r_ref and r_s at their bounds, σ 0.17, fp_rate 6.1.
- **E8r:** the three PX4 runs finished before the stop are kept, since they do not use E4 or E5. Unfinished runs are redone from the start.

**29 Oct, 02:23 — E6 (D2) stopped and restarted (`resume_d2.sh`).**
- **Cause:** the ROS nodes read each world's truth file from the colcon install, which held only worlds 0–2 from 14 Oct; Gazebo loads worlds from the source folder.
  - On world 3, generated tonight, the agent, comm-emulator and metrics nodes exited at start-up while Gazebo and the cameras kept running.
  - The launch process stayed alive, so the run would have waited out its 40 min timeout.
- **Fix:**
  - `run_sitl.sh` passes `world_dir` (the source folder Gazebo uses) to the launch, checks that the truth file exists, and stops a run when a node dies.
  - World 3 was piloted for 30 s in a scratch folder.
  - Worlds 4–9 were generated with the same generator. The victim positions of all ten worlds equal those of the fast simulator's world generator for the same seeds, which C2 uses.
- **Kept:** worlds 0–2 of E6 and all of E8r. Their truth files in the install and source folders are byte-identical, and Gazebo used the source worlds throughout. The aborted world-3 run is redone from the start.
- **Order change:** C1 and C2 continue in `resume_queue.sh`, and D2 runs in `resume_d2.sh`.

**29 Oct, 13:50 — E5 false-alarm conversion corrected; C1 and C2 re-run (pre-submission check).**
- **Check run first** (`scripts/overnight/e5_check.py`, results in `results/e5_recheck_2026-10-29/check_before/`). The fitted model was run through the simulator's own `DetectorModel.look` on the E4 frames (same camera poses, victims and debris, 200 draws) and compared with the fine-tuned YOLO26n on the same frames.
  - Victim recall and confidence agree within a factor of 2.
  - False alarms per frame (≥ 0.25), real val 0.018 (10 in 546 frames), real test 0.0055 (3 in 545):
    - with 480 debris items per world (the 400 m worlds of C1): 0.011 val, 0.012 test;
    - with 30 items per world (the 100 m worlds of C2): 0.17 val, 0.19 test, about 10× (val) and 34× (test) the real rate.
- **Cause: the conversion to `fp_rate`, not the simulator.** `fp_rate` is the mean number of false-alarm *candidates* per look, before the 0.25 floor. Each candidate lands on a debris item drawn from the **whole world** and counts only if that item is in the footprint. The intent stated in §6 (expected false alarms per look per debris item in view equal λ_d) was missed three ways:
  1. **Floor applied twice.** λ_d counts reported false alarms (≥ 0.25), but the simulator applies the floor again to N(fp_mu, fp_sigma) (latent normal, 0.172 ± 0.160), so only 31 % pass. The candidate rate must be λ_d / P(c ≥ 0.25).
  2. **Denominator.** λ_d divided by visible debris *boxes* (0.85 per val frame). The simulator draws at every debris item in its footprint, and about half the items lie inside the rubble they are generated at and render no box (1.58 items per val frame by projection). The denominator is now the number of the world's debris items that project into the frame.
  3. **World size in C2.** `fp_rate` = λ_d × 480 holds only in worlds with 480 items. `make_fitted_configs.py` copied it unchanged into the E6 fast-simulator config, whose 100 m worlds have 30 items: 16× the rate per item.
  - In C1, (1) and (2) partly cancel (0.31 × 1.85 ≈ 0.58×). C2 is off by about 10×.
- **Fix:**
  - `e5_fit.py` stores the world-independent rate per debris item in view (`fp_per_debris_item`, after (1) and (2)).
  - `make_fitted_configs.py` sets `fp_rate` per config from the number of debris items its worlds have (480 for C1, 30 for C2, checked on the world generator).
  - The victim parameters, data, definitions and the analysis are unchanged. Tests cover the conversion: simulated false alarms per debris item in view equal λ_d in worlds of 30 and 480 items.
- **Re-run into `results/e5_recheck_2026-10-29/`** (new folder; the first results stay untouched and are reported as superseded):
  - E5 fit;
  - the check, which must now match the real rate in both world sizes;
  - C1 (`robust_fitted_v2`: same methods, settings, seeds 0–29, tuned values and statistics);
  - C2 (`e6_fastsim_v2`).
  - Paper, highlighted build, letter and extended results are regenerated from the new C1/C2. The PX4 side of E6 (real YOLO26n in the loop) does not use the model and is unchanged.

**29 Oct, 15:20 — Baselines B0–B4 and B6 re-tuned and re-run (fusion-rule inconsistency found by the reproducibility check).**
- **Finding.** After the `detector_model.py` docstring edit, stored hand-set-detector missions were re-run (`check_reproduce.py`). Every SENTINEL, B5, B7, B8 and sensitivity row is bit-identical, but rows of B0–B4 and B6 in E1–E4 and E6 are not.
- **Cause, confirmed.**
  - These rows date from before Amendment A3 (14 Oct). At that time the ledger's fusion rule was a module global defaulting to naive Bayes.
  - Since A3, every run sets the rule per run, and the default is corroborative for every method. `fusion=False` drops extra views only at local sensing, so gossip-merged views are still fused, and the rule changes baseline confirmations. A3's note ("baseline rows are unchanged and valid") was wrong.
  - On the current code with `fusion_rule=naive_bayes`, every failing mission reproduces bit-identically. The only remaining differences are the three link-preservation columns, whose definition changed on 22 Oct (metric only).
  - No commit since 680e05e reproduces these rows, and no library has changed since 15 Oct.
- **Tuning check on the current code** (`tune_check.py`, tuning objective = mean report value on the tuning scenarios, seeds 1000–1002):

  | Method | Recorded | On HEAD |
  |---|---|---|
  | SENTINEL (A4, 15 Oct) | 0.2966 | 0.2966 |
  | B5 (16 Oct, with its policy) | 0.2332 | 0.2332 |
  | B7 (28 Oct) | 0.2307 | 0.2307 |
  | B8 (28 Oct) | 0.2720 | 0.2720 |
  | B0 (round 1, 10 Oct) | 0.285 | 0.242 |
  | B1 | 0.288 | 0.274 |
  | B2 | 0.298 | 0.269 |
  | B3 | 0.209 | 0.230 |
  | B4 | 0.062 | 0.059 |
  | B6 | 0.183 | 0.188 |

  SENTINEL, B5, B7 and B8 are unchanged on HEAD and keep their values. B0–B4 and B6 were tuned on 10 Oct code and are re-tuned.
- **Decision (authors, 29 Oct):**
  - Re-tune B0–B4 and B6 on HEAD with the same protocol: Optuna TPE seed 0, 30 trials, the same objective, scenarios and tuning seeds, one method at a time as for B7/B8.
  - Then re-run every row of B0–B4, B6 and their fusion variants, in every experiment containing them, so all tables use one tuning:
    - pre-A3 rows: E1, E2, E3 (main tables), E4, E6, E6c, E8-fast, E11;
    - post-A3 rows with the old tuned values: E9, E11b, E12 (Table 4), C1 (robustness, `robust_fitted_v2`), C2 (`e6_fastsim_v2`).
  - Same specs (base, grid, seeds 0–29 or as specified), same statistics, 8,023 missions in total.
- **Kept:** SENTINEL, B5, B7, B8 and every other row; they reproduce on HEAD.
- **Mechanics:**
  - `scripts/overnight/run_baseline_rerun.sh` (resumable queue, two attempts per job, done markers, completeness check), starting after the C1 queue:
    - re-tune → `configs/tuned_baselines_v4.yaml`;
    - `configs/tuned_v4.yaml` = `tuned_v3.yaml` with B0–B4/B6 replaced (`merge_tuned_v4.py` checks that nothing else changed);
    - configs `configs/experiments/rerun_2026-10-29/` (`make_rerun_configs.py`);
    - results in `results/baseline_rerun_2026-10-29/<experiment>/`.
  - `make_ios.py` replaces the affected rows by the re-run rows, asserting the same cells and seeds. The original result folders stay untouched; the superseded rows are archived as a copy, and the extended results report them as superseded.
- **Pilot** (one cell, one seed, two methods per experiment, old tuned values): all 13 configs run and pass the completeness check, and their run IDs equal the stored rows', so cells and seeds line up.
- **Estimate:** about 155 CPU-h from the stored run times, about 8–9 h on 19 workers plus about 20 min of re-tuning. After C1 (about 16:30), the queue should finish around 02:00–03:00.
- **Before launch:** reproducibility check on every result folder the paper uses (all methods, two seeds each: `results/baseline_rerun_2026-10-29/repro_before/`).
- **After the queue:** archive, regenerate, re-run every claim check, then update the paper, highlighted build, letter and extended results, check and commit.
- **15:25 addition — E7 no-projection ablation.**
  - The reproducibility check also flags `abl_no_projection` in `e7_ablation` (15 Oct).
  - Seed 0 differs only in `link_reliable_start_frac` (metric definition, 22 Oct). Seed 29 also differs in coverage, path length and deliveries.
  - Cause: since 22 Oct the agent evaluates `_tethered(t)` also when the projection is off, to count link violations for this ablation. That call stamps `tether_until` for a delivering, connected UAV, which other UAVs pick up through beacons, so the ablation's behaviour changed slightly.
  - Its rows in E12 (Table 4) were run after the change and reproduce. Baselines are not affected: strict connectivity returns before the stamp.
  - The 120 rows of `abl_no_projection` in E7 (extended results) are added to the queue (`r_e7_ablation`) with SENTINEL's unchanged tuned values.
- **16:38 — reproducibility check complete; no further stale group.**
  - The five N = 50 folders, checked in parallel (two seeds each), finished at 16:38.
  - `e1_scalability`: only B0–B4 and B6 differ (10 of 16 missions); SENTINEL and B5 are identical.
  - `e11_dense_diag`: only B0 differs (2 of 12); SENTINEL and all four ablations are identical.
  - `e11b_dense_fusion` (16 of 16), `b7b8_scalability` (4 of 4) and `b7b8_dense_fusion` (4 of 4) are identical.
  - With the earlier folders, every differing row belongs to the two groups already in the queue (pre-A3 baseline rows and E7 `abl_no_projection`). The queue is unchanged.
- **16:57 — re-tune result.** `r0_retune` finished in 15 min (30 TPE trials per method, seed 0, objective and tuning seeds as before); `tuned_v4.yaml` = `tuned_v3.yaml` with B0–B4/B6 replaced (merge asserted that nothing else changed).
  - Tuning objective on the released code, old values → re-tuned values (re-checked from the merged file with `tune_check.py`, identical to Optuna's best value): B0 0.242 → 0.289, B1 0.274 → 0.272, B2 0.269 → 0.273, B3 0.230 → 0.240, B4 0.059 → 0.046, B6 0.188 → 0.189 (SENTINEL 0.297, unchanged).
  - For B1 and B4 the 30-trial search ended slightly below the old values' objective (B1 by 0.002, B4 by 0.013; B4 stays the weakest method by far). The protocol is kept as decided (same budget for every method, SENTINEL included, no warm start from earlier values); this is reported, not corrected.
- **30 Oct, 04:10 — baseline re-run complete.** All 14 experiment jobs DONE on attempt 1 (queue end 04:09:27); 8,143 runs, each folder passes the completeness check, no errors in the logs. Superseded rows archived as copies (`results/baseline_rerun_2026-10-29/superseded_rows/`, README); `make_ios.py` replaces them on read (`rerun()`, asserting the same cells and seeds and the old/new tuned values).
- **30 Oct, 04:15 — deviation: PX4 checks keep B0's earlier values on both sides.** Both PX4 checks (E8r, and E6 with YOLO26n) flew B0 with `configs/tuned.yaml` (= `tuned_v3`). Their fast-simulator side must use the same values, so it does not use the re-tuned B0: E8r's fast B0 was re-run with the `tuned_v3` values on the released code (3 runs, `results/baseline_rerun_2026-10-29/sitl_match/e8_fastsim/`; same run IDs and results as the stored rows, only the timing column differs), and E6's fast side uses the C2 v2 rows (released code, `tuned_v3` values). `make_ios.py` asserts that these B0 configs carry the PX4 runs' values. The re-tuned fast-side numbers (E8r worlds B0 0.62, E6 worlds B0 0.94) are given in the extended results.
- **30 Oct, 04:20 — claims that changed with the re-run rows** (every sentence now has a generated assertion in `make_ios.py`):
  - Table 2 (urban failures): SENTINEL over the best baseline 23–29 % → 7–18 %; not significant against B2 under Poisson crashes (was significant against all).
  - Table 3 (400 m): B2 has the highest mean at N = 3 (0.23 vs. 0.20, not significant); SENTINEL significantly above all from N = 10 (was N = 3 to 50).
  - 200 m: a baseline (B0, B1, B3) has the highest mean at every N, never significantly (was SENTINEL except N = 20).
  - Table 4: urban, B2 + fusion no longer significantly behind (0.549 vs. 0.575); dense, baselines + fusion 0.784–0.851, B4 + fusion significantly above SENTINEL in Table 4's Holm family (p_holm = 0.018) but not with Holm over all nine (B1, B3, B4 p_holm = 0.08). The text follows Table 4, the less favourable reading; both are in the extended results.
  - E9: the mesh stays connected in 89 of 90 runs (one B0 run split for 4 % of the mission; was all runs).
  - Robustness (fitted detector model): SENTINEL highest mean in 4 of 6 settings (behind B0 with 30 % simultaneous crashes, B3 with cascading), −1.5 to +3.4 %, never significant.
  - Unchanged: channel profiles (significantly above all in open, suburban, urban), no original-form baseline significantly ahead of SENTINEL in any cell, attribution of the gains to fusion and delivery, link-preservation result.
  - The conclusion's "most settings" is now generated: highest mean in 19 of 26 channel, failure and swarm-size settings.
