# SENTINEL v2 — Progress Log

The governing plan is [REWORK_PLAN.md](REWORK_PLAN.md). This file records what has been done, what was found, and what is open, newest first within each phase.
All code is in `src/SENTINEL/sentinel_ws/src/` on git branch **`revision-v2`**. Nothing is committed yet; commits happen only when asked.

---

## Status at a glance

| Phase | Status |
|---|---|
| 0 Environment | 🟢 Done (venv, ros_gz, TensorRT) |
| 1 `sentinel_core` + tests | 🟢 Done: 396 unit tests pass, and Proposition 1 is verified over 10⁴ trials. |
| 2 Fast simulator + baselines | 🟢 Done: all baselines B0–B6 equal-budget tuned; MAPPO trained and selected. |
| 3 Gazebo assets / SITL | 🟢 Done: 3× PX4 x500_thermal; radiometric L16 thermal camera; generated worlds. |
| 4 ROS 2 integration | 🟢 Done: full loop (PX4 + Gazebo + ROS 2), model/YOLO perception switch, E8 cross-check run. |
| 5 Perception harness | 🟢 Done: YOLO26n vs YOLOv8n benchmark (accuracy from checkpoints; test-split re-measure needs the dataset). |
| 6 Experiments | 🟢 Complete (E1–E11, final design A4) |
| 7 Karthik's model | 🟢 Delivered (YOLO26n + YOLOv8n); integrated |
| 8 Paper + response | 🟡 Full draft written, references verified, response matrix filled; needs compile, screenshots, venue formatting. |

---

## 2026-10-10

### Planning
- Read the reviewer decision (IoT-66280-2026, rejected; resubmitting to IoT-J is not allowed) and the draft paper.
- **Code audit:** the repo does not implement the paper.
  - No PSO; `calculate_fitness` is never called.
  - λ is 0.8 in code vs 0.5 in the paper; the SA offset is 10 m vs 15 m.
  - The ledger stores camera-frame offsets and telemetry is always (0, 0), which is the root cause of the "15 coordinates" bug.
  - No world, thermal camera, metrics or baselines.

  ⇒ All paper numbers must be re-measured.
- The GitHub repo (`nicholaschristo/SENTINEL`) has no extra assets, so everything is being rebuilt.
- Wrote and saved `docs/REWORK_PLAN.md`, then hardened it in a self-review:
  - new contribution C1 (information-survivability-aware tension);
  - Propositions 1 and 2, plus the Lemma;
  - statistics plan;
  - reference audit;
  - claims policy.

### Phase 0 — Environment
- Created `~/sentinel_venv` (`--system-site-packages`) with:
  - torch 2.6.0+cu124 (CUDA OK on the RTX 3060);
  - ultralytics 8.4.175 (has YOLO26 configs);
  - numpy 1.26.4 and scipy 1.14.1;
  - opencv-python 4.11;
  - onnx, onnxruntime-gpu 1.23.2 (TensorRT, CUDA and CPU providers);
  - networkx, pandas, pyarrow, shapely, optuna, pytest, hypothesis.
- Pinning `numpy<2` fixes the old `cv2` / `cv_bridge` breakage; `cv_bridge` imports OK.
- ⚠️ **Pending (you):** `sudo apt install ros-humble-ros-gzharmonic`. The package is available (0.244.12), but sudo needs a password.
- Activate the environment with `source ~/sentinel_venv/bin/activate`. `sentinel_core` and `sentinel_sim` are pip-installed in editable mode.

### Phase 1 — `sentinel_core` (pure Python, used by both the simulator and ROS)

| Module | Purpose |
|---|---|
| `channel.py` | Path loss, NLOS, shadowing (Gudmundson), Rician/Rayleigh fading, SINR → outage; profiles open / suburban / urban_rubble / severe; `reliable_range()` |
| `graph.py` | Laplacian, λ2, components, local LMST (Li–Hou–Sha) |
| `safety.py` | Midpoint-ball projection (Proposition 1) |
| `ledger.py` | World-frame ledger, gating plus consolidation, log-odds multi-view fusion, gossip replication with acks, U_i |
| `maps.py` | Bayesian search map shared as a G-counter CRDT |
| `fitness.py` | Coverage + detection − spring (`none` / `distance_nn` v1 / `outage` with C1 adaptive tension) |
| `pso.py` | Constriction PSO (χ = 0.7298) over the reachable waypoint disk, seeded with neighbour waypoints |
| `sa_trigger.py` | ±15 m flank with an admissibility check |
| `detector_model.py` | Confidence vs range, view-dependent occlusion, buried targets, false positives |
| `failures.py` | Single, simultaneous, cascading, Poisson and brownout failures |
| `metrics.py` | NRR, TDR, information loss, false/duplicate entries, recovery time, bootstrap CI, Wilcoxon with Holm correction, Cliff's δ |
| `agent.py` | Decentralised agent used by the simulator and (later) the ROS node |
| `baselines/` | B3 λ2-gradient (Sabattini; oracle λ2, which favours the baseline), B4 potential with hysteresis (Ji & Egerstedt), B6 lawnmower |

**Channel calibration:** the urban_rubble reliable range at ε = 0.05 is 91 m LoS and 48 m with one obstruction. The 48 m figure is close to v1's 50 m, so it gives a principled justification for the old value.

| Profile | R₀.₀₅ LoS | R₀.₀₅ NLOS1 |
|---|---|---|
| open | 2434 m | 684 m |
| suburban | 339 m | 95 m |
| urban_rubble | 91 m | 48 m |
| severe | 35 m | 21 m |

**Tests:** 359 pass (`python -m pytest test/` in `sentinel_core`). They cover:
- the Rayleigh closed form;
- Monte Carlo vs analytic outage;
- λ2 of path, star and complete graphs;
- LMST ⊇ MST and connectivity preservation;
- the Lemma counter-example (the nearest-neighbour rule does not imply connectivity);
- projected motion preserves connectivity;
- PSO convergence;
- the **R1.6 regression test** (a ledger never holds more than the number of true targets);
- the CRDT merge.

**Bugs the tests caught (and that were fixed):**
1. **Ledger duplicates:** the same victim was entered twice when the position estimates were 3.3 m apart. The fix is a 5 m gate plus a consolidation pass with uid aliases. This is exactly the mechanism behind v1's "15 coordinates".
2. **Projection boundary:** floating-point error could leave a link at R + 1e-12. The fix is a relative slack of 1e-7.
3. **Overshoot:** a UAV outside a ball and moving inward could overshoot. It is now capped at "no worse than now".

**Proposition 1, Monte Carlo** (`sentinel_ws/scripts/verify_prop1.py`): **10,000 trials × 25 steps, N = 2–50, 0 failures.**

### Phase 2 — Fast simulator (`sentinel_sim`)
- `world.py`:
  - procedural disaster maps: tall buildings that obstruct links, low rubble, moving debris, victims (one buried), warm clutter;
  - `disaster_25` rebuilds the v1 scenario;
  - obstruction counting uses an STRtree, optionally with buffered obstacles.
- `linkfield.py`: vectorised per-tick delivery with correlated shadowing, fading and collision interference.
- `methods.py` registers:
  - `sentinel`;
  - baselines B0_pso, B1_v1, B2_commpso, B3_lambda2, B4_potential, B6_lawnmower;
  - six ablations.
- `sim.py`: common random numbers per seed (needed for paired tests), failure injection, ground-truth metrics, about 8 s per 200 s run with N = 5.

**Design fixes found through simulation:**
1. **Staleness margin.** The correct bound is ½·v_j·age per side; it had been double-counted. Static nodes get a margin of 0, so beacons now carry v_max. Recovery no longer breaks other links (it hovers if needed).
2. **Obstruction-aware hard radius.** Each link uses R_ε(n_obs) from the building map. A backtracking step rejects moves that would hide a preserved link behind a building. Obstacles are inflated by 6 m for robustness.
3. **Self-healing re-attachment.** The root (responder) heartbeat is relayed through the mesh. A UAV whose heartbeat goes stale (over 4 s) returns toward the last relay.

**More design fixes (later the same day):**
4. **Joint-motion occlusion.** A link could swing behind a building when *both* UAVs moved. Each UAV now evaluates link radius R_ε(n_obs) at its *proposed* position, against the neighbour's last position, with obstacles inflated by v_j·(dt + age). Each side's radius then lower-bounds the true range, so ‖x_i′ − x_j′‖ ≤ (r_i + r_j)/2 ≤ R_true. Moves that fail are halved; hovering is the fallback.
   - Result: **100% connected on 8/8 seeds** (previously as low as 24%).
   - An earlier over-conservative version froze the swarm (coverage about 10%) and was replaced.
5. **Batch runner** (`sentinel_sim/runner.py`): YAML grid × methods × seeds, process pool (18 workers), parquet output, resumable.
6. **New metric:** report latency from first detection to the responder, and the share of detections reported within 30 s.

**Pilot results** (preliminary, not for the paper; stored in `results/pilot`, `results/pilot2`). Setup: N = 6, 150 m map, urban_rubble channel, 400 s, 8–10 seeds.

| Method | Connected | Coverage | TDR at responder | Info loss |
|---|---|---|---|---|
| SENTINEL | **1.00** | 0.49 | 0.33 | 0.00 |
| B3 λ2 (oracle) | 0.94 | 0.42 | 0.33 | 0.00 |
| B4 potential | 1.00 | 0.05 | 0.03 | 0.00 |
| B0 PSO | 0.13 | 0.96 | 0.68 | 0.00 |
| B1 v1 | 0.13 | 0.90 | 0.61 | 0.02 |
| B6 lawnmower | 0.27 | 0.94 | 0.63 | 0.04 |

**Interpretation (important):**
- SENTINEL is the best of the connectivity-preserving methods: it covers more than B3 (even with B3's oracle) and far more than B4.
- In this scenario, though, connectivity-blind methods still deliver almost everything to the responder within about 1 s. The 150 m map is small relative to radio range, so the packet network rarely truly partitions.

⇒ **This scenario does not stress connectivity.** The experiments must include regimes where it matters (maps much larger than R, the severe channel, many failures). The paper should report a regime map that shows honestly where communication awareness pays off and what it costs in coverage. SENTINEL's soft spring may also be over-tight; to be tested (λ0 sweep).

### Phase 2 — stressed-regime pilot + speedup, 2026-10-10
- **Simulator speedup of about 19×** (100 s, N = 8, 400 m map: 88 s → 4.6 s):
  - a cached list of moving obstacles (the per-look debris check was 68% of runtime);
  - an interpolated outage lookup table (error < 2e-3, unit-tested).

  363 tests pass.
- **Pilot 3** (`results/pilot3`): 400 m map, N = 8, 25% simultaneous crashes, 8 seeds; mean values:

  | Urban channel | Coverage | TDR at responder | Info loss | Mean latency |
  |---|---|---|---|---|
  | B6 lawnmower | 0.91 | 0.58 | 1.5% | 7.3 s |
  | B0 PSO | 0.50 | 0.28 | 2.8% | 4.4 s |
  | B3 λ2 oracle | 0.17 | 0.12 | 0% | 1.7 s |
  | SENTINEL (λ0 = 0.1 / 1 / 0) | 0.08–0.10 | 0.07–0.09 | 0% | 0.4–0.8 s |

  The severe channel shows the same pattern: SENTINEL coverage is about 5%, and the lawnmower still delivers 54% (latency 40 s).

**⚠️ Key finding (decision needed).** Keeping the whole swarm permanently connected to the responder (SENTINEL and B3) severely limits reach in large areas. Store-and-forward methods lose very little data and deliver far more victims overall. SENTINEL wins only on zero loss and latency. The soft-spring strength λ0 barely matters (the hard constraint dominates). Reconnection after mass crashes is also incomplete (connected only 25–40% of the time in the severe channel).
⇒ As designed, SENTINEL is **dominated on the main SAR metric** in large areas. A design decision is required (see Open issues).

### Phase 2 — design pivot implemented (Amendment A1), 2026-10-10
- **Decision (user):** information-triggered intermittent connectivity, recorded as Amendment A1 in REWORK_PLAN.md.
- **Implementation (`agent.py`):** an EXPLORE/DELIVER mode switch.
  - A detection is *secured* once the responder appears in its holder set (no responder: once another UAV holds a copy).
  - DELIVER starts when the oldest unsecured detection reaches T_hold. The UAV flies toward the responder, and the channel-aware projection (Proposition 1) is active only while it is connected and delivering.
  - The old design is kept as `sentinel_strict`. Baselines are pinned to their published behaviour.
- **Exploration layer:** frontier goal plus soft sector preference, shared by every PSO-based method. Coverage rose from 0.66 to about 1.0 on 400 m maps. This is not claimed as a contribution.
- **Pilot 4** (`results/pilot4`): 400 m map, Poisson failures (rate 0.01/s), 600 s, 6 seeds. These are *preliminary* numbers; the final results need ≥30 seeds.

  | Regime | Best on reported-to-responder | Notes |
  |---|---|---|
  | Severe channel, N = 4 (sparse contacts) | **SENTINEL T_hold=30: 0.31** vs B0 0.23, lawnmower 0.27 | B0 *detects* 0.46 but only 0.23 reaches the responder: the Data-Loss Paradox measured directly. SENTINEL's mean latency is 23–35 s vs B0's 71 s; on-time within 30 s is 0.55–0.59 vs 0.23. |
  | Severe channel, N = 8 | B0 0.65 (p90 latency 228 s); SENTINEL T_hold=120 0.63 (162 s); T_hold=30 0.56 (83 s) | Trade-off: the lawnmower (0.49, 93 s) is dominated by T_hold=30. |
  | Urban, N = 4 | B0 0.40, lawnmower 0.38, SENTINEL 0.37–0.38 | On par for total reports; SENTINEL T_hold=0/30 have much lower latency (12–22 s vs 60 s). |
  | Urban, N = 8 | SENTINEL T_hold=30: 0.66 ≈ B0 0.64 | On par. |
  | SENTINEL-strict (any regime) | 0.05–0.08 | Confirms the pivot was necessary. |

- **Honest framing for the paper:** SENTINEL provides a *controllable latency vs discovery trade-off* (T_hold) and wins when contacts are sparse. In dense-contact regimes it matches opportunistic store-and-forward on totals while cutting latency. We do **not** claim to be universally better. The paper will show Pareto fronts (reports vs p90 latency) per regime.

### Phase 2 — realism fixes, metrics, tuning, MAPPO, 2026-10-10
- **Building no-fly zones:** UAVs previously flew straight through 35–60 m buildings.
  - PSO candidates inside buildings (3 m buffer) are rejected, and steps rotate away from them (±135°) or hover. The lawnmower skips blocked route points.
  - Verified: 0 UAV-ticks inside buildings for every method.
- **World generator fix:** building sizes scaled with the map (a 400 m map got 96 buildings covering more than the whole map). Footprints are now an absolute 10–25 m at 4 per hectare, so no-fly covers 16–19% of the map. Coverage is measured over reachable cells only.
  - All earlier pilot numbers came from the old generator and are superseded.
  - Gazebo worlds regenerated: `sentinel_d25_s0`, `sentinel_g100_s0`, `sentinel_g200_s0`.
- **New reporting metrics**, with *all* true targets as the denominator:
  - `rep_within_60s` and `rep_within_120s`;
  - `report_value` = mean of exp(−t_report/300 s), a time-discounted SAR value.

  The old `on_time_30s` used detected targets as its denominator, which flattered methods that detect little.
- **Equal-budget tuning** (`sentinel_sim/tune.py`):
  - Optuna TPE, 30 trials per method, 6 runs per trial (urban N = 8 and severe N = 4; 400 m; Poisson failures).
  - Tuning seeds start at 1000, separate from the evaluation seeds 0–99. The objective is `report_value`. Output: `configs/tuned.yaml` (runner `tuned_file`).
- **Analysis** (`sentinel_sim/analyze.py`): bootstrap 95% CIs, paired Wilcoxon + Holm vs SENTINEL, Cliff's δ, Markdown tables, Pareto plots.
- **B5 MAPPO** (`baselines/mappo_agent.py`, `sentinel_sim/marl.py`):
  - Parameter-shared actor on local observations (26 features), with 10 actions: 8 headings, hover, return.
  - Centralised critic on [global state, local observation], with the same team objective (report value + coverage).
  - Training seeds start at 2000. The rollout pipeline is tested; training starts after tuning finishes.

### Phase 2 — tuning round 1 + SENTINEL refinement, 2026-10-10
- **Equal-budget tuning (30 trials × 6 runs per method)**, best `report_value` on the tuning seeds (`configs/tuned_round1.yaml`):

  | B2 comm-PSO | SENTINEL | B1 v1 | B0 PSO | B3 λ2 | B6 lawnmower | B4 potential |
  |---|---|---|---|---|---|---|
  | 0.298 | 0.293 | 0.288 | 0.285 | 0.209 | 0.183 | 0.062 |

  - The top four are within tuning noise.
  - B3's and B4's tuned gains hit the *lower* bound of their ranges: they do best with their own connectivity mechanism almost switched off. This is an honest finding to report; it is consistent with pilot 3.
- **Refinement:** B2's edge suggests a *soft* channel-aware pull while exploring helps opportunistic delivery. That pull is SENTINEL's own outage spring.
  - New parameter `explore_lam`: a soft outage spring while exploring (no projection); the hard tether still applies while delivering. `explore_lam = 0` reproduces the previous design.
  - SENTINEL is being re-tuned with a fresh, equal 30-trial budget; the other methods keep round-1 values.

- **SENTINEL re-tuning (round 2, with `explore_lam`)** scored worse (0.251 < 0.293) because a larger space with the same 30 trials adds noise. **Not adopted:** SENTINEL keeps its round-1 values (an equal 30-trial budget for every method). Recorded in `configs/tuned_sentinel_round2_rejected.yaml`.
- **Guarantee gap fixed — tether propagation.** A delivering UAV's relay neighbours might be exploring (not projecting), so the delivery path could break. A delivering, connected UAV now stamps `tether_until = t + 2 s`, and the stamp propagates through beacons (max), so the connected component holds its links during delivery and releases them 2 s later (no deadlock). Proposition 1 thus covers every link on the delivery path. Test runs (severe, N = 8): 56–68% reported, latency 12–32 s. The tuned values predate this change; they will be re-checked at evaluation.
- **MAPPO (B5) training started:** 120 iterations × 16 episodes, about 50 s per iteration; report rate 0.7% → 4.8% in the first 3 iterations.

### Long MAPPO run complete → B5 updated (2026-10-16)
- 600 iterations × 16 episodes. Training curve: 0.116 (first 50 iterations), then a **plateau at 0.222** for iterations 150–599, with 0.237 over the last 50. This shows convergence (figure `paper/figures/mappo_curve.pdf`).
- Selection on the tuning seeds: **0.233 (long) vs 0.153 (previous)**, so the long run was adopted. B5 rows of E1–E4 were re-run and the results regenerated.
- **New B5 performance** (reported within 60 s):
  - 0.33–0.50 in E2–E4 (previously 0.16–0.26);
  - **strongest baseline in dense swarms** (200 m, N = 10/20/50: 0.691/0.729/0.779; 400 m, N = 50: 0.725).
- **Effect on the headline:**
  - SENTINEL is still significantly better than all 7 baselines in E2 open/suburban/urban, E3 urban (all 6 models), E4 (all) and E1 at 400 m (every N), and wins at 200 m N = 50 (0.809 vs 0.779).
  - **Changed:** at 200 m N = 10 and N = 20, SENTINEL vs MAPPO is not significant (0.720 vs 0.691, 0.743 vs 0.729); previously N = 10 was a significant win.
  - Still never significantly worse anywhere.
- Paper updated: evaluation (MAPPO training and convergence figure; corrected 200 m claim), introduction ("comes close in small, dense swarms"), limitations (v), and response matrix R2.6. Tables relabelled "B1 Earlier design". Lint 0 issues.

### Gazebo capacity test + two robustness fixes (2026-10-15 evening)
- **Capacity** (MAPPO paused for clean measurements):
  - with thermal cameras: 3 → RTF 1.00, 6 → 0.73, 9 → 0.70 (all fly), 12 → 0.36 (11/12 fly);
  - camera-less: 9 → 0.66 (all fly), 12 → 0.70 (10/12 fly).

  **Practical maximum: 9 drones**, CPU-bound (PX4 + physics); the GPU stays at 2–3%. Recommend `-n 6` for screenshots. Results in `results/gazebo_capacity/`.
- **New `x500_plain` model** (no camera), used automatically with `-p model` (`SENTINEL_UAV_MODEL`). It saves rendering but does not raise the limit.
- **Arming fix** (`flight_commander`): arm only after 3 s of valid PX4 position. Previously the fixed 1 s timer let a UAV take off without a position estimate under load ("blind land"). Occasional estimator failures above 9 drones remain, from PX4 sensor timing under CPU overload.
- **MAPPO long run** at iteration 352 when paused: report value 0.245 (old policy 0.12–0.16). Resumed.

### Paper drafted (2026-10-15)
- **All sections written** in `src/SENTINEL/paper/` (IEEEtran):
  - abstract and title ("Information-Triggered Delivery for Timely Victim Reporting in Infrastructure-Free UAV Search and Rescue");
  - introduction with evidence-led contributions;
  - related work with the comparison table;
  - system model (corroborative fusion added);
  - method (reframed: delivery, sensing, safety layer);
  - implementation (with S1–S6 figure slots);
  - evaluation (10 subsections, all tables `\input` from generated files);
  - discussion and limitations; conclusion.
- **Generated LaTeX tables:** E2, E3 (urban, severe), E1 (200/400 m), E4, E6c, E7, E11, perception.
- **Self-checks:**
  - language lint 0 issues;
  - every number cross-checked against results (2 numeric errors and 1 overclaim in my own draft were caught and fixed);
  - all `\cite` keys in the bib; all `\ref` labels exist.
- **Reference audit:** 23 references, all verified (DBLP, publisher, arXiv, library records). Unverifiable placeholders were dropped (no matching "comm-aware PSO" paper exists, so it was replaced by verified DAPSO and MPSO).
  - Found 2 more v1 citation errors: HIT-UAV authors (Suo et al., not "S. Li et al."), and the MARL paper authors (Shi et al., not "Z. Wang et al.").
  - The rejected v1 manuscript is no longer cited; it is referred to as "the earlier SENTINEL design".
- **`response_to_reviewers.md`** filled with the final evidence for all 16 points plus the AE comment.
- **Pending:**
  - LaTeX compile (texlive not installed; needs sudo);
  - screenshots S1–S6 (user);
  - long MAPPO run → possible B5 update (the abstract and limitations will be re-checked then);
  - target venue (page limit).

### Writing phase started + long MAPPO run (2026-10-15, 18:43)
- **Long MAPPO (B5) run, detached** (`scripts/mappo_long.sh`, log `results/mappo_long.log`): 600 iterations × 16 episodes, about 6 h. It is selected on the tuning seeds against the current policy, and if it wins, only the B5 rows of E1–E4 are re-run and the results regenerated. Purpose: answer "was MARL trained enough?" (R2.6) with either a better policy or a convergence curve.
- **`docs/SCREENSHOTS.md`:** the one-command launch plus the 6 required screenshots (S1 world, S2 occlusion, S3 swarm, S4 thermal, S5 YOLO26n pipeline, S6 ROS graph; optional S7 v1 scene), with file names, how to capture each, where it goes in the paper, and a note on the S5 domain gap.
- Next: paper writing (evaluation → introduction → related work → implementation/discussion → abstract and references).

### One-command launcher (2026-10-15)
- **`~/sentinel_ws/run_sentinel_demo.sh`** (symlink to `scripts/sentinel_demo.sh`) starts Gazebo GUI + N PX4 drones, the full stack, rqt_image_view and rqt_graph, prints live status every 10 s, and stops cleanly on Ctrl-C or `-t`.
  - Options: `-n -w -p -m -t --no-crash --headless`.
  - Tested twice with the GUI (127 s and 77 s): everything started and stopped, with 0 leftover processes.
- **Safety bug fixed:** `launch_sitl.sh` cleanup used `pkill -f "gz sim"` / `"px4 -i"`, which killed *any* process whose command line contained those words. This was the cause of several unexplained "exit 144" errors and could kill unrelated user processes. It now kills only its own process group (verified with a decoy command line).
- RUN_GUIDE now starts with the one-command section.

### A4 complete — final results, E8, YOLO26n demo, hand-over package (2026-10-15)
- **A4 re-run complete** (09:49 → 15:21; final SENTINEL tuning: report value 0.2966, no `explore_lam`). All rows complete, no errors.
- **Final headline** (reported within 60 s, 30 seeds, Holm-corrected):
  - SENTINEL is significantly better than **all 7 baselines** in E2 open/suburban/urban, in E3 urban (all 6 failure models), in E1 at 400 m (every N) and at 200 m (N = 10, 50), in E4 (all), E6/E6c (all) and E11.
  - Severe channel: ahead or tied, **no significant losses anywhere**.
  - E10: T_hold = 0 gives +28% (significant) in severe N = 8.
  - E7: corroborative fusion significant (urban N = 8); spring/LMST/projection/info-tension n.s. (a safety property); SA n.s. in the final configuration.
- **E8 SITL vs fast simulator** (same 3 worlds, 2 methods): reporting within ±0.07, PDR within 0.04, connectivity within 0.07; SENTINEL coverage higher in SITL (0.99 vs 0.83). `results/e8_sitl/e8_comparison.csv`.
- **YOLO26n in-the-loop demo** (`results/demo_yolo26n`):
  - 145 annotated frames; TDR 0.8, but **30 false confirmed entries**;
  - inference 9–17 ms in the loop (3 detectors in PyTorch FP32 alongside Gazebo and PX4);
  - the frames show a **sim-to-real domain gap** (false boxes on rubble edges, missed mannequins), so they are used as a pipeline illustration only.
- **`make_results.py` extended** with the perception, E8 and tuning-sensitivity sections; `docs/RESULTS.md` regenerated.
- **Hand-over package:** `~/sentinel_ws/handover/` (README with headline numbers and caveats; per-experiment CSVs, logs, E8 runs, perception, demo frames, figures, LaTeX tables, configs, docs).

### Final runs, benchmark, E8 → protocol correction A4 (2026-10-15)
- **All 11,346 runs completed** (A3 chain, 23:22 → 05:06), no errors.
- **Perception benchmark** (`results/perception/perception.md`; RTX 3060, 640 px, 300 frames). Accuracy is from the checkpoint validation metrics.

  | Model | Params | GFLOPs | mAP50 | mAP50-95 | P | R | TRT FP16 end-to-end | PyTorch FP32 |
  |---|---|---|---|---|---|---|---|---|
  | **YOLO26n** | 2.53 M | 6.03 | 0.948 | 0.533 | 0.932 | 0.900 | 1.81 ms (553 FPS) | 3.72 ms |
  | YOLOv8n | 3.14 M | 8.75 | 0.947 | 0.520 | 0.909 | 0.913 | 1.73 ms (579 FPS) | 3.32 ms |

  YOLO26n is 19% smaller, needs 31% fewer GFLOPs and has higher mAP50-95 and precision. **There is no speed difference on the desktop GPU**, so no speed claim. The TensorRT engines were built locally from the ONNX.
- **E8 SITL** (6 runs: 3 worlds × {SENTINEL, B0}, 300 s sim, PX4 + Gazebo + ROS 2, model perception): the pipeline is validated. TDR 0.5–0.7, coverage 0.91–0.99, PDR 0.74–0.85. These runs used the A3 tuning, so they are being re-run (moved to `e8_sitl_A3explore`).
- **Results with the A3 tuning** (kept for transparency in `summary_A3explore.parquet`):
  - Big significant wins in dense swarms (E1, N ≥ 10, against all 7 baselines; E11 0.807 vs 0.720), in urban failures and in E6/E6c (including realistic false alarms at 0.55 confidence: 0.60 vs 0.50).
  - **But significant losses in the severe channel** (E3 severe: 0.37 vs lawnmower 0.50).
  - Cause: the re-tune switched on `explore_lam` (soft tether while exploring, 0.155) plus a strong spring (λ0 = 2.39), so severe-channel coverage fell from 0.94 to 0.66. `explore_lam` was *not* part of the design (round-2 decision) but had been left in the tuning space: **my protocol mistake.**
- **User decision → Amendment A4:** remove `explore_lam` from the tuning space, re-tune SENTINEL (30 trials), re-run the SENTINEL rows and E8 SITL, regenerate the results. Running detached: `scripts/rerun_A4.sh`, log `results/rerun_A4.log`. The paper will report the tuning sensitivity.

### Re-run interrupted and resumed (2026-10-14, 23:22)
- The A3 chain was killed when the previous Claude session ended (about 4 minutes into E11; no rows lost).
- Already done before the interruption: purge (backups `summary_preA3.parquet`) and the **final SENTINEL re-tune** (30 trials, best report value 0.284; `configs/tuned.yaml`; pre-A3 values in `tuned_sentinel_preA3.yaml`).
- **Resumed detached** (`nohup setsid scripts/resume_experiments.sh`; log `results/resume_experiments.log`), so it now survives session or VS Code closure. It only computes missing rows; no re-tune.

### Amendment A3: corroborative fusion → protocol re-run (2026-10-14)
- **E11 (30 seeds, 200 m, N = 50), within-60 s reports:**

  | | SENTINEL + corroborative fusion | B0 | No fusion | SENTINEL (naive fusion) |
  |---|---|---|---|---|
  | Within 60 s | **0.823*** | 0.720 | 0.720 | 0.667 |

  0 false entries with corroborative fusion. The dense-swarm weakness becomes a significant win.
- The earlier "no effect" test of the corroborative rule was **invalid**: the rule was a module global that leaked across pooled worker processes. It is now a per-run config field.
- **Adopted as the final design (Amendment A3).** `abl_naive_fusion` is kept as the ablation.
- **Protocol (fairness), running in `scripts/rerun_after_A3.sh`:**
  1. SENTINEL-family rows are purged (backed up as `summary_preA3.parquet`).
  2. SENTINEL is re-tuned from scratch (30 trials; the old values are kept in `tuned_sentinel_preA3.yaml`).
  3. All experiments are re-run (only the missing rows are computed), including E6c, E11 and E8-fastsim.

  Baseline rows are unchanged and valid.
- **Hand-over prep:** `docs/RUN_GUIDE.md` (GUI run, screenshot list S1–S6, reproduction commands). `thermal_vision` now publishes `sentinel/thermal_annotated` with YOLO26n boxes, colour-coded green ≥ 0.70, orange SA band, red < 0.40, for the S5 figure.

### Karthik's models received → YOLO26n deployed (2026-10-14)
- `assets/exported_models/`: YOLO26n and YOLOv8n (each .pt, .onnx and .engine), plus `model_comparison_chart.png` and `ledger_utils.py`.
- **Same training protocol for both models** (from the checkpoints): `hit_uav.yaml`, 100 epochs, AdamW, imgsz 640, batch 16, seed 0, ultralytics 8.4.176, single class `victim`. Validation metrics recorded in the checkpoints:

  | Model | P | R | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | **YOLO26n** | **0.932** | 0.900 | **0.948** | **0.533** |
  | YOLOv8n | 0.909 | **0.913** | 0.947 | 0.520 |

- **YOLO26n is now the deployed model**: the default in `thermal_vision`, `sentinel_ai/models/yolo26n_thermal_best.pt`. The old v1 checkpoint is kept as `yolov8n_v1_best.pt`.
- **Caveats:**
  - These are validation metrics from the training run. A test-split measurement needs the dataset (`bench_detector.py --data`).
  - The YOLOv5n/YOLO11n bars in the chart have no weights in `assets/`, so they are not reproducible and stay out of the paper unless the weights are added.
  - The `.engine` files are GPU/TensorRT-specific, so TensorRT (tensorrt-cu12) was installed in the venv and a local engine is built from the ONNX for the latency benchmark.
  - `ledger_utils.py` (2 m merge, max confidence) is superseded by the SENTINEL ledger (gating, consolidation, fusion, replication).
- **Benchmark script rewritten**: `--data` is optional; it reports the pre/inference/post split, all four backends, and writes exports to the results folder so `assets/` is never modified.
- **E8 SITL cross-validation ready:**
  - new `sim_detector` node: the same calibrated detector model as the fast simulator, fed with Gazebo ground truth;
  - launch switch `perception:=model|yolo`;
  - ROS agents now load `tuned.yaml` (same inheritance rules as the simulator);
  - worlds `sentinel_g100_s{0,1,2}`, `configs/experiments/e8_fastsim.yaml`, `scripts/run_e8_sitl.sh` (process-group cleanup).
- Running: E11 with `sentinel_corrob` (the fusion rule is now a per-run config; the earlier "no effect" test was invalid because the rule leaked across pooled worker processes), then E6c.

### Dense-swarm deficit — diagnosis (2026-10-14)
- Follow-ups done: E9/E10 re-run with the split loss metric; B5 runs for E1–E4 complete.
  - **E9:** failure-induced loss 1.2–1.5%, association merge ~11% (dense 25 m map).
  - **E10:** shorter T_hold → lowest failure-induced loss in severe (T_hold = 0: 2.1% vs 3.5% for never deliver at N = 4; 0.8% vs 1.7% at N = 8). This is the direction Proposition 3 predicts; the magnitudes are small.
- **Quick diagnosis** (200 m, N = 50, 6–8 seeds, *indicative only*):
  1. Relay tether? **No.** full 0.633 / tether off 0.640 / never deliver 0.633.
  2. Exploration parameters? **No.** B0 + SENTINEL params 0.700; SENTINEL + B0 params 0.627.
  3. Fusion? A first signal: no fusion 0.680/0.665 vs 0.633.
     - A "corroborative" fusion rule (occluded views cannot veto) was implemented and tested, and it gave **identical** results to naive Bayes in every cell. The hypothesis is not supported.
     - **Default kept at naive Bayes**, so the finished experiments stay consistent; the corroborative rule stays available behind a flag.
     - The small-sample signals (SE ≈ 0.02) are not trusted.
- **E11 (running):** a proper 30-seed paired test in the dense cell (200 m, N = 50): sentinel vs abl_no_sa vs abl_no_fusion vs abl_no_sa_no_fusion vs B0.
- **Lesson:** do not change the design on 6-seed differences; diagnose with 30 seeds.

### Final suite completed — E1, E3, E4, E6, E6b analysed (2026-10-14)
- All 9 experiments complete: 9,450 runs, no errors. E9/E10 (old loss metric) are being re-run; B5 runs for E1–E4 are in progress. **Framing decision A2 (evidence-led)** recorded in REWORK_PLAN.md.
- **E3: failure models.**
  - **urban:** SENTINEL is *significantly* best on reported-within-60s and total reported under **all six** failure models. Poisson: 0.424 vs 0.356 best baseline, +19%.
  - **severe:** SENTINEL is best or tied; the lawnmower is competitive on on-time reporting (n.s.).
  - **Information loss is rare for all methods** (0% in most cells, ≤ 8.5% under high-rate Poisson), because gossip replication on contact protects data. Do not build the story on data loss (consistent with A2).
- **E4: moving obstacles.** SENTINEL is still best (0.52–0.53 within 60 s). Against B2/B3 the margin becomes n.s. with 5–15 moving obstacles; it stays significant against B1/B4/B6.
- **E6: detection uncertainty.** SENTINEL beats B0/B6 significantly in almost all cells. The SA/fusion ablations are n.s. here.
  - ⚠️ **Model weakness found:** more noise gives *more* reports (σ = 0.25 → 0.68 vs σ = 0.04 → 0.50) with **zero false confirmations**. The false-positive confidence (0.22 ± 0.08) never reaches the 0.6–0.7 thresholds, so noise only helps true victims. This is too optimistic: in Gazebo, best.pt scored warm debris at **0.66**.
  - Fix: **E6c** (new config) uses hot clutter with fp_mu ∈ {0.22, 0.40, 0.55}, noise coupled into false positives (flag default off, so E1–E4 are unaffected). The E6 results will be presented together with E6c.
- **E6b: thresholds.** Lower thresholds report more (0.3/0.6: 0.645 within 60 s vs 0.4/0.7: 0.559) with no false confirmations, for the same reason; to be interpreted with E6c.
- **E1: scalability — regime boundary (must be reported).**

  | Map | N = 3 | N = 5 | N = 10 | N = 20 | N = 50 |
  |---|---|---|---|---|---|
  | 400 m: SENTINEL / best baseline (within 60 s) | **0.199** / 0.153* | **0.341** / 0.313 | **0.597** / 0.535* | 0.639 / 0.647 | 0.641 / **0.712*** |
  | 200 m | **0.536** / 0.459* | 0.601 / 0.597 | 0.632 / **0.664*** | 0.643 / **0.725*** | 0.667 / **0.735*** |

  SENTINEL wins when the swarm is sparse relative to the area and is **significantly worse in dense swarms**.
  - Per-agent compute grows with neighbourhood density (SENTINEL 2.1 → 6.0 ms at 200 m; B2 2.9 → 20.6 ms). The scalability claim must be "cost depends on neighbourhood size k, not N directly".
  - Diagnosis running: hypothesis = the relay tether freezes the entire connected component (the whole dense swarm) after every delivery.

### First final results (30 seeds per cell; mean [95% CI]; * = Holm-corrected paired Wilcoxon p < 0.05 vs SENTINEL)
- **B5 MAPPO selection** (tuning seeds only, as for every method): v1 policy 0.146 vs v2 0.119 → v1 used (`configs/mappo_selection.md`). Its B5 runs are pending.
- **E2: channel profiles** (N = 8, 400 m, Poisson failures). SENTINEL is best on victims reported and report value in **all four profiles**; most differences are significant.

  | Profile | SENTINEL reported | Best baseline reported | SENTINEL within 60 s | Best baseline within 60 s |
  |---|---|---|---|---|
  | open | 0.536 | 0.501 (B4) | 0.549 | 0.501 |
  | suburban | 0.536 | 0.461* (B3) | 0.529 | 0.456 |
  | urban-rubble | 0.539 | 0.487* (B3) | 0.529 | 0.475 |
  | severe | 0.517 | 0.457* (B2) | 0.424 | 0.377 |

  SENTINEL also has the highest coverage (0.86–0.90 vs ≤ 0.82). Every method was tuned with the same budget, but part of the gap comes from the tuned exploration parameters, SA and fusion (see E7).
- **E7: ablations (honest finding).** Removing the outage spring, the LMST set, the information tension or the projection gives **no significant change**. Removing SA is significant (severe N = 8: report value 0.287* vs 0.320). Strict mode is far worse everywhere (0.03–0.10 reported).
  ⇒ The connectivity machinery is active only during short delivery tethers, so its measured effect on totals is small. It must be presented as a **safety property** (Proposition 1, link violations ≈ 0.2%), **not** as the source of the gains.
- **E10: T_hold sweep.** Report value (τ = 300 s) is barely sensitive to T_hold. With the **deadline metric (reported within 60 s)**, short T_hold beats never-delivering *significantly*:
  - severe N = 8: 0.43–0.45* vs 0.35 (+24–28%);
  - severe N = 4: 0.21* vs 0.165;
  - urban N = 8: 0.557* (T_hold = 15) vs 0.50.

  The p90 latency roughly halves in severe (35–58 s vs 92–103 s), and total reports do not drop significantly.
  ⇒ **The contribution the evidence supports: information-triggered delivery significantly increases *timely* reporting (+11–28% within 60 s) at no significant cost in total discovery, plus collaborative SA sensing.** T_hold follows the mission deadline (design rule), and the tuned T_hold = 115 s targets the τ = 300 s objective.
  - E10's loss columns use the old metric; it will be re-run.

### Experiments started (final protocol), 2026-10-10 evening
- **Performance:**
  - Per-agent planning made independent of N (windowed map operations, FFT footprint convolution, cached residual totals, versioned G-counter merge, scipy LMST with identical tie-breaking). 394 tests pass, including equivalence tests against the reference implementations.
  - **BLAS thread oversubscription found and fixed:** every worker process opened a 20-thread BLAS pool, giving a load average of about 130. This had slowed all earlier batches. All entry points now pin BLAS to 1 thread.
  - Timing (600 s mission): N = 8 ≈ 13 s, N = 50 ≈ 200 s. **Per-agent compute is 2.15 ms at N = 8 and 2.71 ms at N = 50**, i.e. essentially flat: evidence for R2.8.
- **MAPPO v1** (obs 26) plateaued at a report value of 0.12–0.16, against about 0.29 for the tuned heuristics. To avoid a straw-man baseline, **MAPPO v2** adds the same frontier-goal direction the PSO methods use (obs 28) and trains for 250 iterations (`configs/mappo_policy_v1_obs26.npz` kept).
- **Suite running:** `scripts/run_experiments.sh 10 --skip-methods B5_mappo` (B5 is added after its training), 30 seeds per cell; results go to `results/e*/`.
- **E9 (faithful v1 re-run)** — 25×25 m, 3 UAVs, no responder, crash at 100 s; 30 seeds; done:

  | Method | TDR | NRR | Connected | Info loss | Dup/false entries |
  |---|---|---|---|---|---|
  | SENTINEL | 0.63 [0.58, 0.68] | 1.00 | 1.00 | 0.13 | 0/0 |
  | SENTINEL-strict | 0.63 | 1.00 | 1.00 | 0.13 | 0/0 |
  | B1 (v1) | 0.65 | 1.00 | 1.00 | 0.13 | 0.03 dup |
  | B0 PSO | 0.64 | 1.00 | 1.00 | 0.12 | 0.03 false |

  ⇒ **In v1's own scenario every method is permanently connected (the map is far smaller than the radio range). The v1 claims (NRR 66% vs 100%, TDR 60/70/90%) cannot arise there.** Measured TDR is about 0.64, not 0.90. This justifies the larger, stressed scenarios.
  - ⚠️ The ~12% "information loss" was diagnosed: **none of it is failure-induced.** Every "lost" victim had a neighbouring victim within 4.1–5.1 m (inside the 5 m gate) and still-alive carriers; the two entries merged.
    - The metric is now split: `info_loss` (every carrier failed; this is what Proposition 3 bounds) and `assoc_loss` (association merge).
    - Re-check: E9 seeds 0/4/5 give info_loss 0, assoc_loss 0.11–0.13.
    - E9 and E10 ran with the old metric and **will be re-run**; later experiments use the new one.
    - Report the dense-cluster association limit (victims < ~5 m apart) as a limitation.

### Phase 8 — paper draft started + Proposition 1 measured, 2026-10-10
- `src/SENTINEL/paper/` (IEEEtran):
  - `main.tex` skeleton;
  - **`sections/system_model.tex`**: network, channel (SINR with path loss, obstructions, Gudmundson shadowing, Rician fading, interference; outage in closed form + Gauss–Hermite), sensing, ledger, failures, and the problem (time-discounted report value J);
  - **`sections/method.tex`**: mode switching, EM-PSO fitness with the outage spring, LMST, midpoint-ball projection, relay tether, SA, complexity, Algorithm 1.
- **Formal results drafted:**
  - Lemma 1: the nearest-neighbour rule does not imply connectivity (answers R2.2).
  - Lemma 2: LMST ⊇ MST.
  - **Proposition 1:** one-step link preservation with staleness + obstruction inflation (full proof).
  - Corollary: connectivity during delivery.
  - **Proposition 2:** union-bound disconnection probability.
  - **Proposition 3:** information-exposure bound P(loss) ≤ (λ_f·W)^k with W = T_h + κD/v + τ_c.
- **Self-review caught an overclaim:** "hovering is always admissible" was false (inflated obstruction counts can shrink radii). Proposition 1 is now stated for feasible steps, with an explicit remark, and the violation rate is measured.
- **Terminology:** "Data-Loss Paradox" is replaced by "discovery–delivery trade-off" (R2.10).
- **New simulator metrics:** `link_violation_rate` (tethered, mutually preserved links that were reliable before a step and are not after it) and `link_reliable_start_frac`. First check (400 s, 1 seed):

  | | Violation rate |
  |---|---|
  | strict, urban | 0.20% over 3076 link-steps |
  | strict, severe | 0.25% over 4314 |
  | intermittent, urban | 0/51 |
  | intermittent, severe | 1/7 |

  The residual comes from infeasible steps (hover fallback). An earlier, wrongly defined version of this metric gave 13–52% because it counted links that were already unreliable at the start; it was fixed before any use.
- MAPPO training: report rate 0.7% → 18% by iteration 7 (continuing).

### Phase 8 groundwork — novelty check and reference verification, 2026-10-10
- See [related_work_notes.md](related_work_notes.md). Verified so far: Kantaros & Zavlanos (TAC 2017), Sabattini et al. (IJRR 2013), the YOLO26 release (Jan 2026, ultralytics v8.4.0).
- **Positioning.** The closest strands are continuous connectivity maintenance, intermittent connectivity (fixed or recurring meetings), and data ferrying / DTN / AoI (*known* data sources). SENTINEL's niche is *event-triggered delivery of information discovered online by the search*, with a loss bound and channel-aware link preservation during delivery. No paper found combines these.
- Flagged for audit: the v1 draft attributes HIT-UAV to "S. Li et al."; the dataset paper's author list must be checked.

### Phase 3/4 — first live SITL flights, 2026-10-10
- `launch_sitl.sh 3 sentinel_g100_s0`:
  - Gazebo plus 3 PX4 instances spawn `x500_thermal_0..2` at the generated spawn poses;
  - thermal camera topics publish (`/world/.../thermal_link/sensor/thermal_camera/image`);
  - PX4 topics appear under `/px4_i/fmu/...`, with versioned names confirmed live (`vehicle_local_position_v1`, `vehicle_status_v4`).
- Bug fixed: `set -u` in the launcher clashed with PX4's `gz_env.sh`.
- **Flight test 1** (`flight_commander` + `emps_agent`, no comms): all 3 UAVs took off to 30 m in offboard mode and flew decentralised EM-PSO exploration for 150 s.
- Bug fixed: `comm_emulator` declared `channel_profile` twice and crashed. Also cleaned up shutdown (no more ExternalShutdownException tracebacks), and the ROS agents now receive the responder id/position, the no-fly map and `t_hold_s`.
- **Flight test 2** (+ `comm_emulator` + `metrics_logger`), 150 s:
  - coverage 0.94 of the 100 m map;
  - PDR 0.77 through the channel model;
  - mean beacon payload **4.1 KB** (max 20 KB with map gossip), which is a bandwidth figure for the paper;
  - NRR 1.0, λ2 mean 1.14.

  TDR is 0 because perception is not in the loop until `ros_gz` is installed.

### Phase 3 — Gazebo assets (`sentinel_gazebo`), 2026-10-10
- **`models/x500_thermal`:** the PX4 x500 plus a nadir thermal camera.
  - 640×512 (HIT-UAV resolution), 50° HFOV, so the footprint at 30 m is about 12 m, the same as in the simulator.
  - 10 Hz, white-hot L8 mapping (270 K → 0, 330 K → 255), matching the HIT-UAV convention, so v1's image inversion is no longer needed.
- **`models/rescue_randy_thermal`:** Rescue Randy (Gazebo Fuel, CC-BY 4.0) with its heat-signature texture at a maximum of 310.15 K, using gz-sim 8 plugin names.
- **`models/warm_debris`:** a 316 K slab that produces thermal false positives.
- **`scripts/make_world.py`:** builds the SDF and a ground-truth JSON **from the same generator and seed as the fast simulator**.
  - Obstacles, targets and obstruction counts are verified identical, so the E8 cross-validation compares like with like.
  - Occluded victims get an overhanging slab on their blocking side; the buried victim gets a full slab (as in v1).
  - Thermal scene: ground 293 K, buildings 291 K, rubble 285 K, slabs 289 K.
  - Generated worlds: `sentinel_d25_s0` (the v1 rebuild: 10 victims, 1 buried) and `sentinel_g100_s0`.
- **`scripts/launch_sitl.sh N WORLD [--gui]`:** starts Gazebo (headless rendering), the Micro XRCE-DDS agent and N PX4 instances.
  - **The PX4 tree is not modified.** Each PX4 instance runs from its own directory, so PX4 does not re-source its `gz_env.sh` and `PX4_GZ_MODELS` points at our models.
  - Uses airframe 4014 with `PX4_SIM_MODEL=gz_x500_thermal` and namespaces `/px4_i`; the offboard-without-RC parameters are set through `PX4_PARAM_*`.
  - The world loads headless with no errors.
- **px4_msgs:** byte-identical to PX4 `main` for the six messages used. The versioned topics are `vehicle_local_position_v1` and `vehicle_status_v4` (to be confirmed live).

### Phase 4 — ROS 2 integration (`sentinel_ai` v2 + `sentinel_interfaces`), 2026-10-10
- **Messages:** `TargetReport` (world-frame detection), `Beacon` (serialised `sentinel_core` beacon), `LedgerSnapshot`, `DetectorStats`.
- **Nodes:**

  | Node | What it does |
  |---|---|
  | `flight_commander` | PX4 offboard control; publishes the **real world-frame pose** (fixes v1's always-(0, 0) telemetry); force-disarms on a kill message |
  | `thermal_vision` | Model-agnostic detector: `model_path` takes any Ultralytics .pt, .onnx or .engine file (Karthik's model drops in here). Geolocates each detection with the camera model, pose and heading, publishes it as a world-frame `TargetReport`, and logs p50/p95 latency |
  | `emps_agent` | Wraps the **same `SentinelAgent`** as the simulator. Decentralised: there is no central coordinator |
  | `comm_emulator` | Routes all inter-UAV traffic through the `sentinel_core` channel model (obstructions taken from the world geometry). Hosts the responder radio and logs PDR and payload size (a bandwidth metric) |
  | `chaos_monkey` | Configurable crash, link-cut and brownout schedules |
  | `metrics_logger` | The same metric definitions as the simulator, written to `series.csv` and `summary.json` |

- **Launch:** `ros2 launch sentinel_ai swarm_launch.py n_uav:=3 world:=sentinel_d25_s0 method:=sentinel model_path:=... failures:='[...]'`
- **Removed** (still in git history): the v1 central `swarm_coordinator.py`, `fitness_function.py` (now part of `sentinel_core`) and the duplicate unzipped `best/` folder. `best.pt` moved to `sentinel_ai/models/`.
- **Build:** `colcon build` passes for all five packages. To run: source ROS, then `install/setup.bash`, then activate `~/sentinel_venv`.
- **Known approximation:** PX4's position-tracking lag is outside the projection proof. It will be measured in SITL.

---

## Open issues / next steps
1. ✅ Core design decided (A1) and implemented; pilot 4 is promising (see above).
2. Next: deadline-based metrics (reports within 60/120 s, time-discounted TDR); finalise the scenario grid; Optuna tuning of all methods (equal budget); MAPPO (B5).
3. ✅ Building no-fly zones (done).
2. Decide the scenario set for E1–E5 from the pilot results, then tune every method with the same Optuna budget.
3. MAPPO baseline (B5); plotting scripts.
4. Phases 3–5 (Gazebo, ROS, perception harness).
5. **Waiting on you:** `sudo apt install ros-humble-ros-gzharmonic` (camera and clock bridge into ROS).
6. **Waiting on Karthik:** the HIT-UAV dataset as used for training (`hit_uav.yaml` + images), needed for the Phase 5 benchmark (`scripts/bench_detector.py`).
7. Tuning in progress (`configs/tuned.yaml`; SENTINEL done: T_hold ≈ 115 s, λ0 ≈ 0.45, β ≈ 0.75, sector bias ≈ 2.8). Then MAPPO training, then E1–E10 (configs ready in `sentinel_sim/configs/experiments/`).

### 2026-10-16 — Paper layout pass
- Compiled with `latexmk -pdf`: 11 pages, no undefined citations or references.
- Fixed every overfull box:
  - generated tables are wrapped in `adjustbox{max width=\columnwidth}`;
  - the related-work table is wrapped in `adjustbox{max width=\textwidth}`;
  - the fitness equation (5) is split with `multline`;
  - the screenshot placeholders are sized to the column.
- Fixed a bug in `make_results.py`: `tex()` escaped `_` inside `$...$`, so the E6c headers printed as "μ\_fp". It now escapes text only, and the headers read $\mu_{\mathrm{fp}}$, $r$.
- Regenerated the tables. `RESULTS.md` numbers are unchanged (diff checked).
- What remains is underfull lines only (cosmetic: one algorithm line, one bib entry).

### 2026-10-16 — Handover to Karthik
- Wrote `karthik_handover.md` (workspace root; a copy is in `handover/`). It covers:
  - background;
  - the new framing;
  - all 16 reviewer comments and their fixes;
  - code improvements;
  - results and caveats;
  - the one-command run;
  - the screenshots S1–S7 with their figure mapping;
  - the paper files, build and regeneration commands;
  - restructuring rules;
  - open decisions (venue, git push).
- Karthik takes over the screenshots and the paper restructuring.
- Fixed a stale note in `paper/README.md`: the lint script already exists. `lint_language.py` reports 0 issues.

### 2026-10-16 — Caption and screenshot-mapping fixes
- `implementation.tex`: the Fig. 3 and Fig. 4 captions now say "six UAVs" (was "three"), to match the 6-drone demo default.
- `docs/SCREENSHOTS.md`: the screenshot table now gives the actual figure numbers (Figs. 1–4, all in Section V) instead of the wrong §III/§IV references.
- `karthik_handover.md` updated to match. The paper recompiles: 11 pages, 0 overfull boxes.

### 2026-10-16 — Commit
- Committed all v2 work on branch `revision-v2` as commit `09bcc55` (550 files).
- The commit adds:
  - the code and the paper;
  - the docs: plan, progress, results and guides;
  - `karthik_handover.md`.
- `.gitignore` now excludes the LaTeX build files.
- The rejection-email PDF and the old draft PDF are not committed.
- The push to `origin` failed because this machine has no GitHub credentials. The user has to push manually.

### 2026-10-21 — Screenshot S1 placed
- Placed the user's oblique Gazebo view as `paper/figures/S1_world_overview.png`.
  - I cropped the sky band (y 0–190) off the top; the uncropped original is kept as `S1_world_overview_uncropped.png`.
  - The user also took three near-top-down attempts. They were rejected: the toolbar was visible, building shadows and perspective covered the map, and the victims were too small to see.
- The Fig. 1 caption is changed from "top view" to "oblique view" and now lists what is visible, including the six UAVs at the launch point.
- The paper rebuilds at 11 pages with 0 overfull boxes.
- S1 replaced by the user's top-down retake, which has the whole map with all buildings in frame. At print width it shows the layout better than the oblique shot. The oblique shot is kept as `figures/S1_alt_oblique.png`.
- The Fig. 1 caption is back to "top view". It now says victims and UAVs are too small at this scale and points to Figs. 2 and 4.
- The paper is still 11 pages, with 0 overfull boxes and 0 undefined references.

### 2026-10-21 — Gazebo scene fix: standing victims poked through slabs
- **Found from the user's S2 screenshot.** The Rescue Randy mesh is 1.82 m tall and standing, but the slabs sat at 1.0–1.3 m. The head and shoulders of every covered victim, including the "fully buried" ones, stuck out through the slab and were visible from above. This contradicted the occlusion described in the paper.
- **Fix in `sentinel_gazebo/scripts/make_world.py`:**
  - Victims now lie flat (roll 90°, 0.36 m high), with the torso centred on the ground-truth position (checked: 0.00 m error).
  - Slabs rest on corner posts with the underside at 0.45 m. Any post that would touch a victim is left out. Result: 58 slabs, 161 posts, 0 posts touching a body, every slab on at least 2 posts.
- **Regenerated all 5 worlds** (`d25_s0`, `g100_s0/1/2`, `g200_s0`). All SDFs pass `gz sdf -k`. The ground-truth JSONs are **byte-identical** to before, so the positions, occlusion values and obstacles are unchanged.
- **Results unaffected.** The fast-simulator experiments and the E8 cross-check use occlusion from the ground truth, not from the rendered meshes.
- **Demo frames now outdated.** The 145 YOLO frames in `handover/demo_yolo26n/` show the old standing victims. Take S5 from a fresh run.
- **Backup.** The old worlds are in the scratchpad (`worlds_before/`).

### 2026-10-22 — Screenshot S2 placed
- The user took six ground-level shots of `victim_5` and `victim_14` in the fixed scene (victims lying, slabs on posts).
- Chose the `victim_14` close-up: the torso is under the slab and the legs are exposed. Cropped it to the subject and checked it is readable at print size. It is saved as `paper/figures/S2_victims_occlusion.png`.
- The wide shot showing a covered victim, an open victim and distant slabs is kept as a spare, `S2_alt_wide.png`.
- The Fig. 2 caption now describes what is shown: one partly covered victim, torso hidden from above, legs exposed.
- The paper is still 11 pages with 0 overfull boxes.
- 2026-10-22: two overhead views around victim_14 (open, partly and nearly fully covered victims as a UAV sees them) are kept as spares, `S2_alt_above_oblique.png` and `S2_alt_above_top.png`. They are not used in the paper.

### 2026-10-22 — Screenshot S3 placed
- **S3:** a crop of the bright overhead shot with two UAVs over victims, slabs and rubble, saved as `paper/figures/S3_swarm_in_flight.png`. It was checked for readability at print size.
- **Spares:** `S3_alt_closeup_shadow.png` (detailed drones, but too dark when shrunk) and `S3_alt_cluster.png` (five UAVs, tiny).
- **Fig. 4 caption:** "two of the six UAVs searching at 30 m above victims, slabs and rubble".
- **Checked:** the user could not see `x500_thermal_0/1` in the Entity Tree. `gz model --list` shows all six models, and the PX4 logs show all six took off. They were only further up in the list.

### 2026-10-22 — Screenshots S5 and S6
- **S5.** I chose the user's saved frame where YOLO26n puts an orange box (0.40, SA band) on the rendered mannequin and a false box (0.52) on a building edge. It is cropped to the top-left region and saved as `paper/figures/S5_yolo26n_pipeline.png`. The spare with red and orange boxes is `S5_alt_red_orange.png`. Box colours, from `thermal_vision.py`: green ≥0.70, orange 0.40–0.70, red <0.40. The Fig. 4 caption now describes exactly that: an SA-band detection, a false box, and the rendering gap.
- **S6.** The user's rqt_graph capture is correct, but with 6 UAVs it is unreadable at column width (about 100 labelled edges). It is kept as `S6_ros_graph.png`, with the window version as `S6_alt_window.png`.
  - Fig. 3 is now a vector diagram, `figures/ros_graph.dot` → `ros_graph.pdf`, transcribed from `ros2 node info` on the running 6-UAV demo. It shows one UAV namespace (thermal_bridge, thermal_vision, emps_agent, flight_commander) and the shared comm_emulator, chaos_monkey and metrics_logger.
  - The caption says the graph was transcribed from the running system.
- **Drone check.** The user again thought only 4 drones were up. All 6 PX4 logs show takeoff with no landing or failsafe, and the sim was still running.
- **Status.** The paper is 11 pages with 0 overfull boxes and 0 undefined references. Only S4 (needs a `-p model` run) and optional S7 remain.

### 2026-10-22 — S4 instructions corrected, helper added
- **Bug in my instructions.** `SCREENSHOTS.md` and the handover said to take S4 with `-p model`. In that mode the demo spawns `x500_plain` (no camera) and starts no image bridge, so the thermal topic would be empty.
- **New `sentinel_ws/scripts/save_thermal_frames.py`,** with the wrapper `~/sentinel_ws/save_thermal_frames.sh`. During a normal (YOLO) run it subscribes to the raw L16 thermal topic of UAV i. It applies the same 270–330 K white-hot conversion as `thermal_vision` and saves box-free PNGs every 2 s of sim time to `results/thermal_frames/uav<i>/`.
- Syntax-checked and the startup tested (it exits cleanly with no simulation running). It is not yet tested against a live camera.
- Both copies of `SCREENSHOTS.md` and of the handover have the corrected S4 instructions.

### 2026-10-22 — Screenshot S4 placed; all required screenshots done
- First live run of `save_thermal_frames.sh` (UAV 0): 40 box-free white-hot frames saved, all 640×512, grey levels 64–195.
- **S4:** `frame_000127.9.png`, saved as `paper/figures/S4_thermal_frame.png`. It shows two victims in the open, two under slabs with only the legs visible, warm debris (316 K, the brightest object) beside a rubble block, and a building wall. Spare: `S4_alt_frame_129.png`.
- The Fig. 2 caption now describes exactly that content.
- The user's second rqt_graph capture showed only part of the system, because ROS 2 discovery was incomplete while the sim was paused (several nodes missing). It is kept as `S6_alt_partial_discovery.png` and not used.
- **Paper:** 11 pages, no screenshot placeholders left, 0 overfull boxes, 0 undefined references. Only S7 (optional) remains.

### 2026-10-22 — Commit c657c70
- Committed on `revision-v2`: screenshots S1–S6, the Fig. 3 diagram, the victim/slab scene fix with regenerated worlds, the thermal-frame helper (.py + .sh) and updated docs.
- The spare screenshots are not committed.
- The push failed again because this machine has no GitHub credentials. The user has to push from their own terminal. Branch `revision-v2` is 2 commits ahead of `main` (09bcc55, c657c70).
- 2026-10-22: removed the forced line break in the title, which had left "Reporting" alone on the second line. The title now wraps naturally over three lines; still 11 pages.
