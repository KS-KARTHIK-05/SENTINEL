# SENTINEL v2 — Rework Plan (response to IoT-66280-2026 reviews)

> This is the governing plan for the rework. Every task, experiment and paper change must trace back to it. When anything deviates, update this file first.

## 1. Context

**Decision.**
- IEEE IoT-J rejected the paper, and resubmitting there is **not allowed**. The reworked paper goes to a **new venue** (see §9).
- The Associate Editor's summary listed unclear novelty and contributions, a vague system model, and unconvincing experimental results.

**Reviewer comments (16).**

| ID | Theme | Comment (short) |
|---|---|---|
| R1.1 | Novelty | The parts read as an existing detector, a distance penalty and a threshold-triggered lateral move. Is that enough? |
| R1.2 | Novelty | Distributed swarms are well studied. A detailed comparison with the closest distributed methods is needed. |
| R1.3 | Structure | The contributions (III-C/D) are too short. The paper reads like an experimental report. |
| R1.4 | Channel | A fixed 50 m threshold: no path loss, fading, interference or packet reliability. |
| R1.5 | Perception | Evaluate YOLO26 for edge deployment. |
| R1.6 | Consistency | 10 targets, 9 detected, yet "15 verified coordinates". |
| R2.1 | Novelty | Signal-Spring is just a penalty term. Distinguish it from topology-maintenance and communication-aware trajectory work. |
| R2.2 | Theory | Pairwise distance does not guarantee global connectivity. A graph-theoretic analysis (λ2) is needed. |
| R2.3 | Claims | "Guarantees total network survival" and "zero data loss" rest on N=3 with one failure. |
| R2.4 | Experiments | More swarm sizes, larger maps, dynamic obstacles, varying channels, multiple failures. |
| R2.5 | Channel | Use a realistic wireless model or justify the abstraction. |
| R2.6 | Baselines | Add communication-aware swarm optimisation, MARL, graph-based coordination and topology-control baselines. |
| R2.7 | Perception | Link perception accuracy to swarm decisions; analyse sensitivity to detection uncertainty. |
| R2.8 | Scalability | The O(N) claim is untested beyond N=3. |
| R2.9 | Related work | Systematic comparison of advantages and limitations. |
| R2.10 | Writing | Informal language ("bulletproof", "brutal reality", "ironclad") and exaggerated claims. |

**Code audit (verified on 2026-10-09).** The repository (`src/SENTINEL/sentinel_ws/src/sentinel_ai`, about 530 lines) **does not implement what the paper describes**:
- **No PSO.** There are no particles, no velocity update and no `Gbest`. `calculate_fitness` is imported but never called.
- **Parameters differ from the paper.** λ is 0.8 in code versus 0.5 in the paper, and the SA offset is 10 m in code versus 15 m.
- **The ledger stores camera-frame pixel error converted to metres,** not world positions. Telemetry is always (0, 0), because `current_x` and `current_y` are never set (`flight_commander.py:132`). Repeated sightings therefore cannot be merged, which is the root cause of R1.6.
- **The coordinator publishes on the same `target_error` topic it subscribes to.**
- **N=3 is hardcoded,** and the chaos test is a hardcoded kill of UAV 1 at t > 100 s.
- **Missing everywhere, including the GitHub repo:** the world SDF, thermal camera, victims, the C++ `sentinel_mobility` package, metrics, plotting, baselines and any latency measurement.
- **The model path is hardcoded** to `/home/nicholas/...`.

**→ Every number in the paper (NRR 66/100, TDR 60/70/90, 4 ms, 1.2 ms on Jetson, 15 coordinates) is re-measured from new code and the old values are dropped.** We don't reuse any number we can't reproduce.

**Environment.**
- **Available:** Ubuntu 22.04, ROS 2 Humble, Gazebo Harmonic 8.15, PX4 `main` (v1.18-dev, SITL built), MicroXRCEAgent v3.0.2, RTX 3060 12 GB (driver supports CUDA 13).
- **Missing:** the `ros_gz` bridge, a thermal-camera model, ultralytics and torch, networkx, the CUDA toolkit and TensorRT.
- **Conflict:** the user-level numpy 2.2.6 breaks the apt OpenCV and cv_bridge.

**User decisions.**
- **Hybrid evaluation:** a fast Python simulator for statistical sweeps, plus Gazebo + PX4 SITL (3–6 UAVs) for validation. Both run the same algorithm library.
- **Scope:** code, results and the paper. The paper is new IEEEtran LaTeX, plus a response matrix.
- **Detector:** Karthik's model is still undecided, so the detector interface stays model-agnostic.

---

## 2. Contributions of the reworked paper (the novelty case)
These are deliberately positioned against the closest prior work, so that R1.1, R1.2 and R2.1 cannot be raised again.

**C1. Information-survivability-aware mobility (the core new idea).**
- Prior connectivity-maintenance work, such as Zavlanos & Pappas (2008), Ji & Egerstedt (2007), Sabattini et al. (2013) and Kantaros & Zavlanos (2016), preserves connectivity as a goal in its own right.
- SENTINEL instead optimises for the survival of **sensed information**. The spring tension is adaptive: λ_i(t) = λ₀·(1 + β·U_i(t)), where U_i is the number of target detections held by UAV i that are **not yet replicated** to a neighbour.
- A UAV carrying unique, unreplicated detections is pulled back towards the mesh until gossip replication finishes. It is then released to explore.
- This turns the "Data-Loss Paradox" into a formal objective: maximise coverage subject to P(loss of a detected target) ≤ δ.

**C2. Channel-aware, connectivity-preserving EM-PSO with a provable guarantee.**
- The spring acts on the **predicted link outage probability**, not on distance. The channel model includes:
  - log-distance path loss;
  - log-normal shadowing;
  - an NLOS penalty taken from LoS ray-casts;
  - Rician or Rayleigh fading;
  - SINR including concurrent transmitters.
- The spring applies to each agent's local **LMST/RNG neighbour set**, and a safety projection runs after the PSO step.
- **Proposition 1 (deterministic):** if G(0) is connected and every LMST edge stays within the reliable range R_ε, then G(t) stays connected for all t, because the LMST contains the MST.
- **Proposition 2 (probabilistic, under fading):** P(G(t) is disconnected) ≤ Σ_{e∈LMST} P_out(e) ≤ (N−1)·ε, by a union bound.
- **Lemma (counter-example):** the v1 nearest-neighbour rule does **not** guarantee connectivity. Two separate clusters can each satisfy it. This answers R2.2 directly.
- **PSO stability:** the update uses Clerc–Kennedy constriction coefficients, so the projected update stays bounded.
- **Complexity:** O(k) per agent per step, where k is the neighbour count, versus O(N²) or O(N³) for global λ2-gradient methods. Measured empirically (C4).

**C3. Uncertainty-aware collaborative sensing (SA).**
- The trigger band uses calibrated detector confidence, and views from several UAVs are combined by log-odds fusion.
- A flanking move is admitted only if it keeps the C2 constraints and the C1 replication constraint satisfied.
- The analysis links perception quality to swarm decisions (R2.7).

**C4. Rigorous, reproducible evaluation.**
- Scale: N up to 50, three map sizes, four channel profiles.
- Faults: k-simultaneous, cascading and brownout failures, plus dynamic obstacles.
- Six baselines.
- Every result is reported with confidence intervals and significance tests.
- The SITL cross-validation and the code are released.

> The YOLO detector is **not** claimed as a contribution. It is an enabling component, and YOLOv8n is compared with YOLO26 and Karthik's model (R1.5).

---

## 3. Target architecture
- **Location:** the SENTINEL repo under `src/SENTINEL/sentinel_ws/src/`, on branch `revision-v2`. The outer colcon workspace `/home/srmist/sentinel_ws` builds it.
- **Commits** happen only when the user asks.

```
sentinel_core/        pure Python, no ROS; the only implementation of the algorithms
  channel.py          path loss, shadowing, NLOS, fading → SINR → PER/outage; profiles
                      (LoS-open, suburban, urban-rubble, severe), parameters from 3GPP TR 36.777 /
                      ITU-R P.1411 with cited justification
  graph.py            comm graph, Laplacian, λ2 and Fiedler vector, components, local LMST/RNG
  fitness.py          coverage gain + detection reward − adaptive link-outage spring (C1+C2);
                      v1 distance-only variant kept for the ablation
  pso.py              decentralised local-best PSO on the comm graph, constriction, safety projection
  sa_trigger.py       calibrated bands (0.40/0.70), ±15 m flank, feasibility check, log-odds fusion
  ledger.py           world-frame ledger, gated association, gossip replication, provenance, U_i
  detector_model.py   stochastic detector: P(detect) and confidence vs slant range, occlusion and view
                      angle, calibrated from real YOLO runs
  failures.py         single, k-simultaneous, cascading, Poisson and gradual-brownout schedules
  metrics.py          see §5
  baselines/          see §4
sentinel_sim/         fast 2.5D simulator: polygon and moving obstacles, LoS ray-casts (channel and
                      occlusion), kinematic UAVs; YAML configs; multiprocess batch runner → parquet;
                      plot scripts (IEEE style, vector PDF)
sentinel_interfaces/  messages: Detection, TargetReport, LedgerEntry, NeighborState, LinkState
sentinel_ai/          ROS 2 nodes, fully parameterised, N set from a launch argument
  thermal_vision      Detector abstraction (Ultralytics .pt/.onnx/.engine); params model_path,
                      imgsz, conf, device; timed pre/infer/post (p50/p95); geolocation using
                      CameraInfo + VehicleLocalPosition/attitude → world-ENU TargetReport
  flight_commander    real VehicleLocalPosition (fixes the telemetry bug); offboard and arm FSM;
                      follows setpoints from emps_agent
  emps_agent          per-UAV decentralised EM-PSO, SA and ledger built on sentinel_core
  comm_emulator       all inter-UAV traffic passes through it; drop/delay from channel.py + world
                      LoS, so the channel model also applies in Gazebo
  chaos_monkey        configurable failure schedules (kill a PX4 instance, cut comms, brownout)
  metrics_logger      CSV + rosbag2: ground truth, ledger, links, λ2
  observer            passive, metrics only; no central coordinator in the control loop
sentinel_gazebo/      worlds disaster_25 (faithful v1 rebuild: 10 victims, one under a slab),
                      disaster_100, disaster_200 (+ moving debris); Rescue Randy + gz Thermal
                      system (310.15 K); x500_thermal (down-facing thermal_camera); PX4 airframe kept
                      as a patch; scripts/launch_sitl.sh N WORLD
docker/               reproducible image (ROS 2 Humble + Gazebo Harmonic + PX4 + venv)
paper/                IEEEtran LaTeX; every figure and table generated by a script from results/;
                      response_to_reviewers.tex
```

---

## 4. Baselines (R2.6). Each is implemented from its cited paper and tuned with the same budget.

| # | Baseline | Family | Reference to cite |
|---|---|---|---|
| B0 | Standard PSO (no comms term) | Swarm search | Kennedy & Eberhart; Clerc & Kennedy |
| B1 | SENTINEL-v1 distance spring (nearest neighbour, 50 m) | Ablation of the old method | our v1 |
| B2 | Soft-constraint communication-aware PSO | Communication-aware optimisation | bandwidth/comm-aware PSO literature (verify) |
| B3 | λ2-gradient decentralised connectivity control + coverage | Graph-based / algebraic | Sabattini et al. 2013; Zavlanos & Pappas 2008 |
| B4 | Potential-field connectivity maintenance (hysteresis) | Topology control | Ji & Egerstedt 2007 |
| B5 | MAPPO with comm-graph observations | MARL | Yu et al. 2022 (MAPPO) |
| B6 | Lawnmower/sector sweep with a static relay chain | Classical SAR practice | — |

- **Fairness:** each method gets the same hyperparameter-search budget (Optuna, same number of trials, tuned on separate seeds).
- **MAPPO budget:** training is capped at about 24 GPU-hours. If it underperforms, that is reported honestly together with the training curves.

## 5. Metrics (defined formally in the paper, which fixes the R1.6 inconsistency)
- **Connectivity:** λ2(t); % of time connected; largest-component fraction; **NRR** (fraction of surviving UAVs still in the main component).
- **Information:**
  - **Target Discovery Rate** = unique true targets confirmed / true targets.
  - **Information loss rate** = confirmed targets lost through failures / confirmed.
  - **Ledger precision**: duplicate and false entries, which is the explicit check on R1.6.
  - Localisation error.
- **Mission:** coverage % over time; time-to-k-th detection; energy proxy (path length and acceleration).
- **Systems:** packet delivery ratio (PDR); recovery time (to reconnection and to re-replication); per-agent compute time vs N.
- **Statistics:**
  - ≥30 seeds per cell, reported as mean ± 95% bootstrap CI.
  - Wilcoxon signed-rank tests with Holm correction against each baseline.
  - Effect sizes (Cliff's δ).

## 6. Phases
Phases 0–6 do not need the new model.

**Phase 0 — Environment (installs software; confirm before running).**
- Create `~/sentinel_venv` with `--system-site-packages`. Install:
  - `numpy<2`;
  - ultralytics (a version with YOLO26);
  - torch with CUDA 12.x wheels;
  - onnx, onnxruntime-gpu;
  - networkx, scipy, pandas, matplotlib;
  - pyyaml, shapely, optuna.
- Set `[build_scripts] executable=/usr/bin/env python3` in each setup.cfg.
- Install `ros-humble-ros-gzharmonic` from the OSRF apt repo.
- Check that `px4_msgs` matches the PX4 `main` message definitions.
- **Verify:** `cv2`, `ultralytics` and `torch.cuda` import cleanly, and `colcon build` passes.

**Phase 1 — `sentinel_core` + pytest.**
- Channel PER curves against the analytic formulas.
- λ2 on known graphs (path, star, complete).
- A property test for Proposition 1 on 10⁴ random motions.
- The Lemma counter-example.
- PSO convergence on a toy field.
- Ledger regression test: 10 targets must never yield more than 10 entries.

**Phase 2 — `sentinel_sim` + baselines B0–B6.** Determinism check: the same seed must produce an identical parquet file.

**Phase 3 — Gazebo assets + SITL bring-up.**
- `disaster_25`, `disaster_100` and `disaster_200` worlds, `x500_thermal`, the airframe patch and `launch_sitl.sh`.
- **Verify:**
  - N=3 and N=6 arm and fly offboard;
  - thermal images arrive in ROS;
  - victims render hot and obstacles render cold.

**Phase 4 — ROS 2 integration.**
- Refactor the nodes and add `comm_emulator`, `chaos_monkey` and `metrics_logger`.
- **Verify:**
  - end-to-end N=3 on `disaster_25`;
  - geolocation error < 2 m;
  - number of ledger entries ≤ number of true targets;
  - the metrics CSV is written.

**Phase 5 — Perception harness (works with any model).**
- Download HIT-UAV into `~/datasets/HIT-UAV`.
- `scripts/bench_detector.py MODEL...` reports:
  - mAP50 and mAP50-95 on the **real** HIT-UAV test split;
  - latency p50/p95 for PyTorch, ORT and (if available) TRT-FP16 at 640;
  - parameter count and FLOPs.
- Baseline `best.pt` now.
- Calibrate `detector_model.py` (P(detect) and confidence vs altitude and occlusion) from YOLO outputs.
- **Note:** perception accuracy is claimed only on real data. Gazebo thermal frames serve only for the swarm loop and for calibrating the occlusion trend, and the domain gap is stated.

**Phase 6 — Experiments (fast simulator unless noted).**
- **E1 Scalability:** N ∈ {3, 5, 10, 20, 50} × map 25/100/200 m. Metrics: connectivity, information, coverage, and compute time vs N (R2.4, R2.8).
- **E2 Channel:** four profiles × ε × transmit power (R1.4, R2.5).
- **E3 Failures:**
  - 1, 2 or 30% simultaneous failures;
  - cascading failures;
  - brownout;
  - Poisson failures.
- **E4 Dynamic obstacles:** moving debris, which blocks both line of sight and links (R2.4).
- **E5 Baselines:** B0–B6 × representative cells from E1–E4 (R2.6).
- **E6 Detection uncertainty:** sweep TPR/FPR, confidence noise σ and the 0.40/0.70 thresholds; measure TDR, false SA triggers, NRR and information loss (R2.7).
- **E7 Ablation:**
  - distance spring vs outage spring;
  - nearest neighbour vs LMST;
  - adaptive β on/off (C1);
  - SA on/off;
  - fusion on/off.
- **E8 SITL cross-validation (Gazebo + PX4):** N=3 and N=6 on `disaster_25` and `disaster_100`, 5 seeds each, with failures. Report the agreement between simulator and SITL trends; the claims rest on this.
- **E9 Faithful v1 re-run:** the v1 scenario (N=3, 25 m, one failure at 100 s) re-measured with the new code, so the old setting is reported honestly.

**Phase 7 — Integrate Karthik's model (when delivered).**
- Set `model_path`, then run `bench_detector.py` on YOLOv8n vs YOLO26n vs the new model(s).
- Recalibrate `detector_model.py`, then re-run E6, E8 and E9 (R1.5).

**Phase 8 — Paper + response.**
- **Structure** (the method gets about 40% of the pages, which addresses R1.3):
  1. Introduction
  2. Related work, with a comparison table: communication-aware planning, distributed coordination, connectivity control, MARL, edge AI (R1.2, R2.9)
  3. System model: network, channel, sensing, failure, and problem formulation
  4. Method: C1, C2 with propositions and proofs, C3, complexity
  5. Implementation (short)
  6. Evaluation
  7. Discussion and limitations
  8. Conclusion
- **Figures** are plots generated by scripts. Terminal screenshots are removed. At most one Gazebo screenshot is kept.
- **Language:** a word blacklist enforced by a lint script ("guarantee" only when a proposition backs it; no "bulletproof", "brutal", "ironclad", "zero data loss", "perfect", "flawless", "blistering", "violently", "hardcore"). The tone is objective and academic.
- **Reference audit** (a desk-reject risk). Every citation is verified against DOI or arXiv, which fixes:
  - text–reference mismatches, such as [14] and [25];
  - authors named in the text with no entry (Erdelj, Zeng, Mozaffari, Ahmad, Olfati-Saber, Poli, Bai, Yang, Cui);
  - any arXiv IDs that cannot be found.

  Missing core works are added: Zavlanos & Pappas, Ji & Egerstedt, Sabattini, Kantaros & Zavlanos, Li–Hou–Sha (LMST), Al-Hourani, 3GPP TR 36.777, MAPPO, the Ultralytics YOLO26 docs, and HIT-UAV (verify).
- **Claims policy:** each claim in the abstract and conclusion is tied to a table or figure value.
- **`response_to_reviewers.tex`:** each of the 16 comments maps to its change, its location in the paper, and its evidence.
- **Pre-submission check:** an internal mock review against the IoT-J reviewer comments, plus the target venue's scope and checklist. The faculty co-author signs off.

## 7. Reviewer-comment coverage check (each must be ✓ before submission)

| ID | Addressed by | Evidence in paper |
|---|---|---|
| R1.1 | C1–C3, detector not claimed as a contribution | Contributions list; E7 ablation |
| R1.2 | Comparison table + B2–B5 | §II table; E5 |
| R1.3 | Restructure; method about 40% | Page budget |
| R1.4 | `channel.py` + comm_emulator | §III channel model; E2 |
| R1.5 | YOLO26 benchmark | Perception table (Phase 5/7) |
| R1.6 | World-frame ledger, formal metrics, regression test | §VI metric definitions; E9 |
| R2.1 | C1 adaptive information-aware tension + outage spring vs prior work | §II table; E5/E7 |
| R2.2 | Lemma + Propositions 1 & 2, λ2(t) plots | §IV proofs; E1–E3 λ2 |
| R2.3 | Claims policy; failure sweeps | E3; abstract wording |
| R2.4 | E1, E2, E3, E4, E8 | Results section |
| R2.5 | 3GPP/ITU-parameterised channel + justification | §III; E2 |
| R2.6 | B0–B6 with equal tuning | E5 |
| R2.7 | E6 sensitivity | Sensitivity figures |
| R2.8 | E1 compute time vs N up to 50 | Scalability figure |
| R2.9 | Systematic table (advantages/limitations) | §II |
| R2.10 | Language lint + rewrite | Whole manuscript |

## 8. Code reused from the current repo
- `fitness_function.py:get_sa_offset_coords`: the perpendicular-offset math.
- `thermal_vision.py:40-52`: the normalize/invert/CLAHE preprocessing.
- `flight_commander.py`: the offboard and arm sequence.
- `best.pt`: the YOLOv8n baseline (ultralytics 8.4.14, HIT-UAV, 100 epochs).
- PX4 `Tools/simulation/gz/sitl_multiple_run.sh` and the `x500` / `x500_depth` models.

## 9. Open items (they do not block Phases 0–6)
- **Target venue.** The scope fit drives the framing and page limit. Candidates to discuss:
  - IEEE Transactions on Vehicular Technology;
  - IEEE Transactions on Network Science and Engineering;
  - IEEE Sensors Journal;
  - Elsevier Ad Hoc Networks or Computer Networks;
  - MDPI Drones.
- **What Karthik delivers:** variant(s) and format.
- **Jetson Orin Nano:** if available, on-device latency and power will be measured. Otherwise the paper reports desktop latency only and makes no Jetson claim.
- **Optional extra strength:** a small hardware or radio test (e.g. RSSI vs distance with 2 ESP32 or Wi-Fi nodes) to calibrate the channel profile.

## 10. Honest risk note
No rework can guarantee acceptance. This plan removes every concrete reason for rejection the reviewers gave, as well as the usual desk-reject triggers: unverifiable numbers, mismatched references, overclaiming and simulation-only evidence without statistics. The remaining risks are:
- **Novelty:** whether C1 is accepted as novel. Mitigation: a thorough related-work search in Phase 8 before final framing.
- **Simulation-only evaluation.** Mitigation: SITL cross-validation, an optional radio calibration, and releasing the code.

---

## Amendment A1 (2026-10-10): information-triggered intermittent connectivity
**Why.** In pilot 3 (400 m map, N = 8), always-connected mobility (SENTINEL v2-strict and B3) found and reported about 6× fewer victims than a lawnmower sweep. Store-and-forward lost under 4% of data. Strict connectivity is dominated on the primary SAR metric. The user chose this pivot.

**New core mechanism (replaces "always tethered" in C1/C2):**
- **EXPLORE:** while a UAV holds no *unsecured* confirmed detection, it explores freely (PSO coverage fitness, no spring, no projection). A detection is unsecured until the responder appears in its holder set, which is learned through gossip acknowledgements.
- **DELIVER:** triggered when the oldest unsecured detection is at least T_hold old.
  - The UAV moves toward the responder (or the last relay of a fresh heartbeat).
  - Once it reconnects, the channel-aware midpoint-ball projection (Proposition 1) keeps its links until the data is secured, then it returns to EXPLORE.
- **Responsibility:** the creator delivers its own detections. Copies held by others are insurance, and any copy older than 3·T_hold is treated as orphaned and delivered by its holder.
- **New theory, Proposition 3 (information exposure bound):** a detection is exposed to loss only during its unsecured window, W ≤ T_hold + D_max/v + τ. Under Poisson failures (rate λ_f per UAV), P(loss) ≤ 1 − e^{−λ_f W} ≤ λ_f·W. Gossip replication to k neighbours reduces this to roughly (λ_f W)^k.
- **The trade-off is explicit:** T_hold ∈ {0, 30, 60, 120, ∞} traces the coverage vs latency vs loss curve. This becomes a Pareto figure in the paper.
- **Kept:** C2 (outage-aware LMST projection and Proposition 1) applies during DELIVER. C3 (SA trigger) and the ledger are unchanged. Strict mode remains an ablation/operating point ("SENTINEL-strict").
- **Novelty positioning:** compare against intermittent-connectivity work (Kantaros & Zavlanos, IEEE TAC 2019 and follow-ups: fixed rendezvous schedules) and data-mule / DTN routing. SENTINEL's distinction is *event-/information-triggered* rendezvous with an analytic loss bound, failure awareness and a channel-aware guarantee. These references must be verified in the Phase 8 reference audit.
- **Experiments:** E1–E7 run with `sentinel` = the intermittent mode (T_hold default chosen from the pilot), plus `sentinel_strict` and a T_hold sweep (new E10).

---

## Amendment A2 (2026-10-14): evidence-led framing (user decision)
**Why.** Final experiments E2, E7 and E10 (30 seeds, Holm-corrected paired tests):
- The connectivity components (outage spring, LMST set, information tension, projection) do **not** change outcomes significantly when ablated. They act only during short delivery tethers.
- The **information-triggered delivery** significantly raises *timely* reporting: +11–28% of victims reported within 60 s of detection vs never delivering, with no significant cost in total discovery.
- **Collaborative SA sensing** contributes significantly.

**Framing rules for the paper:**
1. **Headline contribution:** information-triggered delivery for timely victim reporting, plus collaborative SA sensing with multi-view fusion.
2. **Primary metric:** victims reported to the responder within a deadline (`rep_within_60s`). Secondary metrics: total reported (`tdr_base`), report value (τ = 300 s), p90 latency, information loss.
3. Propositions 1–3 and the link-preservation machinery are presented as a **safety property during delivery** (measured link-violation rate ≈ 0.2%), **not** as a source of performance gains.
4. The ablation table is reported in full, including the null results.
5. T_hold is a mission design rule tied to the reporting deadline. The tuned value (115 s) optimises report value; the deadline analysis uses the E10 sweep.
6. The exploration layer (frontier goal, sector bias) is described as shared engineering available to all PSO-based methods, not as a contribution.

---

## Amendment A3 (2026-10-14): corroborative multi-view fusion
**Why.** E1 showed SENTINEL significantly *worse* than the baselines in dense swarms (200 m, N = 50). E11 (30 seeds) traced this to naive-Bayes fusion. Summing the log-odds of *all* views lets the many low-confidence views from occluded angles (logit < 0) pull a clearly seen victim below the confirmation threshold.

**Change.** Fused logit = the best view's logit + the sum of the *positive* logits of the other views. A weak view from a blocked angle is not evidence of absence, so it cannot veto, while corroborating views still add evidence. With one view, the result is that view's confidence.

**Effect (E11, 30 seeds):** within-60 s reports went from 0.667 (naive) to **0.823**, against 0.720 for B0, with 0 false entries.

**Protocol (to keep the comparison fair):**
- The final SENTINEL is re-tuned from scratch with the same 30-trial budget; earlier SENTINEL tunings are discarded and kept only for the record.
- The SENTINEL and ablation rows of every experiment are re-run (old rows backed up). Baseline rows stay valid (same seeds, unchanged code paths: the baselines use single best view, fusion off).
- `abl_naive_fusion` is kept as the ablation of this change.

---

## Amendment A4 (2026-10-15): tuning-space correction (user decision)
**What went wrong.** The first A3 re-tune searched SENTINEL's parameters **including `explore_lam`** (a soft tether while exploring). On 2026-10-10 that parameter had been explicitly *not adopted* into the SENTINEL design (round-2 tuning, documented in `tuned_sentinel_round2_rejected.yaml`), but it was left in the tuning space by mistake. The re-tune switched it on (`explore_lam` = 0.155) together with a strong spring (`lam0` = 2.39).
- Effect: SENTINEL won strongly in dense swarms but **lost in the severe channel**. Severe-channel coverage dropped from 0.94 to 0.66, and SENTINEL fell behind the lawnmower.

**Correction.** The tuning space now matches the defined design (no `explore_lam`). SENTINEL is re-tuned (same 30-trial budget), and its rows and the E8 SITL runs are re-run.

**Transparency.** The configuration with `explore_lam` and its results are kept (`tuned_sentinel_A3_explore.yaml`, `summary_A3explore.parquet`, `results/e8_sitl_A3explore`). The paper reports the tuning sensitivity: the soft tether trades severe-channel coverage for dense-swarm gains. Main results use the defined design.
