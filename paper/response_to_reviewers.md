# Response to the reviewers of IoT-66280-2026 (internal record)

IEEE IoT-J does not allow resubmission of this manuscript, so the reworked paper goes to a new venue.
This document records how every point raised in the IoT-J review is addressed in the new manuscript.
It can be adapted for a cover letter if the new venue asks about prior submissions.
All numbers refer to tables generated from `results/` (see `docs/RESULTS.md`).

## Associate Editor: "unclear novelty and contributions, vague system model, unconvincing experimental results"
- **Novelty and contributions:** the paper is re-centred on *timely victim reporting*. The contributions are information-triggered delivery, corroborative multi-view fusion, and a link-preservation safety layer with proofs, with the gains attributed by ablation (§I, §IV, §VI-H).
- **System model:** formal network, channel (SINR with path loss, obstructions, correlated shadowing, Rician/Rayleigh fading, interference), sensing, information and failure models, and an explicit objective (§III).
- **Experiments:**
  - 11 experiments, about 11,000 missions, 30 seeds each;
  - seven baselines with equal tuning budgets;
  - paired Wilcoxon tests with Holm correction;
  - a PX4/Gazebo cross-check (§VI).

## Reviewer 1
| # | Point | What changed | Where |
|---|---|---|---|
| 1.1 | Components look incremental (existing detector + distance penalty + threshold move) | The detector is no longer claimed. The contributions are the delivery trigger (+28% on-time reports in severe channel, E10), corroborative fusion (0.809 vs 0.656 for naive fusion and 0.720 for B0 in dense swarms, E11), and a proven safety layer. The ablation shows which parts matter, including null results. | §I, §IV, Tab. E7/E11, Fig. T_h |
| 1.2 | Compare with the closest distributed approaches | Comparison table vs λ2 control, edge potentials, intermittent connectivity, data ferrying/DTN, AoI collection, distributed PSO search, event-driven Bayesian search and MARL. Four of these families are implemented as tuned baselines (B2–B5). | §II, Tab. related; §VI-A |
| 1.3 | Reads like an experimental report | Method and theory (§III–IV) form the core. Implementation is a short section, with one figure per aspect. Terminal screenshots are removed, and all numbers come from scripts. | §III–V |
| 1.4 | Fixed 50 m threshold; no path loss/fading/interference/reliability | SINR-based outage model with obstructions, fading, shadowing and interference. Packets are dropped by sampled SINR, in the simulator and in Gazebo (comm_emulator). 50 m reappears as the urban one-obstruction reliable range, not as an assumption. Four channel profiles are evaluated. | §III-B, Tab. channel, Tab. E2 |
| 1.5 | Evaluate YOLO26 | YOLO26n vs YOLOv8n under an identical training protocol: equal mAP50 (0.948 vs 0.947), higher precision and mAP50-95, 19% fewer parameters, 31% fewer GFLOPs, equal desktop latency (1.8 ms TensorRT). YOLO26n is deployed. | §VI-B, Tab. perception |
| 1.6 | 10 targets / 9 detections / 15 coordinates | Root cause: the earlier code stored camera-frame offsets with constant telemetry, so sightings could not be merged. The new world-frame ledger has gating, consolidation and a regression test. Duplicate and false confirmed entries are ~0 across experiments, and the earlier scenario is re-run faithfully (E9). | §III-D, §VI-J |

## Reviewer 2
| # | Point | What changed | Where |
|---|---|---|---|
| 2.1 | Signal-Spring is just a penalty; distinguish from topology maintenance / comm-aware trajectories | The spring now penalises predicted *outage* (not distance) and applies only during delivery. It is backed by a projection with a proof. Always-on maintenance (B3, B4, SENTINEL-strict) is shown to lower timely reporting (0.03–0.13 for strict). | §IV-B/C, §VI-H |
| 2.2 | Pairwise distance ≠ global connectivity; graph-theoretic analysis needed | Lemma 1 (nearest-neighbour counter-example), Lemma 2 (LMST ⊇ MST), Proposition 1 (link preservation under staleness and obstruction inflation), Corollary 1 (connected delivery component), Proposition 2 (union bound). Measured: 15 violations in 436,057 preserved link-steps. | §IV-C, §VI-I |
| 2.3 | "Guarantees total network survival", "zero data loss" from N=3, one failure | Such claims are removed: v1 had 46 flagged phrases, the new paper has 0 (lint script). Information loss is *measured* under six failure models (≤ 8.5% worst case for any method), and no loss advantage is claimed. | whole paper, §VI-D |
| 2.4 | More swarm sizes, larger areas, dynamic obstacles, channel variation, multiple failures | N = 3–50; 200 and 400 m; 0/5/15 moving obstacles; four channel profiles; six failure models; PX4/Gazebo cross-check. | §VI-C–G, §VI-J |
| 2.5 | Communication model too simple | As R1.4. The parameter sources are cited, and the lack of measurement calibration is stated as a limitation. | §III-B, §VII |
| 2.6 | Baselines: comm-aware optimisation, MARL, graph-based, topology control | B0 PSO, B1 earlier design, B2 comm-aware PSO, B3 λ2-gradient (oracle λ2), B4 potentials with hysteresis, B5 MAPPO, B6 lawnmower. All tuned with the same Optuna budget on separate seeds. MAPPO is trained to a plateau (600 iterations × 16 episodes, learning curve shown) and selected on tuning seeds; it is the strongest baseline in dense swarms, where SENTINEL's lead over it is not significant for N=10–20 on the 200 m area. | §VI-A, Fig. MAPPO, §VI-E |
| 2.7 | Link perception to decisions; sensitivity to detection uncertainty | Calibrated detector model. Sweeps of confidence noise, detector strength, false-alarm rate, SA/confirmation thresholds, and an adverse hot-clutter case (E6, E6b, E6c). SENTINEL stays significantly ahead in all settings. | §VI-G, Tab. E6c |
| 2.8 | O(N) scalability untested | Complexity per agent depends on neighbourhood size. Measured planning time 4–11 ms for N = 3–50, far below the 1 s period; significantly better than all baselines at every N on the 400 m area. | §IV-H, §VI-E, Fig. compute |
| 2.9 | Systematic related-work comparison | Comparison table with six criteria across eight families. | §II |
| 2.10 | Informal language; exaggerated claims | Full rewrite in neutral style; banned-phrase lint; every quantitative claim in the abstract and conclusion traced to a table. Severe-channel results are reported as "ahead or tied", not as wins. | whole paper |

## Corrections found in the authors' own audit (not raised by the reviewers)
- **The earlier code and numbers.** The earlier code did not implement the PSO or the fitness function described in the earlier manuscript, and its reported numbers (NRR 66 → 100%, TDR 60/70/90%, 4 ms recovery, 1.2 ms on Jetson) were not produced by any script. All results in the new paper are new and reproducible.
- **Reference errors in the earlier manuscript:**
  - HIT-UAV was attributed to the wrong authors (correct: Suo et al., Sci. Data 2023);
  - the symmetry-informed MARL paper was attributed to the wrong first author (correct: Shi et al., IEEE TMC 2025);
  - several references were mismatched with the text.

  All references in the new paper were verified (`docs/related_work_notes.md`, `paper/references.bib`).
- **Two design errors were found and fixed during this work, both documented** (REWORK_PLAN Amendments A3/A4):
  - naive-Bayes fusion penalised occluded views;
  - a parameter excluded from the design had been left in the tuning space.
