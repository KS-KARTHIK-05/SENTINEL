# Related-work notes (for Phase 8): verified references and positioning

Status: ✅ = bibliographic details verified online (2026-10-10). Each entry still needs a DOI check when the .bib file is written.

## Closest work and how SENTINEL differs

| Strand | Representative work | What it assumes / does | SENTINEL difference |
|---|---|---|---|
| Continuous connectivity maintenance | ✅ Sabattini, Chopra, Secchi, "Decentralized connectivity maintenance for cooperative control of mobile robotic systems," IJRR 32(12):1411–1423, 2013, doi:10.1177/0278364913499085. Also Zavlanos & Pappas (2008), Ji & Egerstedt (T-RO 2007) (to verify) | The graph stays connected *at all times* (λ2 gradient, edge-tension potentials) | Our pilot 3 shows that always-connected motion cuts discovery about 6× in large areas. SENTINEL tethers only while delivering. These methods are our baselines B3 and B4. |
| Intermittent connectivity | ✅ Kantaros & Zavlanos, "Distributed Intermittent Connectivity Control of Mobile Robot Networks," IEEE TAC 62(7):3109–3121, 2017, doi:10.1109/TAC.2016.2626400 | Robots meet at the vertices of a mobility graph on schedules that recur infinitely often (LTL) | SENTINEL's rendezvous are *event-triggered by detections* (unknown, online information sources), with an analytic loss bound (Proposition 3) and channel-aware link preservation during delivery (Proposition 1). |
| Data ferrying / DTN / store-carry-forward | Joint data ferrying and cell coverage by UAV swarms (arXiv 2307.02988); capacity and delay of UAV networks with mobility (arXiv 2311.15017); a DTN survey for disaster monitoring (Can. Sci. Publishing, doi:10.1139/dsa-2023-0079) | Ferry routes are planned for **known** data sources; delay = waiting time + distance/speed | In SAR the sources are *discovered by the search itself*. SENTINEL decides *when* to stop searching and deliver (T_hold), giving a latency vs discovery trade-off. |
| Age of information (AoI) in UAV data collection | Reliability- and connectivity-constrained AoI for UAV-swarm IoT data collection (arXiv 2608.00061); AoI minimisation in massive IoT via MARL (arXiv 2309.14757); age-aware UAV data-collection survey (Drones 7(4):260) | Minimise the age of data from fixed sensors | Our report latency and time-discounted report value are AoI-like metrics. Cite AoI and connect our metric to it. |
| Event-triggered multi-UAV coordination | Prescribed-time event-triggered guidance (Drones, doi:10.3390/drones10090670); event-driven Bayesian search (Sensors, doi:10.3390/s26165189), which needs a connected graph and lossless links | Event-triggered *communication*, assuming a connected network | We do not assume connectivity; we trigger *mobility* for delivery. |
| SAR swarm search under limited communication | Cooperative search for mountain SAR under distance-limited communication (Drones, doi:10.3390/drones10070493) | Distance-threshold links, no responder | A realistic channel model, a responder sink and failures. |
| Detector | ✅ Ultralytics YOLO26: released 2026-01-14 (ultralytics v8.4.0), NMS-free, DFL removed, edge-oriented. Independent edge benchmarks: arXiv 2509.25164, 2510.09653 (to check) | — | R1.5: benchmark YOLOv8n vs YOLO26 (and Karthik's model) on HIT-UAV. Use vendor speed claims only with attribution. |

## Still to verify (Phase 8 reference audit)
- Zavlanos & Pappas, "Distributed connectivity control of mobile networks," IEEE T-RO 2008.
- Ji & Egerstedt, "Distributed coordination control of multiagent systems while preserving connectedness," IEEE T-RO 2007.
- Li, Hou, Sha, "Design and analysis of an MST-based topology control algorithm," INFOCOM 2003 (LMST).
- Clerc & Kennedy, "The particle swarm — explosion, stability, and convergence…," IEEE TEC 2002 (constriction).
- Yu et al., "The surprising effectiveness of PPO in cooperative multi-agent games" (MAPPO), NeurIPS 2022 D&B.
- Gudmundson, "Correlation model for shadow fading in mobile radio systems," Electron. Lett. 1991.
- 3GPP TR 36.777 (UAV channel); ITU-R P.1411; Al-Hourani et al. (A2G path loss).
- Suo et al., HIT-UAV dataset, Scientific Data 2023 (the v1 draft attributes it to "S. Li et al." — **check the author list**).
- Koopman, search theory (Bayesian detection update used in `maps.py`).
- v1 draft references that are suspicious or mismatched: [14], [25], and authors named in the text with no entry (Erdelj, Zeng, Mozaffari, Ahmad, Olfati-Saber, Poli, Bai, Yang, Cui). arXiv IDs 2605.01371, 2604.13278, 2603.25218 and 2601.14437 must be checked.

## Sources consulted (2026-10-10)
- https://dukespace.lib.duke.edu/items/e4fe0047-913e-400b-a81f-eb8960c4ebcb
- https://journals.sagepub.com/doi/10.1177/0278364913499085
- https://community.ultralytics.com/t/new-release-ultralytics-v8-4-0/1747
- https://www.businesswire.com/news/home/20260114168538/en
- https://arxiv.org/html/2307.02988
- https://arxiv.org/pdf/2608.00061
- https://cdnsciencepub.com/doi/10.1139/dsa-2023-0079
- https://doi.org/10.3390/s26165189
- https://doi.org/10.3390/drones10090670
- https://doi.org/10.3390/drones10070493
