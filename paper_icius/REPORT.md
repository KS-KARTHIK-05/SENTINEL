# ICIUS 2026 revision (UN006): build report

## Build
- **Template:** IEEEtran `\documentclass[conference]{IEEEtran}`. No ICIUS template was supplied.
- **Commands:**
  ```bash
  source ~/sentinel_venv/bin/activate
  python paper_icius/scripts/make_icius.py
  cd paper_icius && latexmk -pdf main.tex
  ```
- **Checks:**
  - 6 pages (`pdfinfo`); the last page is about half full, so about 5.5 pages in total.
  - 0 overfull boxes, 0 undefined references or citations.
  - Abstract: 161 words.
  - Page 1 (`pdftotext`) contains the title, all three authors, their affiliations, e-mail addresses and the Index Terms.
  - Zero hits for the banned filler words and for the forbidden v1 numbers (1.2 ms, 4 ms, 66/100 %, 70/90 %, 94.7 %, Jetson).
- **Title:** matches the original character for character (checked by string comparison).

## Traceability
- **Generated numbers:** `scripts/make_icius.py` writes `generated/numbers.tex` (118 macros) and `generated/tab_*.tex`. It reads them from the per-seed results in `~/sentinel_ws/results/`, the stored run configurations and the channel/agent code. Every result number in `main.tex` is a macro.
- **Literal numbers left in the text** are definitions or descriptions, not results:
  - the 60 s deadline, the 95 % confidence intervals and the 90th percentile;
  - the 50 m radius of the first version;
  - the coordinates in the Lemma 1 example;
  - the hardware name RTX 3060.

## Kept tables and figures
| ICIUS | Source in the 13-page draft |
|---|---|
| Table I, closest approaches | Table I, shrunk to 8 rows × 5 criteria |
| Table II, channel profiles and reliable ranges | Table II, recomputed from `channel.py` |
| Table III, default parameters | parameter table, trimmed, generated from the code and `tuned.yaml` |
| Table IV, channel profiles | Table V, mean ± CI half-width |
| Table V, six failure settings, urban | Table VI. The header bug is fixed: the missing "single crash" label was caused by `30\\%` being escaped twice in `make_results.py` |
| Table VI, swarm size, 400 m | Table VIII |
| Table VII, attribution | condensed from Tables XII/XIII and E11/E11b: urban N=8, severe N=8, dense N=50 |
| Fig. 1, Gazebo scene | screenshots S2 and S4 |
| Fig. 2, ROS 2 node graph | Fig. 3 |
| Fig. 3, T_h trade-off | Fig. 7 |

Propositions 2 and 3, Tables VII, IX, X, XI and XIV, and Figs. 1, 5 and 6 are left out, as instructed.

## Reference verification (19 cited)
| Key | Status | Source |
|---|---|---|
| zavlanos2008 | verified | https://doi.org/10.1109/TRO.2008.2006233 |
| ji2007 | changed: DOI added | https://doi.org/10.1109/TRO.2007.900638 |
| sabattini2013 | verified | https://doi.org/10.1177/0278364913499085 |
| kantaros2017 | verified | https://doi.org/10.1109/TAC.2016.2626400 |
| li2003lmst | changed: vol. 3, pp. 1702–1712 and DOI added | https://doi.org/10.1109/INFCOM.2003.1209193 |
| clerc2002 | changed: DOI added | https://doi.org/10.1109/4235.985692 |
| yu2022mappo | verified | NeurIPS 35, pp. 24611–24624, Crossref 10.52202/068431-1787 |
| hituav (Suo et al.) | verified | https://doi.org/10.1038/s41597-023-02066-6 |
| yolo26 | changed: release date put in the note | https://docs.ultralytics.com/ ; PyPI ultralytics 8.4.0 uploaded 2026-01-14: https://pypi.org/project/ultralytics/8.4.0/ |
| cui2023ferry | changed: pp. 1–8 added | https://doi.org/10.1109/LCN58197.2023.10223388 |
| kaul2012aoi | verified | https://doi.org/10.1109/INFCOM.2012.6195689 |
| bafarassat2026aoi | changed: now accepted in IEEE IoT-J (early access) | https://doi.org/10.1109/JIOT.2026.3743973 ; https://arxiv.org/abs/2608.00061 |
| niu2026eventbayes | verified | https://doi.org/10.3390/s26165189 ; https://pubmed.ncbi.nlm.nih.gov/42655497/ |
| zhu2026mountain | changed: vol. 10, no. 7, art. 493 added | https://doi.org/10.3390/drones10070493 |
| yang2024dapso | verified | https://www.mdpi.com/1424-8220/24/23/7639 |
| shi2025symmetry (Shi et al.) | changed: DOI added | https://doi.org/10.1109/TMC.2025.3553285 |
| gudmundson1991 | changed: DOI added | https://doi.org/10.1049/el:19911328 |
| tr36777 | changed: year 2018 → 2017 (V15.0.0 approved Dec. 2017, uploaded 6 Jan. 2018) | https://www.3gpp.org/dynareport/36777.htm |
| itup1411 | changed: version P.1411-12 (08/2023) stated; since superseded by P.1411-13 (09/2025) | https://www.itu.int/rec/R-REC-P.1411-12-202308-I/en |

Dropped from the v2 bibliography because they are not cited here: stone1975 (verified), phung2020mpso, chandran2024disaster, amodu2023aoi.

## Numbers that could not be traced
None in the ICIUS manuscript. Two errors were found in the 13-page journal draft, which was left untouched as instructed:
- **Link-violation total.** The draft says 93 in 1,349,703 link-steps. That sum wrongly included the superseded `*_oldmetric` result folders. The correct value is **65 in 1,317,947 (4.9×10⁻⁵)**, as used here. The repo README repeats the draft's value.
- **Urban margin.** The draft gives "23–30 %", computed from rounded table values. The exact value from the per-seed data is **23–29 %**.
- **Draft Table VI header.** The header is broken by the escaping bug described above.

## Still unaddressed from the reviews (stated in the response)
- No Jetson Orin Nano latency, power or thermal measurement.
- The detector is not fine-tuned or evaluated at 25–30 m.
- No hard-distance-wall baseline and no relay-formation baseline. B1 and B4 are the closest variants.
- λ/R_max ablation: no λ sweep. λ₀ and β are tuned, and the distance-spring vs outage-spring ablation is reported in the response only, not in the 6-page paper.
- No real flights. A PX4 SITL cross-check is given instead.
- **Wording deviation.** The prompt's table says "calibrated detector model". The model's parameters were set by hand and never fitted, so the paper and the response say "parameterised".
- The revision deadline (11 Oct. 2026) has passed. The authors are contacting the editor.
