# SENTINEL v2 — How to run, verify and capture screenshots

Everything below runs on this workstation (Ubuntu 22.04, ROS 2 Humble, Gazebo Harmonic, PX4 `main`, RTX 3060).
Open a new terminal for each numbered step, and run the **environment block** in every terminal first.

## 0. Environment block (every terminal)
```bash
source /opt/ros/humble/setup.bash
source ~/sentinel_ws/install/setup.bash
source ~/sentinel_venv/bin/activate
```
If you changed any code, rebuild once with `cd ~/sentinel_ws && colcon build`.

---

## ★ One command — everything launches automatically
```bash
~/sentinel_ws/run_sentinel_demo.sh
```
This single command:
1. sources ROS 2, the workspace and the Python environment (no other setup needed);
2. starts **Gazebo with its GUI** and **6 PX4 drones** in the 100 m disaster world (the recommended swarm size on this PC; up to 9 is possible with `-n 9`);
3. starts the **whole SENTINEL stack** (flight control, YOLO26n perception, decentralised agents, radio emulator, crash of UAV 1 at t = 150 s, metrics);
4. opens **rqt_image_view** (YOLO26n annotated thermal feed of UAV 0) and **rqt_graph** (node graph);
5. prints a **live status line every 10 s** (victims found/reported, coverage, connectivity, PDR).

**Ctrl-C stops everything cleanly.** Metrics and logs go to `~/sentinel_ws/results/demo_<date_time>/`.

Useful options (`-h` for all):
```bash
~/sentinel_ws/run_sentinel_demo.sh -w sentinel_d25_s0          # the v1 25 m scene
~/sentinel_ws/run_sentinel_demo.sh -p model                    # parameterised detector model instead of YOLO26n
~/sentinel_ws/run_sentinel_demo.sh -m B0_pso                   # run a baseline instead of SENTINEL
~/sentinel_ws/run_sentinel_demo.sh -n 9 -t 300 --no-crash      # 9 drones (maximum on this PC), stop after 300 s, no crash
~/sentinel_ws/run_sentinel_demo.sh -n 3                        # 3 drones (fastest, real-time)
```
Tested on this machine: the GUI, the drones, the stack and the viewers all start; automatic stop; no processes left behind. With 6 drones the simulation runs at about 0.73× real time (see the capacity table at the end).

---

## 1. Manual launch (step by step, same result as the one command)

### Terminal A — Gazebo + 6 PX4 drones (with GUI)
```bash
cd ~/sentinel_ws/src/SENTINEL/sentinel_ws/src/sentinel_gazebo/scripts
./launch_sitl.sh 6 sentinel_g100_s0 --gui
```
- Wait for `SITL running: 6 vehicles ...`. The Gazebo window opens with a 100 m disaster map: grey buildings (no-fly), brown rubble, Rescue Randy victims (some under concrete slabs) and warm debris.
- Other worlds: `sentinel_d25_s0` (rebuild of the v1 25 m scene), `sentinel_g100_s1`, `sentinel_g100_s2`, `sentinel_g200_s0`.

### Terminal B — SENTINEL stack with the deployed YOLO26n detector
```bash
ros2 launch sentinel_ai swarm_launch.py n_uav:=6 world:=sentinel_g100_s0 perception:=yolo \
    tuned_yaml:=$HOME/sentinel_ws/src/SENTINEL/sentinel_ws/src/sentinel_sim/configs/tuned.yaml \
    failures:='[{"t":150,"uav":1,"kind":"crash"}]' out_dir:=$HOME/sentinel_ws/results/demo_yolo
```
- The drones take off to 30 m (about 20 s), then spread out and search.
- `perception:=yolo` runs YOLO26n on the Gazebo thermal camera.
- `perception:=model` uses the parameterised detector model instead (this is what the three-world PX4 check uses).
- UAV 1 is crashed at t = 150 s (chaos monkey). Watch the others continue and deliver reports.

### Terminal C — watch the thermal camera and the reports
```bash
# live thermal image of UAV 0 (radiometric 16-bit; rqt shows it auto-scaled)
ros2 run rqt_image_view rqt_image_view /world/sentinel_g100_s0/model/x500_thermal_0/link/thermal_link/sensor/thermal_camera/image
# geolocated detections, ledger and radio statistics
ros2 topic echo /px4_0/sentinel/targets
ros2 topic echo /px4_0/sentinel/ledger
ros2 topic echo /sentinel/comm_stats
ros2 topic echo /px4_0/sentinel/detector_stats      # YOLO26n latency (p50/p95)
```
- Metrics are written to `out_dir` every 10 s: `series.csv` and `summary.json`.
- Stop with Ctrl-C in Terminal B, then Terminal A.

---

## 2. Screenshots

The README figures are in `docs/figures/`:

| File | What | How |
|---|---|---|
| `S1_world_overview.png` | Disaster world from above (buildings, rubble, victims, slabs) | Gazebo GUI, View Angle plugin → Top, grid hidden (⋮ menu → Grid Config). |
| `S2_victims_occlusion.png` | A victim partly under a slab | Gazebo GUI: Entity Tree → right-click a `victim_*` or `slab_*` model → Move to, then orbit. |
| `S4_thermal_frame.png` | Raw white-hot thermal frame | While the demo runs (`--no-crash`, drones at altitude): `sentinel_ws/scripts/save_thermal_frames.sh` saves box-free frames to `~/sentinel_ws/results/thermal_frames/uav0/`. |
| `S5_yolo26n_pipeline.png` | YOLO26n boxes in the loop (green ≥ 0.70 confirmed, orange 0.40–0.70 second-view band, red < 0.40) | `ros2 run rqt_image_view rqt_image_view /px4_0/sentinel/thermal_annotated` (published with `perception:=yolo`). |

**Domain gap:** the deployed YOLO26n is trained on real HIT-UAV images. In Gazebo it also boxes rubble edges
and misses many rendered mannequins, so S5 shows the pipeline running, not detection accuracy.
Accuracy on rendered frames, before and after fine-tuning, is in the extended results.

---

## 3. Reproducing the numbers

Run from `~/sentinel_ws/src/SENTINEL` with the environment block sourced.

| What | Command | Output |
|---|---|---|
| Unit tests | `for p in sentinel_core sentinel_sim sentinel_ai; do (cd sentinel_ws/src/$p && python -m pytest -q test/); done` | pass/fail per package |
| Monte-Carlo check that the projection keeps the swarm connected (LMST neighbours, adversarial steps) | `python sentinel_ws/scripts/verify_prop1.py 10000 0` | `failures=0` |
| One experiment | `cd sentinel_ws/src/sentinel_sim && python -m sentinel_sim.runner configs/experiments/e2_channel.yaml --out ~/sentinel_ws/results/e2_channel --jobs 18` | `~/sentinel_ws/results/e2_channel/summary.parquet` |
| Tables + significance for one experiment | `cd sentinel_ws/src/sentinel_sim && python -m sentinel_sim.analyze ~/sentinel_ws/results/e2_channel` | `tables.md`, `stats.csv`, `tests.csv`, Pareto PDFs |
| Detector benchmark | `python sentinel_ws/scripts/bench_detector.py --models <pt files> --out ~/sentinel_ws/results/perception` | `perception.md/.csv` |
| Revision experiments (B7/B8, sensitivity, rendered-frame detector, fitted detector model, PX4) | `sentinel_ws/scripts/overnight/run_overnight.sh`; the later corrections in `run_e5_recheck.sh` and `run_baseline_rerun.sh` (same folder) | `~/sentinel_ws/results/overnight_2026-10-28/`, `e5_recheck_2026-10-29/`, `baseline_rerun_2026-10-29/` |
| PX4/Gazebo checks | `sentinel_ws/scripts/overnight/run_sitl.sh OUT PERCEPTION MODEL_PATH SEEDS...` (PERCEPTION `model` for the three-world check, `yolo` for the ten-world check) | `OUT/sentinel_g100_s<seed>_<method>/summary.json` |
| Every number, table and the extended results of the ICIUS paper | `python paper_icius_ios/scripts/make_ios.py` | `paper_icius_ios/generated/`, `docs/icius2026_extended_results.md` |
| README results section | `python sentinel_ws/scripts/make_readme.py` | `README.md` |

All runs are deterministic per seed (common random numbers), so re-running reproduces the same numbers.
Experiment definitions: `sentinel_ws/src/sentinel_sim/configs/experiments/*.yaml`. Tuned parameters:
`configs/tuned_v4.yaml` (the paper's tables) and `configs/tuned.yaml` (the PX4 checks; see the changelog).

## 4. Where everything is
- Results: `~/sentinel_ws/results/<experiment>/summary.parquet` (one row per run), outside the repository.
  `make_ios.py` reads them from there.
- Results file of the paper: `docs/icius2026_extended_results.md` (generated).
- Analysis plan of the revision experiments: `docs/analysis_plan_icius2026.md`; history: `docs/CHANGELOG.md`.
- Paper: `paper_icius_ios/` (`main.pdf`; `main_highlighted.pdf` marks text changed in the revision).

## Gazebo capacity of this workstation (measured 2026-10-15)
i7-12700 (20 threads), RTX 3060, headless; every drone runs a full PX4 autopilot plus physics.

| Drones | Thermal cameras (YOLO mode) | Camera-less (`-p model`) |
|---|---|---|
| 3 | all fly, real-time factor 1.00 | — |
| 6 | all fly, RTF 0.73 | — |
| 9 | **all fly, RTF 0.70** | all fly, RTF 0.66 |
| 12 | 11/12 fly, RTF 0.36 | 10/12 fly, RTF 0.70 |

**Practical maximum: 9 drones.** The launcher's default is **6**, which keeps the simulation close to real time.
The limit is CPU (one PX4 per drone plus physics); the GPU stays at about 2–3 %. Above 9 drones, PX4's
simulated sensors fall behind and an occasional drone fails its position estimate at take-off.
The main experiments with 3–50 drones use the fast simulator, not Gazebo.
