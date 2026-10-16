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
~/sentinel_ws/run_sentinel_demo.sh -p model                    # calibrated detector model instead of YOLO26n
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
- Wait for `SITL running: 3 vehicles ...`. The Gazebo window opens with a 100 m disaster map: grey buildings (no-fly), brown rubble, Rescue Randy victims (some under concrete slabs) and warm debris.
- Other worlds: `sentinel_d25_s0` (rebuild of the v1 25 m scene), `sentinel_g100_s1`, `sentinel_g100_s2`, `sentinel_g200_s0`.

### Terminal B — SENTINEL stack with the deployed YOLO26n detector
```bash
ros2 launch sentinel_ai swarm_launch.py n_uav:=6 world:=sentinel_g100_s0 perception:=yolo \
    tuned_yaml:=$HOME/sentinel_ws/src/SENTINEL/sentinel_ws/src/sentinel_sim/configs/tuned.yaml \
    failures:='[{"t":150,"uav":1,"kind":"crash"}]' out_dir:=$HOME/sentinel_ws/results/demo_yolo
```
- The drones take off to 30 m (about 20 s), then spread out and search.
- `perception:=yolo` runs YOLO26n on the Gazebo thermal camera.
- `perception:=model` uses the calibrated detector model instead (this is what the E8 cross-validation uses).
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

## 2. Screenshots for the paper (suggested list)

| # | What | How |
|---|---|---|
| S1 | Disaster world overview (buildings, rubble, victims, slabs) | Gazebo GUI, top-down view: click the "view angle" plugin → top. Hide the grid. |
| S2 | Close-up of a victim partly under a slab and one fully buried | Gazebo GUI: zoom on `victim_*` / `slab_*` models (Entity Tree → right-click → Move to). |
| S3 | Swarm in flight: 6 drones over the map | Gazebo GUI during Terminal B, perspective view. |
| S4 | Thermal camera frame (white-hot) with victims | rqt_image_view (Terminal C): pick the topic, then use its "Save as image" button. |
| S5 | YOLO26n running in the loop (boxes: green ≥ 0.70 confirmed, orange 0.40–0.70 SA band, red < 0.40) | `ros2 run rqt_image_view rqt_image_view /px4_0/sentinel/thermal_annotated` (published automatically with `perception:=yolo`). 145 frames are already saved in `handover/demo_yolo26n/frames`. ⚠️ **Domain gap:** in Gazebo, YOLO26n puts false boxes on rubble edges and misses many mannequins. Present S5 as the *deployed pipeline running*, not as detection accuracy (accuracy is claimed on real HIT-UAV data only). |
| S6 | rqt_graph of the decentralised architecture | `ros2 run rqt_graph rqt_graph` while the stack runs (shows the per-UAV nodes, comm_emulator, no central coordinator). |

Gazebo GUI screenshot: use the camera icon in the toolbar, or the system shortcut (Shift+PrintScreen for a region).
Save the files to `~/sentinel_ws/src/SENTINEL/paper/figures/` with the names S1_world.png etc.

---

## 3. Reproducing the numbers

| What | Command | Output |
|---|---|---|
| Unit tests (396) | `cd src/SENTINEL/sentinel_ws/src/sentinel_core && python -m pytest -q test/` | pass/fail |
| Proposition 1 Monte-Carlo (10⁴ trials) | `python src/SENTINEL/sentinel_ws/scripts/verify_prop1.py 10000 0` | `failures=0` |
| One experiment | `cd src/SENTINEL/sentinel_ws/src/sentinel_sim && python -m sentinel_sim.runner configs/experiments/e2_channel.yaml --out ~/sentinel_ws/results/e2_channel --jobs 18` | `results/e2_channel/summary.parquet` |
| Tables + significance for one experiment | `python -m sentinel_sim.analyze ~/sentinel_ws/results/e2_channel` | `tables.md`, `stats.csv`, `tests.csv`, Pareto PDFs |
| Detector benchmark | `python src/SENTINEL/sentinel_ws/scripts/bench_detector.py --models <pt files> --out ~/sentinel_ws/results/perception` | `perception.md/.csv` |
| SITL cross-validation (E8) | `src/SENTINEL/sentinel_ws/scripts/run_e8_sitl.sh 300 3` | `results/e8_sitl/*/summary.json` |

All runs are deterministic per seed (common random numbers), so re-running reproduces the same numbers.
Experiment definitions: `src/SENTINEL/sentinel_ws/src/sentinel_sim/configs/experiments/*.yaml`. Tuned parameters: `configs/tuned.yaml`.

## 4. Where everything is
- Plan and decisions: `docs/REWORK_PLAN.md` (Amendments A1–A3); progress log: `docs/PROGRESS.md`.
- Results: `results/<experiment>/summary.parquet` (one row per run). Final tables: `docs/RESULTS.md` (generated when all runs finish).
- Paper draft: `src/SENTINEL/paper/`; response to the reviewers: `src/SENTINEL/paper/response_to_reviewers.md`.

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
The main experiments with 3–50 drones use the fast simulator (Section VI-E), not Gazebo.
