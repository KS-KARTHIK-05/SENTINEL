# Screenshots for the paper

## 1. Start the simulation (one command)
Open a terminal and run:
```bash
~/sentinel_ws/run_sentinel_demo.sh
```
That's all. Gazebo opens with the disaster world and **6 drones**, the SENTINEL stack starts, and two
viewer windows open (thermal camera with YOLO26n boxes, and the node graph). The drones take off
after ~20 s and start searching. The terminal prints a status line every 10 s.
**Stop with Ctrl-C.**

Variants used for some screenshots:
```bash
~/sentinel_ws/run_sentinel_demo.sh -w sentinel_d25_s0      # small 25 m scene (rebuild of the original v1 world)
~/sentinel_ws/run_sentinel_demo.sh -p model                # perception from the calibrated detector model (no YOLO window boxes)
```

## 2. How to capture
- Whole window: click the window, then **Alt + PrintScreen**.
- Region: **Shift + PrintScreen**, drag a box.
- Gazebo camera views: right panel → *View Angle* plugin (Top / Front / Isometric). If it's missing, use the
  ⋮ menu (top right) → search "View Angle".
- Hide the grid: ⋮ menu → *Grid Config* → untick *Show*.
- Find a model (victim, slab, drone): left panel *Entity Tree* → right-click the name → *Move to*.

Save every file to:
```
~/sentinel_ws/src/SENTINEL/paper/figures/
```
with exactly the file name given below (the LaTeX source refers to these names).

## 3. Required screenshots

| # | File name | What to show | When / how | Figure in the paper |
|---|---|---|---|---|
| **S1** | `S1_world_overview.png` | Whole 100 m disaster world from above: grey buildings, brown rubble, victims, warm debris | Right after launch, before take-off. View Angle → **Top**, grid hidden, zoom out until the whole map fits | Fig. 1 (Section V, Implementation) |
| **S2** | `S2_victims_occlusion.png` | Close-up, oblique: a victim lying partly under a slab on posts (and, if possible, one fully covered — e.g. `victim_0`, look under the slab edge) | Entity Tree → `victim_0` (buried) and a `slab_*` near another victim → *Move to*, then orbit for an oblique view | Fig. 2 left (Section V) |
| **S3** | `S3_swarm_in_flight.png` | The 6 drones flying over the map | Default launch (6 drones); ~60–90 s after launch. Isometric view, zoomed so drones and buildings are both visible | Fig. 4 right (Section V) |
| **S4** | `S4_thermal_frame.png` | Raw thermal camera image (white-hot) of the ground with a victim visible | Default run (`run_sentinel_demo.sh --no-crash`). Once the drones are at 30 m, in a second terminal run `~/sentinel_ws/save_thermal_frames.sh`; it saves 60 box-free white-hot frames (one every 2 s) to `~/sentinel_ws/results/thermal_frames/uav0/`. Pick one with a victim in view. (`-p model` has no camera, so it can't be used for this.) | Fig. 2 right (Section V) |
| **S5** | `S5_yolo26n_pipeline.png` | YOLO26n boxes on the thermal feed (green ≥0.70 confirmed, orange SA band) | Default launch; the rqt_image_view window shows `/px4_0/sentinel/thermal_annotated`; save a frame that shows a box on a victim. Take it from a fresh run (the 145 old frames in `handover/demo_yolo26n/` predate the 2026-10-21 scene fix) | Fig. 4 left (Section V), **as a pipeline illustration only** (see note) |
| **S6** | `S6_ros_graph.png` | ROS 2 node graph: per-drone nodes (`flight_commander`, `thermal_vision`, `emps_agent`), `comm_emulator`, `metrics_logger` — no central coordinator | rqt_graph window → *Nodes only*, refresh, *Fit in view*; screenshot the window | Fig. 3 (Section V) |
| S7 *(optional)* | `S7_v1_scene.png` | The rebuilt v1 25 m scene (for the E9 discussion) | `run_sentinel_demo.sh -w sentinel_d25_s0`, top view | not placed yet (could go in Section VI-J, E9) |

### Note on S5 (important)
YOLO26n is trained on real HIT-UAV thermal images. In Gazebo it also boxes rubble edges and misses many
rendered mannequins (a simulation-to-real gap). Use S5 to show that **the deployed pipeline runs on board**,
not as evidence of detection accuracy — the paper's accuracy numbers come from the HIT-UAV dataset.
Pick a frame where a box is on a victim, but do not crop out false boxes to make it look better.

## 4. Checklist
- [x] S1_world_overview.png
- [x] S2_victims_occlusion.png
- [x] S3_swarm_in_flight.png
- [x] S4_thermal_frame.png
- [x] S5_yolo26n_pipeline.png
- [x] S6_ros_graph.png
- [ ] S7_v1_scene.png (optional)

Plots and tables are generated automatically (`scripts/make_results.py`) — no screenshots needed for those.

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
