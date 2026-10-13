# RTISSLAM

Sonar-only SLAM, localization and navigation for a mobile robot with an embedded 3D imaging sonar (eRTIS: one emitter,
32 MEMS microphones). The repository contains

- a closed-loop 2D simulator of the robot and the sonar (specular walls, drifting wheel odometry), imaging with
  DMAS3 + coherence factor (DMAS3-CF);
- RTISSLAM: the BatSLAM 2.0 pose graph (place recognition on sonar views, sequence verification, robust iSAM2
  back-end) extended with an occupancy grid anchored to the pose-graph nodes, acoustic odometry that calibrates the
  wheel odometry online, and metric registration of recognized revisits;
- localization in a saved map (particle filter, Gaussian mixture of EKFs, or the pose graph itself);
- navigation: global planning on the RTISSLAM grid (Nav2 planner) and the layered acoustic-flow controller;
- ROS 2 Jazzy packages that run the same way against the simulator and against the real robot and sensor;
- the scripts that produce the paper's results, figures and video, and a lidar-style grid-SLAM baseline
  (slam_toolbox on a sonar pseudo-scan).

## Layout

| Path | What |
|---|---|
| `rtissim/` | Simulator: floor plans (`worlds.py`: office, warehouse, atrium + small test plans), robot with drifting odometry and pure pursuit (`robot.py`), the eRTIS on the robot (`sensor.py`), closed-loop `sim.py` |
| `rtisslam/` | RTISSLAM: energyscape descriptor, keyframe occupancy grid, acoustic odometry, link registration, localizers (`localize.py`), the system wrapper with map save / load (`system.py`) |
| `rtisnav/` | Navigation: planning grid from the RTISSLAM map (`costmap.py`), layered acoustic-flow controller (`afc.py`), `navigator.py` |
| `batslam/` | BatSLAM 2.0 pose graph, templates and sequence verifier, with small marked extensions |
| `sondec/` | eRTIS array model, echo synthesis and DAS / DMAS imaging operators |
| `ros/` | ROS 2 Jazzy packages: `rtis_msgs` (messages), `rtisslam_ros` (simulator, imaging, RTISSLAM, navigator, visualization nodes), `rtisslam_bringup` (demo / sim / real launch files, Nav2 planner, slam_toolbox baseline and RViz2 configs) |
| `scripts/` | Drives, SLAM runs, evaluations, baseline, tables and figures |
| `tools/` | Machine setup, interactive demo, demo recording |
| `video/` | Video scripts |
| `tests/` | Unit tests and an end-to-end smoke test |

## Installation

Ubuntu 24.04 with an NVIDIA GPU is recommended: the simulator and the DMAS3-CF imaging run in tens of milliseconds
per pulse on a GPU and take seconds on a CPU (everything also runs on a CPU, just slower).

Automatic (driver, ROS 2 Jazzy with Nav2 and slam_toolbox, a venv with torch and gtsam, the colcon workspace;
asks for sudo, safe to re-run): `python3 tools/setup_linux.py`.

By hand:

1. ROS 2 Jazzy: `sudo apt install ros-jazzy-desktop ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox
   python3-colcon-common-extensions`.
2. Python: `python3 -m venv --system-site-packages ~/venvs/rtisslam && ~/venvs/rtisslam/bin/pip install -r
   requirements.txt` (install torch with CUDA as described on pytorch.org; check
   `python -c "import torch; print(torch.cuda.is_available())"`).
3. ROS 2 workspace: `mkdir -p ~/ros2_ws/src && ln -s $PWD/ros ~/ros2_ws/src/ros && cd ~/ros2_ws &&
   source /opt/ros/jazzy/setup.bash && source ~/venvs/rtisslam/bin/activate && colcon build --symlink-install`.
4. Environment for the ROS nodes: `export RTISSLAM_ROOT=<this repository> RTISSLAM_DATA=<data directory>
   RTISSLAM_VENV=~/venvs/rtisslam` (the nodes add the venv to their path).

Data (drives, maps, runs, videos) live outside the repository in `RTISSLAM_DATA` (default `~/RTISSLAM_data`).

## Quick start

- Smoke test (simulates a short drive, builds a map, saves and loads it, localizes in it; seconds on a GPU, a few
  minutes on a CPU): `~/venvs/rtisslam/bin/python -m pytest tests/`.
- Interactive demo (simulator + RTISSLAM localizer + Nav2 planner + acoustic-flow navigator + RViz2), given a saved
  map in `data/demo/runs/<world>/map.npz`: `python3 tools/demo.py --world office`. Give goals with "2D Goal Pose" in
  RViz2; Ctrl+C stops everything. Options: `--localizer pf|mix|graph` (default `pf`), `--raw` (raw microphone
  signals imaged by `rtis_imaging_node`, as on the robot), `--rtf` (simulation speed). Diagnostics:
  `/rtisslam/status` (JSON per pulse), `/rtisslam/candidates` and `/rtisslam/hypotheses` (RViz2), `/rtis_nav/layer`
  (active controller layer).
- Recording the demo with scripted goals on a virtual display: `python3 tools/record_demo.py --goals 3 --rtf 2`.

## Reproducing the paper

All runs use DMAS3-CF (`dmas3cf:ams2s`) for every role: descriptor, registration, acoustic odometry, grid, controller.

```
# drives (energyscapes stored per pulse): the development drive (1 lap, office) and the evaluation drives (2.1 laps)
python scripts/gen_drive.py office --methods dmas3cf --laps 1.15 --tag map
python scripts/gen_drive.py <world> --methods dmas3cf --laps 2.1 --tag map2 --seed 0      # and --tag map2s<k> --seed <k>, k = 1..4
# descriptor tuning and evidence curve, on the development drive only
python scripts/tune_descriptor.py --imaging dmas3cf --posts ams2s --grid n_sectors=15,20,30,60 gamma=0.2,0.33
python scripts/fit_evidence.py office --variants dmas3cf:ams2s                            # -> defaults in rtisslam/system.py
# RTISSLAM (5 seeds x 3 worlds, final configuration and ablation) and the results table
python scripts/run_final.py
python scripts/table_seeds.py
# baseline: slam_toolbox on the sonar pseudo-scan (tuned on the development drive only; --stock for its defaults)
python scripts/export_scans.py <world> --tag map2                                         # and map2s1..map2s4
python3 scripts/baseline_slam_toolbox.py <world>_map2            # source /opt/ros/jazzy/setup.bash first
# localization curve (a second drive per world) and navigation on the saved maps
python scripts/gen_drive.py <world> --methods dmas3cf --laps 1.15 --seed 1 --tag loc
python scripts/fit_evidence.py <world> --tag map2 --variants dmas3cf:ams2s --query-tag loc
python scripts/with_loc_curve.py <world>_dm_final_s0 --evidence <world>_map2              # -> runs/<world>_dm_final_s0_loc
python scripts/run_navigation.py <world> --map <world>_dm_final_s0_loc --localizer pf --tag nav_dm
# figures and video
python scripts/paper_figures.py all
python video/make_combined_video.py
```

## Real robot

`ros2 launch rtisslam_bringup real.launch.py mode:=mapping|localization ...` runs the same nodes against the robot's
driver, the eRTIS driver (via an adapter to `rtis_msgs/RtisPulse`) and a lidar whose slam_toolbox pose is recorded as
the reference. `scripts/replay_bag.py <bag> --map <map.npz> --localizer pf --ref-topic /pose` localizes offline in a
recorded bag and reports the error after rigid alignment to the reference.

## License

BSD 3-Clause, see `LICENSE`.

## Citation

If you use this code, please cite the paper (preprint on arXiv; reference to be added).
