"""Start the interactive RTISSLAM ROS 2 demo (after tools/setup_linux.py): the eRTIS simulator on the GPU, RTISSLAM
localizing in a saved map, Nav2 global planning, the acoustic-flow navigator and RViz2. Give goals with the
"2D Goal Pose" tool in RViz2; stop with Ctrl+C.

    python3 tools/demo.py [--world office|warehouse|atrium] [--device cuda] [--data <dir>] [--rtf 3] [--localizer mix|pf|graph] [--raw]

The map and its energyscape keys are read from <data>/runs/<world>/map.npz (default data: <project>/data/demo).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def map_keys(map_file: Path) -> list[str]:
    """Energyscape keys ("imaging:post") the map was built with, read from its stored config."""
    import numpy as np
    with np.load(map_file) as z:
        cfg = json.loads(str(z["config"]))
    keys = {cfg["descriptor"]["method"], cfg["grid"]["method"], cfg["acoustic_odometry"]["method"]}
    keys |= {cfg.get("register_method", "")} - {""}
    return sorted(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="office")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default=str(PROJECT / "data" / "demo"))
    ap.add_argument("--venv", default=str(Path.home() / "venvs" / "rtisslam"))
    ap.add_argument("--rtf", type=float, default=1.0, help="simulation speed vs wall clock (0 = as fast as possible)")
    ap.add_argument("--localizer", default="pf", choices=["graph", "mix", "pf"],
                    help="pf: particle filter (default; best in closed loop), mix: Gaussian mixture of EKFs, graph: BatSLAM pose graph")
    ap.add_argument("--raw", action="store_true", help="real-sensor data path: raw RtisPulse + rtis_imaging_node")
    ap.add_argument("--afc-method", default="dmas3cf:ams2s", help="energyscape of the acoustic-flow controller")
    ap.add_argument("--no-rviz", action="store_true")
    ap.add_argument("--rviz-config", default="", help="RViz2 config file (default: the package's rtisslam.rviz)")
    a = ap.parse_args()
    data = Path(a.data)
    map_file = data / "runs" / a.world / "map.npz"
    keys = map_keys(map_file)
    with __import__("numpy").load(map_file) as z:
        viz = json.loads(str(z["config"]))["grid"]["method"]            # show the energyscape the grid is built from
    methods = ",".join(sorted(set(keys) | {a.afc_method}))
    env = dict(os.environ, RTISSLAM_ROOT=str(PROJECT), RTISSLAM_DATA=str(data), RTISSLAM_VENV=a.venv)
    cmd = (f"source /opt/ros/jazzy/setup.bash && source {Path.home() / 'ros2_ws' / 'install' / 'setup.bash'} && "
           f"ros2 launch rtisslam_bringup demo.launch.py world:={a.world} device:={a.device} "
           f"map_file:='{map_file}' methods:={methods} viz_method:={viz} real_time_factor:={a.rtf} localizer:={a.localizer} raw:={str(a.raw).lower()} afc_method:={a.afc_method} rviz_config:='{a.rviz_config}' rviz:={'false' if a.no_rviz else 'true'}")
    print(cmd, flush=True)
    subprocess.run(["bash", "-c", cmd], env=env)


if __name__ == "__main__":
    main()
