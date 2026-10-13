"""Set up a fresh Ubuntu 24.04 machine (NVIDIA GPU) for the interactive RTISSLAM ROS 2 demo.

    python3 tools/setup_linux.py            # run from anywhere; asks for sudo; safe to re-run
    python3 tools/demo.py --world office    # afterwards: simulator + RTISSLAM + Nav2 + RViz2

Steps (each skipped when already done):
 1. checks Ubuntu 24.04 (ROS 2 Jazzy needs it)
 2. NVIDIA driver (`ubuntu-drivers install`); after installing it, REBOOT and run this script again
 3. ROS 2 Jazzy desktop + Nav2 + slam_toolbox + colcon (official apt repository via ros2-apt-source)
 4. venv ~/venvs/rtisslam (system site packages, for rclpy) with torch (CUDA), numpy, scipy, matplotlib, imageio, gtsam
 5. ROS 2 workspace ~/ros2_ws with src/ros -> <project>/ros, colcon build --symlink-install
 6. checks: CUDA in torch, gtsam import, demo data in <project>/data/demo (runs/<world>/map.npz)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
VENV = Path.home() / "venvs" / "rtisslam"
WS = Path.home() / "ros2_ws"
DEMO_DATA = PROJECT / "data" / "demo"


def sh(cmd: str, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(["bash", "-c", cmd])
    if check and r.returncode != 0:
        sys.exit(f"failed ({r.returncode}): {cmd}")
    return r.returncode


def ok(cmd: str) -> bool:
    return subprocess.run(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    # 1. OS
    osr = Path("/etc/os-release").read_text()
    if 'VERSION_ID="24.04"' not in osr:
        sys.exit("ROS 2 Jazzy needs Ubuntu 24.04 (Noble); this is:\n" + osr)
    sh("sudo -v")

    # 2. NVIDIA driver
    if not ok("nvidia-smi"):
        sh("sudo apt-get update && sudo apt-get install -y ubuntu-drivers-common")
        sh("sudo ubuntu-drivers install")
        print("\nNVIDIA driver installed: REBOOT now, then run this script again.")
        return
    sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")

    # 3. ROS 2 Jazzy
    if not Path("/opt/ros/jazzy/setup.bash").exists():
        sh("sudo apt-get update && sudo apt-get install -y software-properties-common curl git python3-venv python3-pip")
        sh("sudo add-apt-repository -y universe")
        sh("V=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest "
           "| grep -F tag_name | awk -F'\"' '{print $4}'); "
           "curl -L -o /tmp/ros2-apt-source.deb "
           "\"https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb\" "
           "&& sudo dpkg -i /tmp/ros2-apt-source.deb")
        sh("sudo apt-get update && sudo apt-get upgrade -y")
    sh("sudo apt-get install -y ros-jazzy-desktop ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox "
       "ros-jazzy-rmw-cyclonedds-cpp python3-colcon-common-extensions python3-venv python3-pip")

    # 4. venv
    py = VENV / "bin" / "python"
    if not py.exists():
        VENV.parent.mkdir(parents=True, exist_ok=True)
        sh(f"python3 -m venv --system-site-packages {VENV}")
    sh(f"{py} -m pip install --upgrade pip")
    sh(f"{py} -m pip install torch numpy scipy matplotlib imageio imageio-ffmpeg gtsam")
    sh(f"{py} -c \"import torch, gtsam; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), "
       f"torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); print('gtsam ok')\"")

    # 5. workspace
    (WS / "src").mkdir(parents=True, exist_ok=True)
    link = WS / "src" / "ros"
    if link.is_symlink() and link.resolve() != (PROJECT / "ros").resolve():
        link.unlink()
    if not link.exists():
        link.symlink_to(PROJECT / "ros")
    sh(f"source /opt/ros/jazzy/setup.bash && source {VENV}/bin/activate && cd {WS} && colcon build --symlink-install")

    # 6. data
    need = [DEMO_DATA / "runs" / w / "map.npz"
                                                        for w in ("office", "warehouse", "atrium")]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        print("\nmissing demo data (unzip rtisslam_demo_data.zip into <project>/data/, see README \"Getting started\"):\n  " + "\n  ".join(missing))
    print(f"\nDone. Start the demo with:\n  python3 {PROJECT / 'tools' / 'demo.py'} --world office")


if __name__ == "__main__":
    if shutil.which("bash") is None:
        sys.exit("needs bash")
    os.chdir(Path.home())
    main()
