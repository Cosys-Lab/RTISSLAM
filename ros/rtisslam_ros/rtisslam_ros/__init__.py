"""RTISSLAM ROS 2 nodes. The algorithms live in the project's Python packages (rtissim, rtisslam, rtisnav, batslam,
sondec), found through RTISSLAM_ROOT (the project folder). The colcon entry points run the system python3, so the
venv with torch / gtsam (RTISSLAM_VENV, e.g. ~/venvs/rtisslam) is added to sys.path here as well."""
import glob
import os
import sys

_venv = os.environ.get("RTISSLAM_VENV")
if _venv:
    for sp in glob.glob(os.path.join(os.path.expanduser(_venv), "lib", "python3*", "site-packages")):
        if sp not in sys.path:
            sys.path.insert(0, sp)
ROOT = os.environ.get("RTISSLAM_ROOT")
if ROOT and ROOT not in sys.path:
    sys.path.insert(0, ROOT)
