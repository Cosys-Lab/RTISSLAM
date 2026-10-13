"""Export a drive as a compact sonar pseudo-laser scan for the grid-SLAM baseline (slam_toolbox / Karto).

The scan is exactly the echo detection of the RTISSLAM occupancy grid (rtisslam/grid.py, KeyframeGrid.evidence): per
azimuth, the range of the first detection in the (post-processed) energyscape, inf where there is none; 181 beams
over +-90 deg, in the sensor frame (mount `mount` in base_link), max range 4.5 m.

    python scripts/export_scans.py office --tag map2 [--imaging dmas3cf --post ams2s]
Writes <project>/data/baseline/<world>_<tag>_scan.npz (not in git): t [N], gt [N, 3], odom [N, 3] (wheel
odometry, odom frame), ranges [N, 181] (m, inf = no return), angles [181] (rad, + = left), mount (x, y), range_max.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtisslam.grid import GridConfig, KeyframeGrid  # noqa: E402
from rtisslam.imaging_post import POST  # noqa: E402
from sondec.paths import data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--tag", default="map2")
    ap.add_argument("--imaging", default="dmas3cf")
    ap.add_argument("--post", default="ams2s")
    a = ap.parse_args()
    f = np.load(data_dir() / "drives" / f"{a.world}_{a.tag}.npz")
    g = KeyframeGrid(replace(GridConfig(), method=f"{a.imaging}:{a.post}"), f["ranges"], f["azimuths"])
    E, post = f[f"{a.imaging}_db"], POST[a.post]
    c = g.cfg
    keep = g.keep
    r = g.r[keep]
    R = np.full((len(E), len(g.az)), np.inf, np.float32)
    for i, es in enumerate(E):
        v = (np.asarray(post(es), np.float32) + g.tvg)[keep]
        thr = max(np.median(v) + c.det_db, v.max() - c.peak_db)
        ri, ai = np.nonzero(v > thr)
        if len(ri):
            first = np.full(len(g.az), np.inf)
            np.minimum.at(first, ai, r[ri])
            R[i] = first
    out = ROOT / "data" / "baseline" / f"{a.world}_{a.tag}_scan.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t=f["t"], gt=f["gt"], odom=f["odom"], ranges=R, angles=np.asarray(g.az, np.float32),
                        mount=np.asarray(c.mount, np.float32), range_max=np.float32(c.max_range),
                        energyscape=f"{a.imaging}:{a.post}")
    n = np.isfinite(R).sum(1)
    print(f"{out}: {len(R)} scans, returns per scan median {np.median(n):.0f} (min {n.min()}, max {n.max()})")


if __name__ == "__main__":
    main()
