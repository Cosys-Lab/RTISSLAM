"""Occupancy grids of a drive from the ground-truth poses for imaging methods with and without RW-CFAR (range-wise CFAR): the effect of the post-processing on the map alone (no SLAM involved).

    python scripts/plot_grids_post.py office [out.pdf]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtisslam.grid import GridConfig, KeyframeGrid  # noqa: E402
from rtisslam.imaging_post import POST  # noqa: E402
from rtissim.worlds import PLANS, build_world  # noqa: E402
from sondec.paths import data_dir  # noqa: E402


def main():
    w = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"results/grids_post_{w}.pdf"
    f = np.load(data_dir() / "drives" / f"{w}_map.npz")
    gt, odo = f["gt"], f["odom"]
    occ, (x0, y0), res = build_world(PLANS[w]()).occupancy(0.05)
    b = (x0 - 1, x0 + occ.shape[0] * res + 1, y0 - 1, y0 + occ.shape[1] * res + 1)
    methods, posts = [("das", "DAS"), ("dmas3cf", "DMAS3 + CF"), ("deconv", "Deconvolution")], [("none", "No RW-CFAR"), ("ams", "RW-CFAR (Gain 1)"), ("ams15", "RW-CFAR (Gain 1.5)"), ("ams2", "RW-CFAR (Gain 2)"), ("ams25", "RW-CFAR (Gain 2.5)"), ("ams2s", "RW-CFAR (Gain 2, Smoothed)"), ("cfar2d2", "2D CA-CFAR (Gain 2)")]
    fig, axs = plt.subplots(len(posts), len(methods), figsize=(7 * len(methods), 4.6 * len(posts)))
    for c, (m, ml) in enumerate(methods):
        E = f[f"{m}_db"]
        for r, (po, pl) in enumerate(posts):
            g = KeyframeGrid(replace(GridConfig(), method=m), f["ranges"], f["azimuths"])
            for i in range(len(gt)):
                if g.want_keyframe(odo[i]):
                    g.add(i, odo[i], POST[po](E[i]))
            L, org, rs = g.render(gt, bounds=b)
            ax = axs[r, c]
            ax.imshow(1 - 1 / (1 + np.exp(-L.T)), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                      extent=(org[0], org[0] + L.shape[0] * rs, org[1], org[1] + L.shape[1] * rs), rasterized=True)
            ax.set_title(f"{ml}, {pl}"); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(Path(out).with_suffix(".png"), dpi=55)


if __name__ == "__main__":
    main()
