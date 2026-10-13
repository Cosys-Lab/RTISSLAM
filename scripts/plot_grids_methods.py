"""Occupancy grids of one drive for each imaging method (same inverse sensor model and thresholds), rendered from the
ground-truth poses (top row: what each method can do) and from the RTISSLAM poses of the run (bottom row).

    python scripts/plot_grids_methods.py office [out.pdf]
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
from rtissim.worlds import PLANS, build_world  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

METHODS = [("das", "DAS"), ("dmas2", "DMAS"), ("dmas3cf", "DMAS3 + CF"), ("deconv", "Deconvolution (SPL)")]


def main():
    run = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"results/grids_methods_{run}.pdf"
    f = np.load(data_dir() / "drives" / f"{run}_map.npz")
    t = np.load(data_dir() / "runs" / run / "trace.npz")
    gt, fi, odo = f["gt"], t["final"], f["odom"]
    w = build_world(PLANS[run]())
    occ, (x0, y0), res = w.occupancy(0.05)
    b = (x0 - 1, x0 + occ.shape[0] * res + 1, y0 - 1, y0 + occ.shape[1] * res + 1)
    fig, axs = plt.subplots(2, len(METHODS), figsize=(7.5 * len(METHODS), 10.5))
    for col, (m, lab) in enumerate(METHODS):
        g = KeyframeGrid(replace(GridConfig(), method=m), f["ranges"], f["azimuths"])
        E = f[f"{m}_db"]
        for i in range(len(gt)):
            if g.want_keyframe(odo[i]):
                g.add(i, odo[i], E[i].astype(np.float32))
        for row, (P, nm) in enumerate(((gt, "Ground-Truth Poses"), (fi, "RTISSLAM Poses"))):
            L, org, r = g.render(P, bounds=b)
            p = 1 / (1 + np.exp(-L))
            ax = axs[row, col]
            ax.imshow(1 - p.T, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest", rasterized=True,
                      extent=(org[0], org[0] + L.shape[0] * r, org[1], org[1] + L.shape[1] * r))
            ax.set_title(f"{lab}, {nm}"); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        print(m, "keyframes", len(g.keyframes), "occupied pts / kf", np.mean([len(k.occ) for k in g.keyframes]).round(1))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(Path(out).with_suffix(".png"), dpi=60)


if __name__ == "__main__":
    main()
