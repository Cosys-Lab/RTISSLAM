"""Overview of an offline RTISSLAM run: trajectories (GT, odometry, final), view links, and the occupancy grids
rendered from the final graph and from the ground-truth poses (the best this inverse sensor model can do).

    python scripts/plot_run.py office [out.pdf]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtisslam.system import RTISSLAM  # noqa: E402
from rtissim.worlds import PLANS, build_world  # noqa: E402
from sondec.paths import data_dir  # noqa: E402


def grid_img(ax, L, org, res, title):
    p = 1 / (1 + np.exp(-L))
    ax.imshow(1 - p.T, origin="lower", cmap="gray", vmin=0, vmax=1,
              extent=(org[0], org[0] + L.shape[0] * res, org[1], org[1] + L.shape[1] * res), interpolation="nearest", rasterized=True)
    ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def main():
    run = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"results/run_{run}.pdf"
    d = data_dir() / "runs" / run
    t = np.load(d / "trace.npz")
    slam = RTISSLAM.load(d / "map.npz")
    w = build_world(PLANS[run.split("_")[0]]())
    occ, (x0, y0), res = w.occupancy(0.05)
    b = (x0 - 1, x0 + occ.shape[0] * res + 1, y0 - 1, y0 + occ.shape[1] * res + 1)
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    ax = axs[0]
    ax.imshow(np.where(occ.T, 0.3, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1, extent=(x0, b[1] - 1, y0, b[3] - 1))
    gt, od, fi = t["gt"], t["odom"], t["final"]
    for q, a_ in t["links"]:
        ax.plot([fi[q, 0], fi[a_, 0]], [fi[q, 1], fi[a_, 1]], "-", c="tab:orange", lw=0.8, alpha=0.6)
    ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1, label="Ground Truth")
    ax.plot(od[:, 0], od[:, 1], "-", c="tab:red", lw=1, label="Odometry")
    ax.plot(fi[:, 0], fi[:, 1], "-", c="tab:blue", lw=1.2, label="RTISSLAM")
    ax.plot([], [], "-", c="tab:orange", label="View Links")
    ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3]); ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Trajectories: ATE Odometry {np.sqrt(np.mean(np.sum((od[:, :2] - gt[:, :2]) ** 2, 1))):.2f} m, "
                 f"RTISSLAM {np.sqrt(np.mean(np.sum((fi[:, :2] - gt[:, :2]) ** 2, 1))):.2f} m")
    L, org, r = slam.grid.render(fi, bounds=b)
    grid_img(axs[1], L, org, r, "Occupancy Grid (RTISSLAM Poses)")
    L, org, r = slam.grid.render(gt, bounds=b)
    grid_img(axs[2], L, org, r, "Occupancy Grid (Ground-Truth Poses)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(Path(out).with_suffix(".png"), dpi=70)


if __name__ == "__main__":
    main()
