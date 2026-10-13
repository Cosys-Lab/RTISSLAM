"""Floor plans, driving lines, concave corners, objects, and one pure-pursuit drive (true vs odometry) per plan.

    python scripts/plot_plans.py [out.pdf] [plan ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtissim.robot import DiffDrive, PurePursuit  # noqa: E402
from rtissim.worlds import PLANS, build_world, concave_corners, driving_line  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "plans.pdf"
    names = sys.argv[2:] or list(PLANS)
    fig, axs = plt.subplots(1, len(names), figsize=(7 * len(names), 6), squeeze=False)
    axs = axs[0]
    for ax, k in zip(axs, names):
        p = PLANS[k]()
        w = build_world(p)
        a = w.arrays()
        occ, (x0, y0), res = w.occupancy(0.05)
        ax.imshow(occ.T, origin="lower", extent=(x0, x0 + occ.shape[0] * res, y0, y0 + occ.shape[1] * res), cmap="Greys")
        ax.plot(a["pos"][:, 0], a["pos"][:, 1], ",", c="tab:blue", alpha=0.3, rasterized=True)
        path = driving_line(p, w)
        ax.plot(path[:, 0], path[:, 1], "g-", lw=2, alpha=0.5, label="Driving Line")
        c = np.array(concave_corners(p))
        ax.plot(c[:, 0], c[:, 1], "r^", label="Concave Corners")
        o = np.array(p.objects)
        ax.plot(o[:, 0], o[:, 1], "mo", label="Objects")
        pp = PurePursuit(path)
        th0 = np.arctan2(*(path[1] - path[0])[::-1])
        r = DiffDrive(pose=np.r_[path[0], th0])
        T, O = [], []
        for _ in range(int(len(path) * 0.05 / 0.3 / 0.1)):                 # about one lap
            v, wz = pp.command(r.pose)
            r.step(v, wz, 0.1)
            T.append(r.pose.copy()); O.append(r.odom.copy())
        T, O = np.array(T), np.array(O)
        ax.plot(T[:, 0], T[:, 1], "k--", lw=0.8, label="True")
        ax.plot(O[:, 0], O[:, 1], "c-", lw=0.8, label="Odometry")
        ax.set_title(f"{k.title()}: {len(T)} Steps, Odometry Drift {np.hypot(*(T[-1, :2] - O[-1, :2])):.2f} m")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.savefig(out, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
