"""The five imaging methods at a few robot poses of a plan, as Cartesian fans (range-compensated, 40 log10 r),
with the visible ground-truth reflectors overlaid (EVALUATION ONLY).

    python scripts/plot_energyscapes.py [out.pdf] [plan] [x,y,yaw_deg ...]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtissim.sensor import METHODS, RtisSensor  # noqa: E402
from rtissim.worlds import PLANS, build_world  # noqa: E402

LABELS = {"das": "DAS", "dmas2": "DMAS", "dmas2cf": "DMAS + CF", "dmas3cf": "DMAS3 + CF", "deconv": "Deconvolution (SPL)"}


def fan(ax, img_db, r, az, dr_db=30.0):
    """img [R, D] dB -> pcolormesh in the sensor frame (x forward, y left), range-compensated, top dr_db shown."""
    v = img_db + 40 * np.log10(r)[:, None]
    v = v - np.percentile(v, 99.95)
    re = np.r_[r - (r[1] - r[0]) / 2, r[-1] + (r[1] - r[0]) / 2]
    ae = np.r_[az - (az[1] - az[0]) / 2, az[-1] + (az[1] - az[0]) / 2]
    A, R = np.meshgrid(ae, re)
    return ax.pcolormesh(R * np.cos(A), R * np.sin(A), v, vmin=-dr_db, vmax=0, cmap="inferno", shading="flat",
                         rasterized=True)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/energyscapes.pdf"
    plan = sys.argv[2] if len(sys.argv) > 2 else "office"
    poses = [tuple(float(v) for v in s.split(",")) for s in sys.argv[3:]] or [(4.0, 6.7, 0.0), (16.0, 3.8, -90.0), (33.5, 11.0, 90.0)]
    p = PLANS[plan]()
    w = build_world(p)
    sen = RtisSensor(w.arrays(), device="cuda")
    r, az = sen.ranges, sen.azimuths
    fig, axs = plt.subplots(len(poses), len(METHODS) + 1, figsize=(3.3 * (len(METHODS) + 1), 3.0 * len(poses)), squeeze=False)
    occ, (x0, y0), res = w.occupancy(0.05)
    for row, (x, y, yd) in zip(axs, poses):
        pose = (x, y, math.radians(yd))
        maps, contrib = sen.render(pose)
        ax = row[0]
        ax.imshow(np.where(occ.T, 0.25, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1,
                  extent=(x0, x0 + occ.shape[0] * res, y0, y0 + occ.shape[1] * res))
        ax.plot(x, y, "o", c="tab:green"); ax.arrow(x, y, 0.8 * math.cos(pose[2]), 0.8 * math.sin(pose[2]), width=0.08, color="tab:green")
        ax.set_xlim(x - 5.5, x + 5.5); ax.set_ylim(y - 5.5, y + 5.5); ax.set_aspect("equal")
        ax.set_title(f"Pose ({x:.1f}, {y:.1f}, {yd:.0f} deg)", fontsize=9)
        u, rr = contrib["u"], contrib["r"]
        for ax, m in zip(row[1:], METHODS):
            fan(ax, maps[m], r, az)
            ax.plot(rr * u[:, 0], rr * u[:, 1], "o", mfc="none", mec="tab:cyan", ms=4, mew=0.6)
            ax.set_xlim(0, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
            ax.set_title(LABELS[m], fontsize=9); ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(Path(out).with_suffix(".png"), dpi=80)


if __name__ == "__main__":
    main()
