"""Navigation figure for the report: per world, the RTISSLAM map (the planning grid's source), the goals in visiting
order, the ground-truth trajectory and the RTISSLAM estimate, and where the collision / obstacle avoidance layers of
the acoustic-flow controller were active.

    python scripts/plot_nav_worlds.py [out.pdf] [run ...]        (default runs: office_nav warehouse_nav atrium_nav)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sondec.paths import data_dir  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/nav_worlds.pdf"
    runs = sys.argv[2:] or ["office_nav", "warehouse_nav", "atrium_nav"]
    fig, axs = plt.subplots(1, len(runs), figsize=(8 * len(runs), 6.5), squeeze=False)
    for ax, run in zip(axs[0], runs):
        f = np.load(data_dir() / "nav" / f"{run}.npz")
        m = json.loads((data_dir() / "nav" / f"{run}.json").read_text())
        L, org, res = f["grid_L"], f["grid_org"], float(f["grid_res"])
        ext = (org[0], org[0] + L.shape[0] * res, org[1], org[1] + L.shape[1] * res)
        ax.imshow(1 - 1 / (1 + np.exp(-L.T)), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest", rasterized=True)
        gt, est, lay = f["gt"], f["est"], f["layer"]
        ax.plot(gt[:, 0], gt[:, 1], "-", c="k", lw=1.0, label="Ground Truth")
        ax.plot(est[:, 0], est[:, 1], "-", c=(0.69, 0, 0.57), lw=1.0, label="RTISSLAM Estimate")
        for name, c in (("CA", "tab:red"), ("OA", "tab:orange")):
            sel = lay == name
            ax.plot(gt[sel, 0], gt[sel, 1], "o", c=c, ms=1.5, label={"CA": "Collision Avoidance", "OA": "Obstacle Avoidance"}[name])
        G = f["goals"]
        for i, (g, r) in enumerate(zip(G, m["results"])):
            ok = r["state"] == "reached"
            ax.plot(*g, "*", ms=15, mec="k", mew=0.6, c="tab:green" if ok else "tab:red")
            ax.text(g[0] + 0.35, g[1] + 0.35, str(i + 1), fontsize=11)
        ax.plot(*gt[0, :2], "o", c="tab:blue", ms=8, mec="k", label="Start")
        ax.set_title(f"{m['world'].title()}: {m['reached']} of {m['goals']} Goals, {m['distance_m']:.0f} m, "
                     f"{m.get('contact_events', m['bumps'])} Contacts, Loc. Error {m['loc_err_mean_m']:.2f} m")
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    axs[0, 0].legend(fontsize=8, loc="upper left", markerscale=3, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".png"), dpi=70, bbox_inches="tight")


if __name__ == "__main__":
    main()
