"""Overview of a navigation run: planner costmap (lethal cells), all global paths, ground truth and estimate,
goals, and where the acoustic-flow layers were active.

    python scripts/plot_nav.py office_nav [out.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sondec.paths import data_dir  # noqa: E402


def main():
    run = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"results/nav_{run}.png"
    f = np.load(data_dir() / "nav" / f"{run}.npz")
    L, org, res = f["grid_L"], f["grid_org"], float(f["grid_res"])
    fig, ax = plt.subplots(figsize=(16, 10))
    ext = (org[0], org[0] + L.shape[0] * res, org[1], org[1] + L.shape[1] * res)
    ax.imshow(1 - 1 / (1 + np.exp(-L.T)), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    P = f["paths"]
    ax.plot(P[:, 0], P[:, 1], ".", c="tab:green", ms=1, alpha=0.4)
    gt, est = f["gt"], f["est"]
    ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1, label="Ground Truth")
    ax.plot(est[:, 0], est[:, 1], "-", c="m", lw=1, label="Estimate")
    lay = f["layer"]
    for L_, c in (("CA", "red"), ("OA", "orange")):
        m = lay == L_
        ax.plot(gt[m, 0], gt[m, 1], "o", c=c, ms=2, label=L_)
    G = f["goals"]
    ax.plot(G[:, 0], G[:, 1], "*", c="tab:red", ms=16)
    for i, g in enumerate(G):
        ax.text(g[0] + 0.3, g[1] + 0.3, str(i + 1), fontsize=14, color="tab:red")
    ax.legend(); ax.set_aspect("equal")
    fig.savefig(out, dpi=60, bbox_inches="tight")


if __name__ == "__main__":
    main()
