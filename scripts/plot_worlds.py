"""Clean overview figure of the floor plans: occupancy (walls and furniture footprints), reflector points, concave
corners, door frames, point objects, and the mapping drive's driving line.

    python scripts/plot_worlds.py [out.pdf] [plan ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtissim.worlds import PLANS, build_world, concave_corners, driving_line  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/worlds.pdf"
    names = sys.argv[2:] or ["office", "warehouse", "atrium"]
    fig, axs = plt.subplots(1, len(names), figsize=(8 * len(names), 6.5), squeeze=False)
    for ax, k in zip(axs[0], names):
        p = PLANS[k]()
        w = build_world(p)
        a = w.arrays()
        occ, (x0, y0), res = w.occupancy(0.05)
        ext = (x0, x0 + occ.shape[0] * res, y0, y0 + occ.shape[1] * res)
        ax.imshow(np.where(occ.T, 0.25, 1.0), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest", rasterized=True)
        z = a["pos"][:, 2]
        sel = (z > 0.1) & (z < 0.9)
        ax.scatter(a["pos"][sel, 0], a["pos"][sel, 1], s=0.15, c="tab:blue", alpha=0.25, lw=0, rasterized=True,
                   label="Reflector Points")
        path = driving_line(p, w)
        ax.plot(*np.vstack([path, path[:1]]).T, "-", c="tab:green", lw=1.6, label="Mapping Drive")
        ax.plot(*path[0], "o", c="tab:green", ms=7, mec="k")
        c = np.array(concave_corners(p, w))
        ax.plot(c[:, 0], c[:, 1], "^", c="tab:red", ms=5, label="Concave Corners")
        if p.frames:
            f = np.array(p.frames)
            ax.plot(f[:, 0], f[:, 1], "s", c="tab:orange", ms=2.5, label="Door Frames")
        o = np.array(p.objects)
        ax.plot(o[:, 0], o[:, 1], "o", c="tab:purple", ms=5, label="Point Objects")
        L = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        ax.set_title(f"{k.title()}: {ext[1] - ext[0]:.0f} x {ext[3] - ext[2]:.0f} m, Drive {L:.0f} m, "
                     f"{len(a['pos']) / 1e3:.0f}k Reflector Points")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    axs[0, 0].legend(fontsize=8, loc="upper left", markerscale=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
