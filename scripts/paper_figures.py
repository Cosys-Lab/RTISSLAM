"""Figures of the RTISSLAM paper (vector PDF, image data rasterized; capitalized titles and labels).

    python scripts/paper_figures.py [slam|loop|ao|all]
Writes to paper/figures/:
    slamWorlds.pdf    per world: trajectories (ground truth, wheel odometry, RTISSLAM, view links) and the occupancy
                      grid rendered from the final pose graph
    loopClosure.pdf   the office grid just before and just after the largest loop closure of the online run
    acousticOdometry.pdf   a) convergence of the wheel calibration, b) per-step error of acoustic and wheel
                      odometry, c) dead reckoning with and without the calibration
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtisslam.grid import KeyframeGrid  # noqa: E402
from rtisslam.system import RTISSLAMConfig  # noqa: E402
from rtissim.worlds import PLANS, build_world  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

OUT = ROOT / "paper" / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5})
WORLDS = ["office", "warehouse", "atrium"]
RUN = "{}_dm_final_s0"   # final runs (scripts/run_final.py, seed 0), on the 2-lap mapping drives
TAG = "map2"


def run_dir(w: str) -> Path:
    return data_dir() / "runs" / RUN.format(w)


def load_grid(run: str) -> KeyframeGrid:
    with np.load(run_dir(run) / "map.npz") as z:
        d = {k: z[k] for k in z.files}
    g = KeyframeGrid(RTISSLAMConfig.from_dict(json.loads(str(d["config"]))).grid, d["ranges"], d["azimuths"])
    g.load_state(d)
    return g


def bounds_of(world):
    occ, (x0, y0), res = build_world(PLANS[world]()).occupancy(0.05)
    return occ, (x0, y0), res, (x0 - 0.8, x0 + occ.shape[0] * res + 0.8, y0 - 0.8, y0 + occ.shape[1] * res + 0.8)


BOX = (39.6, 27.6)            # common panel extent (m): every world centered in the same box, so all panels match


def box_of(world):
    """Limits of a BOX-sized window centered on the world."""
    occ, (x0, y0), res = build_world(PLANS[world]()).occupancy(0.05)
    cx, cy = x0 + occ.shape[0] * res / 2, y0 + occ.shape[1] * res / 2
    return cx - BOX[0] / 2, cx + BOX[0] / 2, cy - BOX[1] / 2, cy + BOX[1] / 2


def show_grid(ax, L, org, res):
    ax.imshow(1 - 1 / (1 + np.exp(-L.T)), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest",
              extent=(org[0], org[0] + L.shape[0] * res, org[1], org[1] + L.shape[1] * res), rasterized=True)


def ate(a, b):
    return float(np.sqrt(np.mean(np.sum((a[:, :2] - b[:, :2]) ** 2, 1))))


def show_baseline_grid(ax, z):
    """slam_toolbox OccupancyGrid (-1 unknown, 0 free, 100 occupied) in its map frame (= the start pose) -> world."""
    from matplotlib.transforms import Affine2D
    g = z["grid"].astype(float)
    img = np.where(g < 0, 0.5, 1 - g / 100.0)
    x0, y0, th0 = z["gt"][0]
    ox, oy = z["grid_origin"]; r = float(z["grid_resolution"])
    tr = Affine2D().rotate(th0).translate(x0, y0) + ax.transData
    ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest", rasterized=True,
              extent=(ox, ox + g.shape[1] * r, oy, oy + g.shape[0] * r), transform=tr)


def fig_slam():
    """Per world (2 laps, seed 0): trajectories; the RTISSLAM grid with its pose graph; the slam_toolbox grid with its
    trajectory. All panels share the same extent."""
    fig, axs = plt.subplots(3, 3, figsize=(7.16, 5.05))
    for c, w in enumerate(WORLDS):
        t = np.load(run_dir(w) / "trace.npz")
        bz = np.load(ROOT / "data" / "baseline" / f"{w}_{TAG}_slamtoolbox.npz", allow_pickle=True)
        occ, (x0, y0), res, b = bounds_of(w)
        bx = box_of(w)
        gt, od, fi, links = t["gt"], t["odom"], t["final"], t["links"]
        bl = bz["est_final"]
        ax = axs[0, c]
        ax.imshow(np.where(occ.T, 0.78, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                  extent=(x0, x0 + occ.shape[0] * res, y0, y0 + occ.shape[1] * res), rasterized=True)
        ax.plot(od[:, 0], od[:, 1], "-", c="tab:red", lw=0.5, label=f"Wheel Odometry ({ate(od, gt):.2f} m)")
        ax.plot(bl[:, 0], bl[:, 1], "-", c="tab:green", lw=0.5, label=f"slam_toolbox ({ate(bl, bz['gt']):.2f} m)")
        ax.plot(gt[:, 0], gt[:, 1], "-", c="k", lw=0.6, label="Ground Truth")
        ax.plot(fi[:, 0], fi[:, 1], "-", c="tab:blue", lw=0.7, label=f"RTISSLAM ({ate(fi, gt):.2f} m)")
        ax.plot([], [], "o", c=(1.0, 0.55, 0.0), ms=2.2, mew=0.3, mec="k", label="Recognized Revisits")
        ax.set_title(w.title(), fontsize=9)
        ax.legend(loc="upper left", fontsize=5, framealpha=0.9, handlelength=1.0, borderpad=0.3, labelspacing=0.15)
        g = load_grid(w)
        L, org, r = g.render(fi, bounds=b)
        ax = axs[1, c]
        ax.set_facecolor((0.5, 0.5, 0.5))
        show_grid(ax, L, org, r)
        # the pose graph: the estimate, its template nodes, and the nodes where a run of loop closures starts (a
        # recognized revisit, i.e. the first query node of a run of consecutive linked nodes)
        ax.plot(fi[:, 0], fi[:, 1], "-", c="tab:blue", lw=0.4)
        tp = t["templates"]
        ax.plot(fi[tp, 0], fi[tp, 1], "o", c="tab:blue", ms=0.8, mew=0)
        q = np.unique(links[:, 0]) if len(links) else np.zeros(0, int)
        starts = q[np.r_[True, np.diff(q) > 3]] if len(q) else q
        ax.plot(fi[starts, 0], fi[starts, 1], "o", c=(1.0, 0.55, 0.0), ms=2.2, mew=0.3, mec="k")
        ax = axs[2, c]
        ax.set_facecolor((0.5, 0.5, 0.5))
        show_baseline_grid(ax, bz)
        ax.plot(bl[:, 0], bl[:, 1], "-", c="tab:green", lw=0.45)
        for row, ax in enumerate(axs[:, c]):
            ax.set_xlim(bx[0], bx[1]); ax.set_ylim(bx[2], bx[3]); ax.set_aspect("equal")
            ax.tick_params(labelsize=6, length=2, pad=1)
            if row < 2:
                ax.set_xticklabels([])
        axs[2, c].set_xlabel("x (m)", fontsize=7)
    for row, lab in enumerate(("Trajectories", "RTISSLAM", "slam_toolbox")):
        axs[row, 0].set_ylabel(lab + ", y (m)", fontsize=7)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4)
    fig.savefig(OUT / "slamWorlds.pdf", dpi=300)


def fig_loop():
    w = "office"
    t = np.load(run_dir(w) / "trace.npz")
    S, at = t["snaps"], t["snap_at"]
    gt = t["gt"]
    occ, (x0, y0), res, b = bounds_of(w)
    # the snapshot pair with the largest change of the already existing nodes
    # the loop closure that reduces the error of the current node the most
    best, kb, gain = 0.0, 1, 0.0
    for k in range(1, len(at)):
        n0, n = int(at[k - 1]) + 1, int(at[k]) + 1
        e0 = math.hypot(*(S[k - 1, n0 - 1, :2] - gt[n0 - 1, :2]))
        e1 = math.hypot(*(S[k, n - 1, :2] - gt[n - 1, :2]))
        if e0 - e1 > gain:
            gain, kb = e0 - e1, k
            best = np.nanmax(np.hypot(*(S[k, :n0, :2] - S[k - 1, :n0, :2]).T))
    g = load_grid(w)
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.9))
    nk = int(at[kb]) + 1
    Pk = np.nan_to_num(S[kb])[:nk]
    cx0, cx1 = Pk[:, 0].min() - 2.0, Pk[:, 0].max() + 2.0
    cy0, cy1 = Pk[:, 1].min() - 2.0, Pk[:, 1].max() + 2.0
    for ax, k, ttl in ((axs[0], kb - 1, "Before"), (axs[1], kb, "After")):
        n = int(at[k]) + 1
        kfs = [kf for kf in g.keyframes if kf.node < n]
        P = np.nan_to_num(S[k])
        L, org, r = g.render(P, bounds=b, keyframes=kfs)
        show_grid(ax, L, org, r)
        ax.plot(gt[:n, 0], gt[:n, 1], "--", c="k", lw=0.5, label="Ground Truth")
        ax.plot(P[:n, 0], P[:n, 1], "-", c="tab:blue", lw=0.7, label="RTISSLAM (Online)")
        ax.plot(*P[n - 1, :2], "o", c=(0.69, 0, 0.57), ms=4)
        ax.set_title(f"{ttl} the Loop Closure (Node {n}, Error {math.hypot(*(P[n - 1, :2] - gt[n - 1, :2])):.2f} m)",
                     fontsize=8)
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1); ax.set_aspect("equal")
        ax.tick_params(labelsize=6); ax.set_ylabel("y (m)", fontsize=7)
    axs[1].set_xlabel("x (m)", fontsize=7)
    axs[0].legend(loc="upper left", framealpha=0.9, fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "loopClosure.pdf", dpi=300)
    print(f"loop closure: snapshot {kb}, node {int(at[kb]) + 1}, largest node shift {best:.2f} m")


def rel(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, math.remainder(b[2] - a[2], 2 * math.pi)])


def fig_ao():
    w = "office"
    t = np.load(run_dir(w) / "trace.npz")
    f = np.load(data_dir() / "drives" / f"{w}_{TAG}.npz")
    gt, odo = f["gt"], f["odom"]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(odo[:, :2], axis=0).T))]
    k = t["kcal"]
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    ax = axs[0]
    ax.plot(s, k[:, 0], c="tab:blue", lw=0.9, label=r"$\kappa_s$ (Distance)")
    ax.plot(s, k[:, 1], c="tab:red", lw=0.9, label=r"$\kappa_\theta$ (Heading)")
    ax.axhline(1 / 1.02, c="tab:blue", ls=":", lw=0.8); ax.axhline(1 / 1.03, c="tab:red", ls=":", lw=0.8)
    ax.set_ylabel("Scale Factor"); ax.set_title("a) Wheel Calibration", fontsize=9)
    ax.legend(loc="upper right"); ax.set_ylim(0.95, 1.005)
    # accumulated heading error along the drive: raw wheels, calibrated wheels, acoustic odometry (wheels where rejected)
    N = len(gt)
    G = np.array([rel(gt[i - 1], gt[i]) for i in range(1, N)])
    O = np.array([rel(odo[i - 1], odo[i]) for i in range(1, N)])
    A = t["ao_log"]
    ao_th = O[:, 2].copy()
    for row in A:
        i = int(row[0])
        if row[3] > 0 and 1 <= i < N:
            ao_th[i - 1] = row[2]
    ax = axs[1]
    ax.plot(s[1:], np.degrees(np.cumsum(O[:, 2] - G[:, 2])), c="tab:red", lw=0.9, label="Wheel Odometry")
    ax.plot(s[1:], np.degrees(np.cumsum(ao_th - G[:, 2])), c="tab:green", lw=0.9, label="Acoustic Odometry")
    ax.plot(s[1:], np.degrees(np.cumsum(O[:, 2] * k[1:, 1] - G[:, 2])), c="tab:blue", lw=0.9, label="Calibrated Wheels")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("Distance Traveled (m)"); ax.set_ylabel(r"Accumulated Heading Error ($^\circ$)")
    ax.set_title("b) Heading Drift", fontsize=9); ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "acousticOdometry.pdf", dpi=300)


def fig_worlds():
    from rtissim.worlds import concave_corners, driving_line
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [36.6, 33.6, 39.6]})
    for ax, w in zip(axs, WORLDS):
        p = PLANS[w](); wd = build_world(p); a = wd.arrays()
        occ, (x0, y0), res = wd.occupancy(0.05)
        ax.imshow(np.where(occ.T, 0.3, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                  extent=(x0, x0 + occ.shape[0] * res, y0, y0 + occ.shape[1] * res), rasterized=True)
        z = a["pos"][:, 2]; sel = (z > 0.1) & (z < 0.9)
        ax.scatter(a["pos"][sel, 0], a["pos"][sel, 1], s=0.05, c="tab:blue", alpha=0.25, lw=0, rasterized=True)
        L = driving_line(p, wd)
        ax.plot(*np.vstack([L, L[:1]]).T, "-", c="tab:green", lw=0.8, label="Mapping Drive")
        ax.plot(*L[0], "o", c="tab:green", ms=3.5, mec="k", mew=0.4)
        c = np.array(concave_corners(p, wd)); ax.plot(c[:, 0], c[:, 1], "^", c="tab:red", ms=2.2, label="Corners")
        if p.frames:
            f = np.array(p.frames); ax.plot(f[:, 0], f[:, 1], "s", c="tab:orange", ms=1.2, label="Door Frames")
        o = np.array(p.objects); ax.plot(o[:, 0], o[:, 1], "o", c="tab:purple", ms=2.2, label="Point Objects")
        ax.set_title(w.title()); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.tick_params(labelsize=7)
    axs[0].set_ylabel("y (m)")
    axs[0].legend(loc="upper left", fontsize=5.5, framealpha=0.9, markerscale=1.5, handlelength=1.0)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "worlds.pdf", dpi=300)


def fan_up(ax, img_db, r, az, dr_db=30.0, rmax=5.0):
    """Energyscape [R, D] (dB) as a fan in the sensor frame, forward up and left on the left, range-compensated,
    the top dr_db shown (colormap hot), with grid lines every 30 deg and every 1 m."""
    v = img_db + 40 * np.log10(r)[:, None]
    v = v - np.percentile(v, 99.95)
    re = np.r_[r - (r[1] - r[0]) / 2, r[-1] + (r[1] - r[0]) / 2]
    ae = np.r_[az - (az[1] - az[0]) / 2, az[-1] + (az[1] - az[0]) / 2]
    A, R = np.meshgrid(ae, re)
    ax.set_facecolor("white")
    ax.pcolormesh(-R * np.sin(A), R * np.cos(A), v, vmin=-dr_db, vmax=0, cmap="hot", shading="flat", rasterized=True)
    fan_grid(ax, rmax)


def fan_grid(ax, rmax=5.0, color="w"):
    t = np.linspace(-np.pi / 2, np.pi / 2, 181)
    for rr in np.arange(1, rmax + 0.1, 1.0):
        ax.plot(-rr * np.sin(t), rr * np.cos(t), "-", c=color, lw=0.3, alpha=0.45)
    for d in range(-90, 91, 30):
        a = np.radians(d)
        ax.plot([0, -rmax * np.sin(a)], [0, rmax * np.cos(a)], "-", c=color, lw=0.3, alpha=0.45)
    ax.set_xlim(-rmax, rmax); ax.set_ylim(0, rmax); ax.set_aspect("equal")


def fig_energyscapes():
    """Three poses in the office: the scene in the robot frame and the energyscapes of DAS, DMAS3 + CF and
    DMAS3 + CF + RW-CFAR, all forward up."""
    from matplotlib.transforms import Affine2D
    from rtissim.sensor import RtisSensor
    from rtisslam.imaging_post import POST
    poses = [(4.0, 6.7, 0.0), (16.0, 3.8, -90.0), (33.5, 11.0, 90.0)]
    p = PLANS["office"](); wd = build_world(p)
    sen = RtisSensor(wd.arrays(), device="cuda")
    r, az = sen.ranges, sen.azimuths
    occ, (x0, y0), res = wd.occupancy(0.05)
    labels = {"das": "DAS", "dmas3cf": "DMAS3 + CF", "dmas3cf:ams2s": "DMAS3 + CF + RW-CFAR"}
    fig, axs = plt.subplots(3, 4, figsize=(7.16, 3.25))
    for ri, (row, (x, y, yd)) in enumerate(zip(axs, poses)):
        pose = (x, y, math.radians(yd))
        maps, contrib = sen.render(pose)
        ax = row[0]
        # the scene in the sensor frame: world -> sensor (mount 0.1 m ahead), rotated so that forward is up
        mx = x + 0.1 * math.cos(pose[2]); my = y + 0.1 * math.sin(pose[2])
        tr = Affine2D().translate(-mx, -my).rotate(math.pi / 2 - pose[2]) + ax.transData
        ax.set_facecolor("white")
        ax.imshow(np.where(occ.T, 0.3, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1, rasterized=True,
                  interpolation="nearest", extent=(x0, x0 + occ.shape[0] * res, y0, y0 + occ.shape[1] * res),
                  transform=tr)
        fan_grid(ax, color="0.5")
        ax.plot(0, 0, "^", c="tab:green", ms=4)
        u, rr = contrib["u"], contrib["r"]
        for ax, m in zip(row[1:], labels):
            im, post = (m.split(":") + ["none"])[:2]
            fan_up(ax, POST[post](maps[im]), r, az)
        for ax in row:
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            if ri < 2:
                ax.set_xticklabels([])
        for ax in row[1:]:
            ax.set_yticklabels([])
        if ri == 0:
            row[0].set_title("Scene (Sensor Frame)", fontsize=8)
            for ax, m in zip(row[1:], labels):
                ax.set_title(labels[m], fontsize=8, color=(0.69, 0, 0.57) if m == "dmas3cf:ams2s" else "k")
        row[0].set_ylabel("Forward (m)", fontsize=6)
    for ax in axs[-1]:
        ax.set_xlabel("Lateral (m)", fontsize=6)
    fig.tight_layout(pad=0.2, w_pad=0.15, h_pad=0.9)
    fig.savefig(OUT / "energyscapes.pdf", dpi=300)


def fig_grids_methods():
    """Per world (rows) the grid rendered from the final RTISSLAM poses (seed 0), per imaging method (columns): DAS and
    DMAS3 + CF, without and with RW-CFAR; same inverse sensor model, same extent."""
    from dataclasses import replace
    from rtisslam.grid import GridConfig
    from rtisslam.imaging_post import POST
    cols = [("das", "none", "DAS"), ("das", "ams2s", "DAS + RW-CFAR"), ("dmas3cf", "none", "DMAS3 + CF"),
            ("dmas3cf", "ams2s", "DMAS3 + CF + RW-CFAR")]
    fig, axs = plt.subplots(3, 4, figsize=(7.16, 3.9))
    for ri, w in enumerate(WORLDS):
        f = np.load(data_dir() / "drives" / f"{w}_{TAG}.npz")
        fi = np.load(run_dir(w) / "trace.npz")["final"]
        odo = f["odom"]
        occ, (x0, y0), res, b = bounds_of(w)
        bx = box_of(w)
        for ci, (m, post, lab) in enumerate(cols):
            g = KeyframeGrid(replace(GridConfig(), method=m), f["ranges"], f["azimuths"])
            E = f[f"{m}_db"]
            for i in range(len(fi)):
                if g.want_keyframe(odo[i]):
                    g.add(i, odo[i], POST[post](E[i].astype(np.float32)))
            L, org, r = g.render(fi, bounds=b)
            ax = axs[ri, ci]
            ax.set_facecolor((0.5, 0.5, 0.5))
            show_grid(ax, L, org, r)
            ax.set_xlim(bx[0], bx[1]); ax.set_ylim(bx[2], bx[3]); ax.set_aspect("equal")
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            if ri < 2:
                ax.set_xticklabels([])
            if ci > 0:
                ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(lab, fontsize=8, color=(0.69, 0, 0.57) if (m, post) == ("dmas3cf", "ams2s") else "k")
        axs[ri, 0].set_ylabel(f"{w.title()}, y (m)", fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("x (m)", fontsize=7)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    fig.savefig(OUT / "gridsMethods.pdf", dpi=300)


def fig_colorbar():
    """A small horizontal bar of the 'hot' colormap, for inline use in a caption."""
    fig = plt.figure(figsize=(0.6, 0.08))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.linspace(0, 1, 256)[None], aspect="auto", cmap="hot", rasterized=True)
    ax.axis("off")
    fig.savefig(OUT / "colorbarHot.pdf", dpi=600)


def fig_evidence(variants=(("das:ams2", "DAS + RW-CFAR 2"), ("deconv:none", "Deconvolution"),
                            ("dmas3cf:ams2s", "DMAS3 + CF + RW-CFAR 2 s"), ("deconv:ams15", "Deconvolution + RW-CFAR 1.5"))):
    """Evidence curves fitted on the office drive (scripts/fit_evidence.py): binned P(correct | s) of the top-5
    candidates, and the logistic of the clipped linear evidence l(s) used by the sequence verifier."""
    ev = json.loads((data_dir() / "evidence" / "office.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3))
    cols = ["tab:gray", "tab:blue", "tab:orange", (0.69, 0, 0.57)]
    s = np.linspace(0.7, 1.0, 301)
    for (k, lab), c in zip(variants, cols):
        e = ev[k]
        b = np.array([float(x) for x in e["binned"]]) + 0.01
        P = np.array([v[0] if v[0] is not None else np.nan for v in e["binned"].values()])
        n = np.array([v[1] for v in e["binned"].values()])
        ok = n >= 10
        axs[0].plot(b[ok], P[ok], "o", c=c, ms=3)
        l = np.clip(e["llr_slope"] * (s - e["llr_mid"]), -4, 4)
        axs[0].plot(s, 1 / (1 + np.exp(-l)), "-", c=c, lw=1, label=lab)
        axs[1].plot(s, l, "-", c=c, lw=1, label=f"{lab}: $s_0$ = {e['llr_mid']:.3f}, $a$ = {e['llr_slope']:.0f}")
        axs[1].axvline(e["min_similarity"], c=c, lw=0.5, ls=":")
    axs[0].set_xlabel("Similarity $s$"); axs[0].set_ylabel("P(Correct $|$ $s$)")
    axs[0].set_title("Correct Candidates (Office Drive)")
    axs[1].set_xlabel("Similarity $s$"); axs[1].set_ylabel(r"Evidence $\ell(s)$")
    axs[1].set_title("Fitted Evidence Curves")
    axs[0].legend(loc="upper left", framealpha=0.9)
    axs[1].legend(loc="lower right", framealpha=0.9, fontsize=6)
    for ax in axs:
        ax.set_xlim(0.7, 1.0); ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "evidenceCurves.pdf", dpi=300)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("slam", fig_slam), ("loop", fig_loop), ("ao", fig_ao), ("worlds", fig_worlds),
                     ("energyscapes", fig_energyscapes), ("grids", fig_grids_methods), ("evidence", fig_evidence),
                     ("colorbar", fig_colorbar)):
        if what in (name, "all"):
            fn()


if __name__ == "__main__":
    main()
