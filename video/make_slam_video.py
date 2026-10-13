"""RTISSLAM video (Cosys-Lab style, see 2026 - Skill Drafts/video-steckel-style): the whole SLAM system at work on
the three simulated worlds: the BatSLAM 2.0 pose graph growing (templates, view links, loop closures pulling the
trajectory into place) and the occupancy grid built from the eRTIS energyscapes and re-rendered from the current
graph in every frame.

    python video/make_slam_video.py --out results/rtisslam_slam.mp4 [--worlds office warehouse atrium] [--test]

Inputs: RTISSLAM_DATA/drives/<world>_map.npz (energyscapes, odometry, GT) and RTISSLAM_DATA/runs/<world>/
(trace.npz with graph snapshots every 2 nodes, map.npz with the grid keyframes), from scripts/gen_drive.py and
scripts/run_slam.py --snap 2. One frame per snapshot (2 pose-graph nodes), 10 fps.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "video"))
import cosys_video as cv  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import json  # noqa: E402
from rtisslam.grid import KeyframeGrid  # noqa: E402
from rtisslam.system import RTISSLAMConfig  # noqa: E402
from rtissim.worlds import PLANS, build_world  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

TITLE = "RTISSLAM: Sonar-Only SLAM With a 3D Imaging Sonar\nBatSLAM 2.0 Pose Graph and Occupancy Grid From eRTIS Energyscapes"
DR_DB = 25.0


class SlamFrame:
    def __init__(self, world: str, bounds, occ_gt, ranges, azimuths):
        self.fig = fig = cv.new_fig()
        fig.text(0.5, 0.925, TITLE, ha="center", va="center", fontsize=28, linespacing=1.2)
        self.labels = ["World", "Pulse", "Distance", "Templates", "Loop closures", "Error (reference)"]
        self._st = {}
        for i, lab in enumerate(self.labels):
            fig.text(0.02, 0.80 - 0.038 * i, f"{lab}:", fontsize=17, fontweight="bold")
            self._st[lab] = fig.text(0.135, 0.80 - 0.038 * i, "", fontsize=17)
        # energyscape fan
        self.fan_ax = fig.add_axes([0.025, 0.2, 0.19, 0.34])
        r, az = np.asarray(ranges), np.asarray(azimuths)
        keep = r <= 5.0
        self.keep = keep
        r = r[keep]
        re = np.r_[r - (r[1] - r[0]) / 2, r[-1] + (r[1] - r[0]) / 2]
        ae = np.r_[az - (az[1] - az[0]) / 2, az[-1] + (az[1] - az[0]) / 2]
        A, R = np.meshgrid(ae, re)
        self.tvg = 40 * np.log10(r)[:, None]
        self.cmap = plt.get_cmap("inferno")
        self.fan = self.fan_ax.pcolormesh(R * np.sin(A) * -1, R * np.cos(A), np.zeros((len(r), len(az))), vmin=-DR_DB,
                                          vmax=0, cmap=self.cmap, shading="flat", rasterized=True)
        for rr in range(1, 6):
            t = np.linspace(-math.pi / 2, math.pi / 2, 90)
            self.fan_ax.plot(rr * np.sin(t), rr * np.cos(t), c="w", lw=0.3, alpha=0.35)
        self.fan_ax.set_xlim(-5.1, 5.1); self.fan_ax.set_ylim(-0.1, 5.2); self.fan_ax.set_aspect("equal")
        self.fan_ax.set_title(ES_TITLE, fontsize=14, fontweight="bold", pad=4)
        self.fan_ax.set_xlabel("lateral (m)", fontsize=11); self.fan_ax.set_ylabel("forward (m)", fontsize=11)
        self.fan_ax.tick_params(labelsize=9)
        cax = fig.add_axes([0.222, 0.22, 0.005, 0.28])
        cb = fig.colorbar(self.fan, cax=cax); cb.ax.tick_params(labelsize=9)
        cb.set_label("Normalized energy (dB)", fontsize=10)
        # pose graph and occupancy grid panels
        x0, x1, y0, y1 = bounds
        asp = (y1 - y0) / (x1 - x0)
        pw = 0.345
        ph = min(0.64, pw * asp * 1920 / 1080)
        yb = 0.14 + (0.66 - ph) / 2
        self.pg = fig.add_axes([0.29, yb, pw, ph])
        self.og = fig.add_axes([0.645, yb, pw, ph])
        for ax, t in ((self.pg, "Pose Graph"), (self.og, "Occupancy Grid")):
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            ax.set_title(t, fontsize=19, fontweight="bold", color=cv.MAGENTA if t == "Occupancy Grid" else "k", pad=6)
            ax.tick_params(labelsize=9); ax.set_xlabel("x (m)", fontsize=11)
        self.pg.set_ylabel("y (m)", fontsize=11)
        occ, (ox, oy), res = occ_gt
        self.pg.imshow(np.where(occ.T, 0.82, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                       extent=(ox, ox + occ.shape[0] * res, oy, oy + occ.shape[1] * res))
        self.links = LineCollection([], colors=[(1.0, 0.55, 0.0, 0.55)], linewidths=0.8)
        self.pg.add_collection(self.links)
        (self.odo,) = self.pg.plot([], [], "-", c="tab:red", lw=1.0, alpha=0.8, label="Wheel Odometry")
        (self.est,) = self.pg.plot([], [], "-", c="tab:blue", lw=1.4, label="RTISSLAM")
        self.tpl = self.pg.scatter([], [], s=2.5, c="0.25", zorder=3, lw=0, label="Templates")
        self.pg.plot([], [], "-", c=(1.0, 0.55, 0.0), label="Loop Closures")
        self.pg.legend(loc="upper right", fontsize=9, framealpha=0.85)
        self.grid_im = self.og.imshow(np.zeros((2, 2)), origin="lower", cmap="gray", vmin=0, vmax=1,
                                      interpolation="nearest", extent=bounds)
        (self.og_tr,) = self.og.plot([], [], "-", c="tab:blue", lw=0.8, alpha=0.6)
        self.robot = []
        for ax in (self.pg, self.og):
            (dot,) = ax.plot([], [], "o", c=cv.MAGENTA, ms=7, mec="w", zorder=5)
            (hd,) = ax.plot([], [], "-", c=cv.MAGENTA, lw=2.2, zorder=5)
            self.robot.append((dot, hd))
        fig.text(0.53, 0.11, "Floor plan (light grey) and position error: ground truth, for reference only. The grid is rendered "
                 "from the current pose graph in every frame.\nEnergyscape: 25 dB range, range-compensated (+40 log r), "
                 "normalized per image.",
                 ha="center", va="center", fontsize=12, color="0.4", linespacing=1.4)
        cv.logos(fig)

    def status(self, k, v):
        self._st[k].set_text(v)

    def update(self, es_db, P, odo, n, tpl_nodes, links, L):
        v = es_db[self.keep] + self.tvg
        v = v - np.percentile(v, 99.9)
        self.fan.set_array(np.clip(v, -DR_DB, 0).ravel())
        self.odo.set_data(odo[:n, 0], odo[:n, 1])
        self.est.set_data(P[:n, 0], P[:n, 1])
        self.og_tr.set_data(P[:n, 0], P[:n, 1])
        t = tpl_nodes[tpl_nodes < n]
        self.tpl.set_offsets(P[t, :2] if len(t) else np.zeros((0, 2)))
        lk = links[(links[:, 0] < n) & (links[:, 1] < n)]
        self.links.set_segments([[P[q, :2], P[a, :2]] for q, a in lk])
        x, y, th = P[n - 1]
        for dot, hd in self.robot:
            dot.set_data([x], [y]); hd.set_data([x, x + 0.9 * math.cos(th)], [y, y + 0.9 * math.sin(th)])
        self.grid_im.set_data(1 - 1 / (1 + np.exp(-L.T)))
        return cv.grab(self.fig)


ES_TITLE = "Energyscape (Deconvolved)"
ES_NAMES = {"das": "DAS", "dmas3cf": "DMAS3+CF", "deconv": "Deconvolved", "dmas2": "DMAS", "dmas2cf": "DMAS+CF"}


def es_title(key: str) -> str:
    im, post = (key.split(":") + ["none"])[:2]
    s = ES_NAMES.get(im, im)
    if post != "none":
        s += ", RW-CFAR"
    return f"Energyscape ({s})"


def world_frames(wr, world: str, idx: int, n_worlds: int, test: bool = False, tag: str = "map", run_fmt: str = "{world}"):
    f = np.load(data_dir() / "drives" / f"{world}_{tag}.npz")
    run = data_dir() / "runs" / run_fmt.format(world=world)
    t = np.load(run / "trace.npz")
    d = np.load(run / "map.npz")
    cfg = RTISSLAMConfig.from_dict(json.loads(str(d["config"])))
    global ES_TITLE
    ES_TITLE = es_title(cfg.grid.method)            # the energyscape the occupancy grid is built from
    grid = KeyframeGrid(cfg.grid, d["ranges"], d["azimuths"])
    grid.load_state(d)
    w = build_world(PLANS[world]())
    occ_gt = w.occupancy(0.05)
    occ, (ox, oy), res = occ_gt
    bounds = (ox - 0.8, ox + occ.shape[0] * res + 0.8, oy - 0.8, oy + occ.shape[1] * res + 0.8)
    lay = SlamFrame(world, bounds, occ_gt, f["ranges"], f["azimuths"])
    gt, odo, snaps, snap_at = t["gt"], t["odom"], t["snaps"], t["snap_at"]
    links, tpl = t["links"], t["templates"]
    # the energyscape the map was built with ("imaging:post")
    from rtisslam.imaging_post import POST
    im, post = (cfg.grid.method.split(":") + ["none"])[:2]
    E, post_fn = f[f"{im}_db"], POST[post]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(f["odom"][:, :2], axis=0).T))]
    kfs = grid.keyframes
    kf_nodes = np.array([k.node for k in kfs])
    N = len(gt)
    frames = range(len(snap_at)) if not test else range(0, len(snap_at), max(1, len(snap_at) // 12))
    nf = len(frames)
    nfade = 6
    # links appear when their query node is processed; a link's anchor is always older
    for fi, k in enumerate(frames):
        n = int(snap_at[k]) + 1
        P = snaps[k]
        sel = [kf for kf, kn in zip(kfs, kf_nodes) if kn < n]
        L, _, _ = grid.render(np.nan_to_num(P), bounds=bounds, keyframes=sel)
        lay.status("World", f"{world.title()} ({idx} of {n_worlds})")
        lay.status("Pulse", f"{n} of {N}")
        lay.status("Distance", f"{s[n - 1]:.1f} m")
        lay.status("Templates", f"{int((tpl < n).sum())}")
        lay.status("Loop closures", f"{int(((links[:, 0] < n)).sum())}")
        lay.status("Error (reference)", f"{np.hypot(*(P[n - 1, :2] - gt[n - 1, :2])):.2f} m")
        img = lay.update(post_fn(E[n - 1].astype(np.float32)), P, odo, n, tpl, links, L)
        a = min(1.0, (fi + 1) / nfade, (nf - fi) / nfade)
        wr.append_data(cv.fade(img, a))
        if fi % 100 == 0:
            print(f"  {world}: frame {fi + 1}/{nf}", flush=True)
    plt.close(lay.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "rtisslam_slam.mp4"))
    ap.add_argument("--worlds", nargs="*", default=["office", "warehouse", "atrium"])
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--tag", default="map2", help="drive tag (map: 1 lap, map2: 2 laps)")
    ap.add_argument("--run", default="{world}_final", help="run folder pattern")
    a = ap.parse_args()
    wr = cv.writer(a.out)
    cv.hold(wr, cv.slide_title("RTISSLAM: Sonar-Only SLAM\nWith a 3D Imaging Sonar"), 4.5, fade_in=False)
    cv.hold(wr, cv.slide_text("What is shown", [
        "A simulated robot with an eRTIS imaging sonar (32 microphones, 10 pulses per second) drives a mapping route "
        "of two laps through three worlds with specular walls: an office floor, a warehouse and an atrium.",
        "Every pulse is beamformed into energyscapes (range x azimuth). The pose graph runs on the deconvolved "
        "energyscape with a range-wise CFAR: it is the place descriptor of BatSLAM 2.0, whose verified recognitions are "
        "registered on their echoes and become loop-closure links in a robust pose graph; acoustic odometry from "
        "consecutive energyscapes calibrates the wheel odometry online.",
        "Left: the energyscape the occupancy grid is built from (third-order DMAS with coherence factor and range-wise "
        "CFAR). Middle: odometry (red), the RTISSLAM estimate (blue), the templates and the view links. Right: the "
        "occupancy grid, anchored to the pose-graph nodes and re-rendered whenever loop closures move the graph.",
        "The floor plan and the position error are ground truth, shown for reference only."], wrap=96), 12)
    for i, wname in enumerate(a.worlds):
        world_frames(wr, wname, i + 1, len(a.worlds), a.test, a.tag, a.run)
    cv.hold(wr, cv.slide_end(), 3.5, fade_out=False)
    wr.close()
    print("->", a.out)


if __name__ == "__main__":
    main()
