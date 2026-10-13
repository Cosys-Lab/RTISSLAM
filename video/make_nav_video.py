"""RTISSLAM navigation video (Cosys-Lab style): fully autonomous navigation on the RTISSLAM map in the simulator.
Localization in the saved map (BatSLAM 2.0 recognition + odometry + acoustic odometry), global planning on the
RTISSLAM occupancy grid, local control by the layered acoustic-flow controller (Jansen et al., Sensors 2022).

    python video/make_nav_video.py office_nav [warehouse_nav ...] --out results/rtisslam_nav.mp4 [--test]

Inputs: RTISSLAM_DATA/nav/<run>.npz from scripts/run_navigation.py. One frame per control step (10 Hz, real time).
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
from matplotlib.patches import Circle, Polygon, Wedge  # noqa: E402

from rtisnav.afc import AFCConfig  # noqa: E402
from rtissim.worlds import PLANS, build_world  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

TITLE = "RTISSLAM: Autonomous Navigation With a 3D Imaging Sonar\nRTISSLAM Map, Global Planner and Acoustic-Flow Controller"
DR_DB = 25.0
LAYER_NAME = {"CA": "Collision Avoid.", "OA": "Obstacle Avoid.", "AFF": "Flow Following",
              "RCF": "Corridor Following", "PATH": "Path Following", "RECOVERY": "Recovery"}
LAYER_COL = {"CA": "tab:red", "OA": "tab:orange", "AFF": "tab:cyan", "RCF": "tab:green", "PATH": "w", "RECOVERY": "tab:purple"}


class NavFrame:
    def __init__(self, f, world):
        self.fig = fig = cv.new_fig()
        fig.text(0.5, 0.925, TITLE, ha="center", va="center", fontsize=28, linespacing=1.2)
        self.labels = ["World", "Goal", "To goal", "Controller", "Error (reference)", "Playback"]
        self._st = {}
        for i, lab in enumerate(self.labels):
            fig.text(0.02, 0.80 - 0.038 * i, f"{lab}:", fontsize=17, fontweight="bold")
            self._st[lab] = fig.text(0.135, 0.80 - 0.038 * i, "", fontsize=17)
        # energyscape fan with the controller zones (sensor frame, forward up, left = left)
        c = AFCConfig()
        r, az = f["ranges"], f["azimuths"]
        keep = r <= 3.0
        self.keep = keep
        r = r[keep]
        self.tvg = 40 * np.log10(r)[:, None]
        re = np.r_[r - (r[1] - r[0]) / 2, r[-1] + (r[1] - r[0]) / 2]
        ae = np.r_[az - (az[1] - az[0]) / 2, az[-1] + (az[1] - az[0]) / 2]
        A, R = np.meshgrid(ae, re)
        ax = self.fan_ax = fig.add_axes([0.025, 0.2, 0.19, 0.34])
        self.fan = ax.pcolormesh(-R * np.sin(A), R * np.cos(A), np.zeros((len(r), len(az))), vmin=-DR_DB, vmax=0,
                                 cmap="inferno", shading="flat", rasterized=True)
        self.zones = {
            "CA": ax.add_patch(Wedge((0, 0), c.r_ca, 0, 180, fill=False, ec="tab:red", lw=1.2, ls="--")),
            "OA": ax.add_patch(Polygon([(-c.y_oa, c.r_ca * 0.9), (c.y_oa, c.r_ca * 0.9), (c.y_oa, c.x_oa), (-c.y_oa, c.x_oa)],
                                       closed=True, fill=False, ec="tab:orange", lw=1.2, ls="--")),
        }
        for s in (-1, 1):
            self.zones[f"RCF{s}"] = ax.add_patch(Polygon([(s * c.rcf_y[0], c.rcf_x[0]), (s * c.rcf_y[1], c.rcf_x[0]),
                                                          (s * c.rcf_y[1], c.rcf_x[1]), (s * c.rcf_y[0], c.rcf_x[1])],
                                                         closed=True, fill=False, ec="tab:green", lw=1.0, ls=":"))
        ax.set_xlim(-3.1, 3.1); ax.set_ylim(-0.1, 3.2); ax.set_aspect("equal")
        ax.set_title("Energyscape and Controller Zones", fontsize=15, fontweight="bold", pad=4)
        ax.set_xlabel("lateral (m)", fontsize=11); ax.set_ylabel("forward (m)", fontsize=11); ax.tick_params(labelsize=9)
        cax = fig.add_axes([0.222, 0.22, 0.005, 0.28])
        cb = fig.colorbar(self.fan, cax=cax); cb.ax.tick_params(labelsize=9); cb.set_label("Normalized energy (dB)", fontsize=10)
        # map panel
        L, org, res = f["grid_L"], f["grid_org"], float(f["grid_res"])
        self.ext = (org[0], org[0] + L.shape[0] * res, org[1], org[1] + L.shape[1] * res)
        img = 1 - 1 / (1 + np.exp(-L.T))
        w = build_world(PLANS[world]())
        occ, (ox, oy), gres = w.occupancy(0.05)
        x0, x1, y0, y1 = ox - 0.8, ox + occ.shape[0] * gres + 0.8, oy - 0.8, oy + occ.shape[1] * gres + 0.8
        asp = (y1 - y0) / (x1 - x0)
        pw = 0.385
        ph = min(0.66, pw * asp * 1920 / 1080)
        yb = 0.14 + (0.68 - ph) / 2
        self.mp = fig.add_axes([0.285, yb, pw, ph])
        self.mp.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, extent=self.ext, interpolation="nearest")
        self.mp.set_xlim(x0, x1); self.mp.set_ylim(y0, y1); self.mp.set_aspect("equal")
        self.mp.set_title("RTISSLAM Map, Plan and Robot", fontsize=19, fontweight="bold", pad=6)
        self.mp.tick_params(labelsize=9); self.mp.set_xlabel("x (m)", fontsize=11); self.mp.set_ylabel("y (m)", fontsize=11)
        G = f["goals"]
        self.goal_pts = self.mp.scatter(G[:, 0], G[:, 1], marker="*", s=180, c="0.6", ec="k", lw=0.5, zorder=4)
        for i, g in enumerate(G):
            self.mp.text(g[0] + 0.35, g[1] + 0.35, str(i + 1), fontsize=12, color="k", zorder=5)
        (self.plan,) = self.mp.plot([], [], "-", c="tab:green", lw=1.8, label="Global Path")
        (self.trail,) = self.mp.plot([], [], "-", c=cv.MAGENTA, lw=1.2, alpha=0.8, label="Estimate")
        (self.gt,) = self.mp.plot([], [], "--", c="k", lw=0.8, label="Ground Truth")
        (self.rob,) = self.mp.plot([], [], "o", c=cv.MAGENTA, ms=8, mec="w", zorder=6)
        (self.hd,) = self.mp.plot([], [], "-", c=cv.MAGENTA, lw=2.4, zorder=6)
        self.mp.legend(loc="upper right", fontsize=9, framealpha=0.85)
        # local view
        self.lv = fig.add_axes([0.715, 0.22, 0.265, 0.265 * 1920 / 1080 * 0.9])
        self.lv.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, extent=self.ext, interpolation="nearest")
        self.lv.set_title("Local View (6 m)", fontsize=19, fontweight="bold", color=cv.MAGENTA, pad=6)
        self.lv.tick_params(labelsize=9); self.lv.set_aspect("equal")
        (self.lplan,) = self.lv.plot([], [], "-", c="tab:green", lw=2)
        self.echo = self.lv.scatter([], [], s=4, c="tab:orange", zorder=4, lw=0)
        self.foot = self.lv.add_patch(Circle((0, 0), 0.25, fc=(0.69, 0, 0.57, 0.35), ec=cv.MAGENTA, lw=1.5, zorder=5))
        (self.lhd,) = self.lv.plot([], [], "-", c=cv.MAGENTA, lw=2.4, zorder=6)
        (self.lgt,) = self.lv.plot([], [], "o", c="k", ms=4, zorder=6)
        self.cmd_txt = fig.text(0.02, 0.80 - 0.038 * 6.3, "", fontsize=15, color="0.25")
        fig.text(0.47, 0.11, "Ground truth (dashed line, black dot) and position error: for reference only. Map: the RTISSLAM "
                 "occupancy grid of the mapping drive.\nOrange: the echoes of the current pulse, placed with the "
                 "estimated pose. Energyscape: 25 dB range, range-compensated (+40 log r).", ha="center", va="center", fontsize=12,
                 color="0.4", linespacing=1.4)
        cv.logos(fig)
        r_all, az_all = f["ranges"], f["azimuths"]
        RR, AA = np.meshgrid(r_all, az_all, indexing="ij")
        self.sx, self.sy = 0.10 + RR * np.cos(AA), RR * np.sin(AA)
        self.tvg_all = 40 * np.log10(r_all)[:, None]

    def status(self, k, v):
        self._st[k].set_text(v)

    def update(self, es_db, est, gt_trail, est_trail, path, goal_idx, layer, v, w):
        e = es_db[self.keep] + self.tvg
        e = e - np.percentile(e, 99.9)
        self.fan.set_array(np.clip(e, -DR_DB, 0).ravel())
        for k, p in self.zones.items():
            act = (k == layer) or (k.startswith("RCF") and layer == "RCF")
            p.set_linewidth(3.0 if act else 1.0)
        x, y, th = est
        self.rob.set_data([x], [y]); self.hd.set_data([x, x + 1.0 * math.cos(th)], [y, y + 1.0 * math.sin(th)])
        self.trail.set_data(est_trail[:, 0], est_trail[:, 1]); self.gt.set_data(gt_trail[:, 0], gt_trail[:, 1])
        if path is not None:
            self.plan.set_data(path[:, 0], path[:, 1]); self.lplan.set_data(path[:, 0], path[:, 1])
        cols = np.full(len(self.goal_pts.get_offsets()), "0.6", dtype=object)
        cols[:goal_idx] = "tab:green"; cols[goal_idx] = "tab:red"
        self.goal_pts.set_color(list(cols)); self.goal_pts.set_edgecolor("k")
        # local view: echoes above the detection threshold, placed with the estimate
        vv = es_db + self.tvg_all
        thr = max(np.median(vv) + 22.0, vv.max() - 30.0)
        m = (vv > thr)
        c, s = math.cos(th), math.sin(th)
        px, py = x + c * self.sx[m] - s * self.sy[m], y + s * self.sx[m] + c * self.sy[m]
        self.echo.set_offsets(np.c_[px, py] if m.any() else np.zeros((0, 2)))
        self.foot.center = (x, y)
        self.lhd.set_data([x, x + 0.6 * math.cos(th)], [y, y + 0.6 * math.sin(th)])
        self.lgt.set_data([gt_trail[-1, 0]], [gt_trail[-1, 1]])
        self.lv.set_xlim(x - 3, x + 3); self.lv.set_ylim(y - 3, y + 3)
        self.cmd_txt.set_text(f"cmd_vel:  v = {v:.2f} m/s,  w = {math.degrees(w):+.0f} deg/s")
        return cv.grab(self.fig)


def run_frames(wr, run: str, idx: int, n_runs: int, test: bool = False, stride: int = 1):
    f = np.load(data_dir() / "nav" / f"{run}.npz")
    world = run.split("_")[0]
    lay = NavFrame(f, world)
    gt, est, v, w, layer, goal, bump, pulse = (f[k] for k in ("gt", "est", "v", "w", "layer", "goal", "bump", "pulse"))
    es = f["es"]
    goals = f["goals"]
    pa, pl, P = f["path_at"], f["path_len"], f["paths"]
    po = np.r_[0, np.cumsum(pl)]
    N = len(gt)
    steps = range(0, N, stride) if not test else range(0, N, max(1, N // 15))
    nf = len(steps)
    bumps = np.cumsum(bump)
    for fi, k in enumerate(steps):
        j = int(np.searchsorted(pa, k, side="right")) - 1
        path = P[po[j]:po[j + 1]] if j >= 0 else None
        g = int(goal[k])
        lay.status("World", f"{world.title()} ({idx} of {n_runs})")
        lay.status("Goal", f"{g + 1} of {len(goals)}")
        lay.status("To goal", f"{np.hypot(*(est[k, :2] - goals[g])):.1f} m")
        lay.status("Controller", LAYER_NAME[str(layer[k])])
        lay.status("Error (reference)", f"{np.hypot(*(est[k, :2] - gt[k, :2])):.2f} m")
        lay.status("Playback", f"{stride}x real time")
        img = lay.update(es[max(int(pulse[k]), 0)].astype(np.float32), est[k], gt[:k + 1], est[:k + 1], path, g,
                         str(layer[k]), float(v[k]), float(w[k]))
        wr.append_data(cv.fade(img, min(1.0, (fi + 1) / 6, (nf - fi) / 6)))
        if fi % 500 == 0:
            print(f"  {run}: frame {fi + 1}/{nf}", flush=True)
    plt.close(lay.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "results" / "rtisslam_nav.mp4"))
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--no-intro", action="store_true")
    ap.add_argument("--no-outro", action="store_true")
    ap.add_argument("--index", type=int, default=0, help="number of the first run in the status line (segments)")
    ap.add_argument("--total", type=int, default=0, help="total number of runs in the status line (segments)")
    ap.add_argument("--stride", type=int, default=4, help="control steps per frame (playback speed at 10 fps)")
    a = ap.parse_args()
    wr = cv.writer(a.out)
    if not a.no_intro:
      cv.hold(wr, cv.slide_title("RTISSLAM: Autonomous Navigation\nWith a 3D Imaging Sonar"), 4.5, fade_in=False)
      cv.hold(wr, cv.slide_text("What is shown", [
        "The robot drives fully autonomously to a sequence of goals, using only its eRTIS sonar and its wheel odometry.",
        "It localises in the RTISSLAM map of an earlier two-lap mapping drive: BatSLAM 2.0 recognition of the "
        "energyscapes and the wheel odometry, calibrated by acoustic odometry, in one pose graph, without laying new "
        "templates.",
        "A global planner computes the path on the RTISSLAM occupancy grid (unknown space is not driven through); the "
        "layered acoustic-flow controller of Jansen et al. (2022) follows it and reacts to the live echoes "
        "(collision and obstacle avoidance, acoustic flow and corridor following).",
        "Ground truth and the position error are shown for reference only."], wrap=96), 12)
    for i, run in enumerate(a.runs):
        run_frames(wr, run, (a.index or 1) + i, a.total or len(a.runs), a.test, a.stride)
    if not a.no_outro:
        cv.hold(wr, cv.slide_end(), 3.5, fade_out=False)
    wr.close()
    print("->", a.out)


if __name__ == "__main__":
    main()
