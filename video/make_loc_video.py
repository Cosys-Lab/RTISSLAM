"""Localization video (Cosys-Lab style): the Gaussian-mixture localizer (rtisslam/localize.py) in a frozen RTISSLAM map,
from an unknown start pose, during autonomous navigation in the simulator.

    python video/make_loc_video.py <trace.pkl> --world office --map office --out results/rtisslam_localizer.mp4 [--test]

Input: a per-pulse trace of a closed-loop navigation run (ground truth, estimate and covariance, the hypotheses with
their covariances, the recognition candidates, goal and planned path), written by the tracing wrapper around
scripts/run_navigation.py used on 2026-10-12 (see report/tonight). One frame per processed pulse.
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "video"))
import cosys_video as cv  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, Polygon  # noqa: E402

from rtisnav.costmap import planning_grid  # noqa: E402
from rtisslam.system import RTISSLAM  # noqa: E402
from rtissim.worlds import PLANS  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

TITLE = "RTISSLAM: Localization With a 3D Imaging Sonar\nGaussian Mixture of EKFs on BatSLAM Recognitions and the Occupancy Grid"


def ellipse(mu, S, n_sigma=2.0, **kw):
    w, V = np.linalg.eigh(S[:2, :2])
    w = np.sqrt(np.maximum(w, 1e-8)) * n_sigma
    ang = math.degrees(math.atan2(V[1, 1], V[0, 1]))
    return Ellipse(mu[:2], 2 * w[1], 2 * w[0], angle=ang, **kw)


def robot_patch(p, size=0.45, **kw):
    x, y, th = p
    pts = np.array([[size, 0], [-0.6 * size, 0.5 * size], [-0.6 * size, -0.5 * size]])
    c, s = math.cos(th), math.sin(th)
    return Polygon(pts @ np.array([[c, s], [-s, c]]) + [x, y], closed=True, **kw)


class Frame:
    def __init__(self, world, slam, T):
        self.fig = fig = cv.new_fig()
        fig.text(0.5, 0.925, TITLE, ha="center", va="center", fontsize=28, linespacing=1.2)
        self.ax = ax = fig.add_axes([0.02, 0.05, 0.64, 0.80])
        tern, L, org, res = planning_grid(slam)
        img = np.where(tern == 100, 0.15, np.where(tern == 0, 1.0, 0.82)).T
        ext = (org[0], org[0] + tern.shape[0] * res, org[1], org[1] + tern.shape[1] * res)
        ax.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for w in PLANS[world]().walls:
            ax.plot([w[0], w[2]], [w[1], w[3]], color="tab:blue", lw=0.8, alpha=0.5)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("RTISSLAM map (DMAS3+CF), floor plan in blue (reference)", fontsize=16)
        self.gt_line, = ax.plot([], [], color="0.35", lw=1.5, label="true path (reference)")
        self.est_line, = ax.plot([], [], color=cv.MAGENTA, lw=1.5, label="estimate")
        self.plan_line, = ax.plot([], [], color="tab:green", lw=1.2, ls="--", label="planned path")
        self.goal, = ax.plot([], [], marker="*", ms=22, color="tab:green", ls="")
        ax.legend(loc="lower left", fontsize=12, framealpha=0.85)
        self.dyn = []
        # right: status + error
        self.st = fig.text(0.685, 0.80, "", fontsize=17, va="top", family="monospace", linespacing=1.5)
        fig.text(0.685, 0.505, "Legend", fontsize=17, weight="bold", va="top")
        fig.text(0.685, 0.47, "magenta: estimate + 2-sigma ellipse\norange: other hypotheses (alpha = weight)\n"
                 "arrows: recognitions of this pulse,\n   green = near the estimate, red = elsewhere\n"
                 "gray triangle: true pose (reference)", fontsize=13, va="top", linespacing=1.4)
        self.ae = ae = fig.add_axes([0.70, 0.07, 0.27, 0.18])
        ae.set_xlabel("pulse", fontsize=12); ae.set_ylabel("error to truth (m)", fontsize=12)
        ae.set_ylim(0, 2.0); ae.set_xlim(0, len(T)); ae.grid(alpha=0.3)
        self.err_line, = ae.plot([], [], color=cv.MAGENTA, lw=1.2)
        self.sig_line, = ae.plot([], [], color="0.5", lw=1.0, ls="--")
        ae.legend(["error", "reported 2-sigma"], fontsize=10, loc="upper right")

    def draw(self, T, i, err, phase):
        t = T[i]
        for a in self.dyn:
            a.remove()
        self.dyn = []
        ax = self.ax
        k0 = max(0, i - 4000)
        g = np.array([q["gt"] for q in T[k0:i + 1]])
        e = np.array([q["est"] for q in T[k0:i + 1]])
        self.gt_line.set_data(g[:, 0], g[:, 1])
        self.est_line.set_data(e[:, 0], e[:, 1])
        if t.get("path") is not None:
            self.plan_line.set_data(t["path"][:, 0], t["path"][:, 1])
        if t.get("goal") is not None:
            self.goal.set_data([t["goal"][0]], [t["goal"][1]])
        for c in t["cands"]:
            sim, p, x, y, th = c[:5]
            near = math.hypot(x - t["est"][0], y - t["est"][1]) < 1.0
            self.dyn.append(ax.arrow(x, y, 0.5 * math.cos(th), 0.5 * math.sin(th), width=0.06, length_includes_head=True,
                                     color="tab:green" if near else "tab:red", alpha=0.25 + 0.7 * p, zorder=4))
        for (hid, w, x, y, th), S in zip(t["hyps"], t.get("S", [None] * len(t["hyps"]))):
            if hid == t["cur"] or S is None:
                continue
            self.dyn.append(ax.add_patch(ellipse(np.array([x, y]), S, fc="orange", ec="darkorange", alpha=float(0.15 + 0.6 * w), zorder=5)))
        self.dyn.append(ax.add_patch(robot_patch(t["gt"], fc="0.45", ec="k", zorder=6)))
        self.dyn.append(ax.add_patch(ellipse(t["est"], t["cov"], fc="none", ec=cv.MAGENTA, lw=2.5, zorder=7)))
        self.dyn.append(ax.add_patch(robot_patch(t["est"], fc=cv.MAGENTA, ec="k", alpha=0.85, zorder=7)))
        sig = math.sqrt(max(t["cov"][0, 0] + t["cov"][1, 1], 0))
        hw = sorted([h[1] for h in t["hyps"]], reverse=True)
        near = sum(1 for c in t["cands"] if math.hypot(c[2] - t["est"][0], c[3] - t["est"][1]) < 1.0)
        self.st.set_text(f"{phase}\n\n"
                         f"hypotheses       {len(t['hyps'])}\n"
                         f"weight (best)    {hw[0] if hw else 0:.2f}\n"
                         f"sigma xy         {sig:.2f} m\n"
                         f"recognitions     {len(t['cands'])} ({near} near)\n"
                         f"error (ref.)     {err[i]:.2f} m")
        self.err_line.set_data(np.arange(i + 1), np.minimum(err[:i + 1], 1.99))
        sg = [2 * math.sqrt(max(q["cov"][0, 0] + q["cov"][1, 1], 0)) for q in T[:i + 1]]
        self.sig_line.set_data(np.arange(i + 1), np.minimum(sg, 1.99))
        return cv.grab(self.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--world", default="office")
    ap.add_argument("--map", default="office")
    ap.add_argument("--out", default=str(ROOT / "results" / "rtisslam_localizer.mp4"))
    ap.add_argument("--spin", type=int, default=0, help="number of leading pulses that are the turn in place")
    ap.add_argument("--test", action="store_true", help="a few frames only")
    a = ap.parse_args()
    T = pickle.load(open(a.trace, "rb"))
    slam = RTISSLAM.load(data_dir() / "runs" / a.map / "map.npz")
    err = np.array([math.hypot(*(t["est"][:2] - t["gt"][:2])) for t in T])
    wr = cv.writer(a.out)
    if not a.test:
        cv.hold(wr, cv.slide_title("RTISSLAM: Localization\nWith a 3D Imaging Sonar"), 4.0, fade_in=False)
        cv.hold(wr, cv.slide_text("What is shown", [
            "A simulated robot with an eRTIS imaging sonar navigates autonomously in a map built earlier by RTISSLAM "
            "(BatSLAM 2.0 pose graph and an occupancy grid, both from DMAS3 + coherence-factor energyscapes).",
            "Its start pose is unknown: it first turns once in place. Every pulse, BatSLAM's place recognition compares "
            "the sonar view with the map's templates; each recognized template implies a pose (its anchor node in the "
            "pose graph). A Gaussian mixture of up to ten EKF hypotheses is propagated with the wheel odometry, updated "
            "by the recognitions near each hypothesis, and weighted by how well the echoes fit the occupancy grid.",
            "Recognitions far from every hypothesis can start a new one, but only with a small prior weight: the views "
            "depend on the driving direction, and a corridor seen backwards can look like a parallel one.",
            "The floor plan, the true pose and the error are ground truth, shown for reference only."], wrap=96), 13)
    fr = Frame(a.world, slam, T)
    idx = range(0, min(len(T), 40)) if a.test else range(len(T))
    for i in idx:
        phase = "Turning in place: finding itself" if i < a.spin else "Navigating to the goal"
        wr.append_data(fr.draw(T, i, err, phase))
    if not a.test:
        cv.hold(wr, cv.slide_end(), 3.0)
    wr.close()
    print(f"{a.out}: {len(idx)} frames")


if __name__ == "__main__":
    main()
