"""Mapping drive (the teach run): the robot follows the plan's tour (pure pursuit on the TRUE pose, as a human
driver would), the sonar fires at 10 Hz, and the pulses gated at >= GATE_M of travel or >= GATE_RAD of turning are
stored with their energyscapes:

    RTISSLAM_DATA/drives/<plan>_<tag>.npz
        gt [N, 3], odom [N, 3], t [N], k [N]     (per stored pulse; gt = EVALUATION ONLY)
        <method>_db [N, R, D] float16            for the stored methods
        ranges [R], azimuths [D], plan, laps, seed
        gt_all, odom_all [K, 3]                  (every control step, for plots)

    python scripts/gen_drive.py office --laps 1.15
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtissim.robot import PurePursuit  # noqa: E402
from rtissim.sensor import METHODS  # noqa: E402
from rtissim.sim import Simulator  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

GATE_M, GATE_RAD = 0.10, math.radians(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--laps", type=float, default=1.15, help="laps of the tour (>1: the start is revisited)")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="map")
    ap.add_argument("--v", type=float, default=0.4)
    a = ap.parse_args()
    sim = Simulator(a.plan, methods=a.methods, seed=a.seed)
    line = sim.line
    L = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
    pp = PurePursuit(line, v_max=a.v)
    ds_line = L / len(line)
    target = int(a.laps * len(line))
    out = {"gt": [], "odom": [], "t": [], "k": []}
    maps = {m: [] for m in a.methods}
    gt_all, odom_all = [], []
    last = None
    progress = 0
    prev_i = 0
    t0 = time.time()
    while progress < target:
        v, w = pp.command(sim.robot.pose)
        obs = sim.step(v, w, fire=False)
        di = (pp.index - prev_i) % len(line)
        progress += di if di < len(line) // 2 else 0
        prev_i = pp.index
        gt_all.append(obs["gt"]); odom_all.append(obs["odom"])
        if obs["k"] % sim.pulse_every:
            continue
        o = obs["odom"]
        if last is not None and np.hypot(*(o[:2] - last[:2])) < GATE_M and abs(math.remainder(o[2] - last[2], 2 * math.pi)) < GATE_RAD:
            continue
        last = o
        m, _ = sim.sensor.render(sim.robot.pose)
        for k in a.methods:
            maps[k].append(m[k].astype(np.float16))
        out["gt"].append(obs["gt"]); out["odom"].append(o); out["t"].append(obs["t"]); out["k"].append(obs["k"])
        n = len(out["gt"])
        if n % 250 == 0:
            print(f"  {n} pulses, {progress * ds_line:.0f} / {target * ds_line:.0f} m, t = {obs['t']:.0f} s, "
                  f"{time.time() - t0:.0f} s wall", flush=True)
        if obs["k"] > 200000:
            raise RuntimeError("drive did not finish")
    path = data_dir() / "drives" / f"{a.plan}_{a.tag}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in out.items()}, **{f"{k}_db": np.stack(v) for k, v in maps.items()},
             ranges=sim.sensor.ranges, azimuths=sim.sensor.azimuths, plan=a.plan, laps=a.laps, seed=a.seed,
             gt_all=np.asarray(gt_all), odom_all=np.asarray(odom_all), line=line, dt=sim.dt)
    print(f"{path}: {len(out['gt'])} pulses, {L * a.laps:.0f} m, {sim.t:.0f} s simulated, {time.time() - t0:.0f} s wall")


if __name__ == "__main__":
    main()
