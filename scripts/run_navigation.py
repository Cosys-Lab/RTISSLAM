"""Fully autonomous navigation in the simulator on a saved RTISSLAM map (offline closed loop, no ROS):
the robot localises in the map (BatSLAM recognition + odometry + acoustic odometry, no new templates), plans on the
RTISSLAM occupancy grid and drives with the layered acoustic-flow controller to a sequence of goals.

Goals: poses of the mapping drive (so they lie in explored space), spread over the drive and visited in a shuffled
order, so the routes differ from the mapping route (other directions, other combinations of corridors).

    python scripts/run_navigation.py office [--map office] [--goals 6] [--no-afc] [--tag nav]
Writes RTISSLAM_DATA/nav/<world>_<tag>.npz (per control step: GT, estimate, command, active layer, goal, bumps;
per pulse: the controller's energyscape; every replanned global path) and a metrics json next to it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtisnav.navigator import Navigator, NavConfig  # noqa: E402
from rtisslam.system import RTISSLAM  # noqa: E402
from rtissim.sim import Simulator  # noqa: E402
from sondec.paths import data_dir  # noqa: E402


def pick_goals(gt: np.ndarray, n: int, seed: int, min_sep: float = 6.0):
    rng = np.random.default_rng(seed)
    idx = np.linspace(0, len(gt) - 1, 4 * n).astype(int)
    goals = []
    for i in rng.permutation(idx):
        p = gt[i, :2]
        if all(np.hypot(*(p - q)) > min_sep for q in goals):
            goals.append(p)
        if len(goals) == n:
            break
    return np.array(goals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--map", default=None, help="run folder with map.npz (default: the world)")
    ap.add_argument("--goals", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-afc", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="ground-truth localization (diagnostics)")
    ap.add_argument("--afc-set", nargs="*", default=[], help="acoustic-flow controller overrides key=value")
    ap.add_argument("--localizer", default="graph", choices=["graph", "mix", "pf"],
                    help="localization in the frozen map (rtisslam/localize.py)")
    ap.add_argument("--tag", default="nav")
    ap.add_argument("--timeout", type=float, default=600.0, help="s per goal")
    a = ap.parse_args()
    drive = np.load(data_dir() / "drives" / f"{a.world}_map.npz")
    slam = RTISSLAM.load(data_dir() / "runs" / (a.map or a.world) / "map.npz")
    goals = pick_goals(drive["gt"], a.goals, a.seed)
    start = drive["gt"][0]
    # energyscapes the map was built with ("imaging:post" keys; plain names in older maps) + the controller's
    from rtisslam.imaging_post import POST
    afc_key = NavConfig().afc.method
    keys = slam.cfg.energyscape_keys() | {afc_key}
    split = {k: (k.split(":") + ["none"])[:2] for k in keys}
    methods = tuple(sorted({im for im, _ in split.values()}))
    sim = Simulator(a.world, methods=methods, seed=100 + a.seed, start=start)

    def energyscapes(raw):
        if raw is None:
            return None
        out = dict(raw)
        out.update({k: POST[post](raw[im]) for k, (im, post) in split.items()})
        return out
    rng = np.random.default_rng(a.seed)
    guess = start + rng.normal(0, [0.15, 0.15, 0.05])                       # the start is roughly known
    slam.start_localization(guess, sigma=(0.3, 0.3, 0.1), localizer=a.localizer)
    cfg = NavConfig(use_afc=not a.no_afc)
    for kv in a.afc_set:
        k, v = kv.split("=")
        setattr(cfg.afc, k, type(getattr(cfg.afc, k))(v))
    nav = Navigator(slam, cfg)
    log = {k: [] for k in ("t", "gt", "est", "odom", "v", "w", "layer", "goal", "bump", "pulse")}
    es_log, paths = [], []
    results = []
    t0 = time.time()
    obs = sim.step(0.0, 0.0, fire=True)
    for gi, g in enumerate(goals):
        nav.set_goal(g)
        tg = sim.t
        last_path = None
        while nav.state == "driving" and sim.t - tg < a.timeout:
            v, w = nav.step(obs["odom"], energyscapes(obs.get("maps")), sim.dt, oracle=obs["gt"] if a.oracle else None)
            if nav.path is not None and nav.path is not last_path:
                paths.append((len(log["t"]), nav.path.copy())); last_path = nav.path
            if "maps" in obs:
                es_log.append(energyscapes(obs["maps"])[afc_key].astype(np.float16))
            log["t"].append(sim.t); log["gt"].append(obs["gt"]); log["est"].append(nav.pose.copy()); log["odom"].append(obs["odom"])
            log["v"].append(v); log["w"].append(w); log["layer"].append(nav.layer); log["goal"].append(gi)
            log["bump"].append(obs["bump"]); log["pulse"].append(len(es_log) - 1)
            obs = sim.step(v, w)
        err_gt = float(np.hypot(*(obs["gt"][:2] - g)))
        loc_err = float(np.hypot(*(nav.pose[:2] - obs["gt"][:2]))) if nav.pose is not None else float("nan")
        results.append({"goal": gi, "xy": g.tolist(), "state": nav.state, "time_s": sim.t - tg,
                        "final_dist_gt_m": err_gt, "loc_err_m": loc_err})
        print(f"  goal {gi + 1}/{len(goals)} {nav.state}: {sim.t - tg:.0f} s, GT distance to goal {err_gt:.2f} m, "
              f"localization error {loc_err:.2f} m, bumps so far {sim.collisions}, {time.time() - t0:.0f} s wall", flush=True)
    gt, est = np.array(log["gt"]), np.array(log["est"])
    e = np.hypot(*(gt[:, :2] - est[:, :2]).T)
    layers = np.array(log["layer"])
    m = {"world": a.world, "tag": a.tag, "map": a.map or a.world, "localizer": a.localizer, "afc": not a.no_afc, "goals": len(goals),
         "reached": sum(r["state"] == "reached" for r in results),
         "reached_within_0p5m_gt": sum(r["final_dist_gt_m"] < 0.5 for r in results),
         "loc_err_mean_m": float(e.mean()), "loc_err_max_m": float(e.max()), "bumps": sim.collisions,
         "contact_events": int(np.sum(np.diff(np.r_[0, np.asarray(log["bump"], int)]) > 0)),
         "contact_time_s": float(np.sum(log["bump"]) * sim.dt),
         "sim_time_s": sim.t, "distance_m": float(np.sum(np.hypot(*np.diff(gt[:, :2], axis=0).T))),
         "layer_share": {L: float((layers == L).mean()) for L in ("CA", "OA", "AFF", "RCF", "PATH", "RECOVERY")},
         "recoveries": nav.n_recoveries,
         "wall_s": time.time() - t0, "results": results}
    out = data_dir() / "nav" / f"{a.world}_{a.tag}"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **{k: np.asarray(v) for k, v in log.items()}, goals=goals,
                        es=np.stack(es_log), ranges=slam.ranges, azimuths=slam.azimuths,
                        path_at=np.array([p[0] for p in paths]), path_len=np.array([len(p[1]) for p in paths]),
                        paths=np.vstack([p[1] for p in paths]) if paths else np.zeros((0, 2)),
                        grid_L=nav.L, grid_org=np.asarray(nav.org), grid_res=nav.res)
    out.with_suffix(".json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: v for k, v in m.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    main()
