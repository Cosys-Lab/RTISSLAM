"""Compare the localizers in a frozen map on a second drive of the same world (different sensor and odometry noise):
graph (BatSLAM 2.0 detached session), mix (Gaussian mixture of EKFs), pf (particle filter), see rtisslam/localize.py.

    python scripts/eval_localizers.py office --query loc --localizers graph mix pf --starts known kidnapped global
Pulses are gated as on the robot. Per run: error to ground truth, error relative to the map (the map's own error at the
nearest mapping node subtracted: what a perfect localizer in this map would reach is 0), reported sigma, time to the
first fix below 0.5 m (relative), share of time within 0.5 m, ms per pulse. Writes RTISSLAM_DATA/loc_eval/<world>_<query>.json
and a trace npz per run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtisslam.imaging_post import POST  # noqa: E402
from rtisslam.system import RTISSLAM  # noqa: E402
from sondec.paths import data_dir  # noqa: E402


def run(map_file, drive, localizer, start, n, seed=0, loc_cfg=None):
    slam = RTISSLAM.load(map_file)
    f = np.load(drive)
    gt, odo = f["gt"][:n], f["odom"][:n]
    keys = slam.cfg.energyscape_keys()
    split = {k: (k.split(":") + ["none"])[:2] for k in keys}
    raw = {k: f[f"{im}_db"] for k, (im, _) in split.items()}
    rng = np.random.default_rng(seed)
    if start == "known":
        pose, sigma = gt[0] + rng.normal(0, [0.15, 0.15, 0.05]), (0.3, 0.3, 0.1)
    elif start == "kidnapped":                    # confidently wrong: 3 m off, small sigma
        a = rng.uniform(-np.pi, np.pi)
        pose, sigma = gt[0] + np.array([3 * np.cos(a), 3 * np.sin(a), 0.3]), (0.3, 0.3, 0.1)
    else:
        pose, sigma = None, (0.3, 0.3, 0.1)
    slam.start_localization(pose, sigma, localizer=localizer, loc_cfg=loc_cfg)
    est, cov, at, ms = [], [], [], []
    for i in range(len(gt)):
        maps = {k: POST[p](raw[k][i]) for k, (im, p) in split.items()}
        t0 = time.perf_counter()
        log = slam.process(maps, odo[i])
        if log is None:
            continue
        ms.append(1e3 * (time.perf_counter() - t0))
        est.append(slam.pose())
        try:
            cov.append(slam.covariance())
        except Exception:                                     # noqa: BLE001
            cov.append(np.full((3, 3), np.nan))
        at.append(i)
    return np.array(est), np.array(cov), np.array(at), np.array(ms), slam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--map", default=None, help="runs/<map>/map.npz (default: the world)")
    ap.add_argument("--query", default="loc", help="drives/<world>_<query>.npz")
    ap.add_argument("--map-tag", default="map", help="the map's mapping drive: drives/<world>_<map-tag>.npz")
    ap.add_argument("--localizers", nargs="+", default=["graph", "mix", "pf"])
    ap.add_argument("--starts", nargs="+", default=["known", "kidnapped", "global"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    map_file = data_dir() / "runs" / (a.map or a.world) / "map.npz"
    drive = data_dir() / "drives" / f"{a.world}_{a.query}.npz"
    gt_all = np.load(drive)["gt"]
    n = a.n or len(gt_all)
    gm = np.load(data_dir() / "drives" / f"{a.world}_{a.map_tag}.npz")["gt"]
    out_dir = data_dir() / "loc_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for loc in a.localizers:
        for st in a.starts:
            if loc == "graph" and st == "global":
                continue
            t0 = time.time()
            est, cov, at, ms, slam = run(map_file, drive, loc, st, n, a.seed)
            gt = gt_all[at]
            n_map = slam.n_map_nodes
            P_map = slam.poses()[:n_map] if loc == "graph" else slam._rec.P_map
            g = gm[:n_map]
            near = np.argmin(np.hypot(g[None, :, 0] - gt[:, None, 0], g[None, :, 1] - gt[:, None, 1]), axis=1)
            off = P_map[near, :2] - g[near, :2]
            e_gt = np.hypot(*(est[:, :2] - gt[:, :2]).T)
            e_rel = np.hypot(*((est[:, :2] - gt[:, :2]) - off).T)
            sig = np.sqrt(np.maximum(cov[:, 0, 0] + cov[:, 1, 1], 0))
            ok = e_rel < 0.5
            first = int(np.argmax(ok)) if ok.any() else -1
            k = f"{loc}/{st}"
            res[k] = {"pulses": int(len(at)), "err_gt_mean": float(e_gt.mean()), "err_gt_max": float(e_gt.max()),
                      "err_rel_mean": float(e_rel.mean()), "err_rel_p95": float(np.percentile(e_rel, 95)),
                      "err_rel_max": float(e_rel.max()), "share_rel_below_0p5": float(ok.mean()),
                      "first_fix_pulse": first, "first_fix_m": float(np.sum(np.hypot(*np.diff(gt[:first + 1, :2], axis=0).T)))
                      if first > 0 else 0.0,
                      "sigma_xy_median": float(np.nanmedian(sig)),
                      "nees_xy": float(np.nanmean([d @ np.linalg.solve(C[:2, :2], d) for d, C in
                                                   zip(est[:, :2] - gt[:, :2] - off, cov) if np.all(np.isfinite(C))])),
                      "ms_per_pulse": float(ms.mean()), "wall_s": time.time() - t0}
            print(f"{k:16s} " + "  ".join(f"{kk} {v:.3g}" for kk, v in res[k].items() if isinstance(v, float)), flush=True)
            np.savez_compressed(out_dir / f"{a.world}_{a.query}{a.tag}_{loc}_{st}.npz", est=est, cov=cov, at=at, gt=gt,
                                e_gt=e_gt, e_rel=e_rel, ms=ms, track=np.array(slam.track))
    p = out_dir / f"{a.world}_{a.query}{a.tag}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(res)
    p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
