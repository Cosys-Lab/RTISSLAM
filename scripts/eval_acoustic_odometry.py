"""Acoustic odometry (2013 T-RO, Sec. IV-B) on a drive, per imaging method, against ground truth (SCORING only).

Pairs of consecutive stored pulses (gated at 0.1 m / 6 deg). Prediction window centred on the wheel odometry
(which has a 2 % scale error and a 3 % heading bias). Reports, for the accepted estimates, the error of the forward
displacement and of the rotation, the acceptance rate, and the same errors for the wheel odometry.

    python scripts/eval_acoustic_odometry.py office [--methods das dmas2 dmas3cf deconv]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtisslam.acoustic_odometry import AcousticOdometry, AcousticOdometryConfig  # noqa: E402
from sondec.paths import data_dir  # noqa: E402


def rel(a, b):
    """pose b in the frame of pose a -> (forward, lateral, dtheta)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, math.remainder(b[2] - a[2], 2 * math.pi)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drive")
    ap.add_argument("--methods", nargs="*", default=["das", "dmas2", "dmas3cf", "deconv"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="translation")
    ap.add_argument("--post", default="none", help="RW-CFAR post-processing (rtisslam/imaging_post.py)")
    ap.add_argument("--set", nargs="*", default=[], help="AcousticOdometryConfig overrides key=value")
    ap.add_argument("--sharp", type=float, default=1.6, help="also score the pairs with peak sharpness >= this "
                                                               "(the ones that feed the wheel calibration)")
    ap.add_argument("--label", default="", help="tuning label: results go to RTISSLAM_DATA/tuning/ao.json")
    a = ap.parse_args()
    from rtisslam.imaging_post import POST
    post = POST[a.post]
    f = np.load(data_dir() / "drives" / f"{a.drive}_map.npz")
    gt, odo = f["gt"], f["odom"]
    N = len(gt) if not a.n else min(a.n, len(gt))
    G = np.array([rel(gt[i], gt[i + 1]) for i in range(N - 1)])
    O = np.array([rel(odo[i], odo[i + 1]) for i in range(N - 1)])
    out = {"wheel": {"ds_rmse_mm": float(1e3 * np.sqrt(np.mean((O[:, 0] - G[:, 0]) ** 2))),
                     "dth_rmse_deg": float(np.degrees(np.sqrt(np.mean((O[:, 2] - G[:, 2]) ** 2)))),
                     "dth_scale_err": float(np.sum((O[:, 2] - G[:, 2]) * G[:, 2]) / np.sum(G[:, 2] ** 2)),
                     "ds_scale_err": float(np.sum((O[:, 0] - G[:, 0]) * G[:, 0]) / np.sum(G[:, 0] ** 2))}}
    print(f"{a.drive}: {N - 1} pairs; wheel odometry: {out['wheel']}")
    for m in a.methods:
        cfg = AcousticOdometryConfig(method=m, model=a.model)
        for kv in a.set:
            k, v = kv.split("=")
            setattr(cfg, k, type(getattr(cfg, k))(v))
        ao = AcousticOdometry(cfg, f["ranges"], f["azimuths"])
        E = f[f"{m}_db"]
        imgs = [ao.image(post(E[i].astype(np.float32))) for i in range(N)]
        est = [ao.estimate(imgs[i], imgs[i + 1], O[i, 0], O[i, 2]) for i in range(N - 1)]
        ok = np.array([e.ok for e in est])
        ds = np.array([e.ds for e in est]); dth = np.array([e.dth for e in est])
        eds, eth = ds - G[:, 0], dth - G[:, 2]
        r = {"accepted": float(ok.mean()),
             "ds_rmse_mm": float(1e3 * np.sqrt(np.mean(eds[ok] ** 2))), "ds_bias_mm": float(1e3 * np.mean(eds[ok])),
             "dth_rmse_deg": float(np.degrees(np.sqrt(np.mean(eth[ok] ** 2)))),
             "dth_bias_deg_per_rad": float(np.degrees(np.sum(eth[ok])) / max(np.sum(np.abs(G[ok, 2])), 1e-9)),
             "dth_mad_deg": float(np.degrees(np.median(np.abs(eth[ok])))),
             "dth_scale_err": float(np.sum(eth[ok] * G[ok, 2]) / np.sum(G[ok, 2] ** 2)),
             "ds_scale_err": float(np.sum(eds[ok] * G[ok, 0]) / np.sum(G[ok, 0] ** 2))}
        sh = ok & (np.array([e.sharp for e in est]) >= a.sharp)
        r["sharp_share"] = float(sh.mean())
        r["dth_scale_err_sharp"] = float(np.sum(eth[sh] * G[sh, 2]) / max(np.sum(G[sh, 2] ** 2), 1e-12))
        r["ds_scale_err_sharp"] = float(np.sum(eds[sh] * G[sh, 0]) / max(np.sum(G[sh, 0] ** 2), 1e-12))
        out[m] = r
        print(f"  {m:8s} " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
    if a.label:
        p = data_dir() / "tuning" / "ao.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        allr = json.loads(p.read_text()) if p.exists() else {}
        allr[a.label] = {"post": a.post, "set": a.set, "model": a.model, **out}
        p.write_text(json.dumps(allr, indent=1))
        return
    p = data_dir() / "runs" / a.drive
    p.mkdir(parents=True, exist_ok=True)
    (p / f"acoustic_odometry_{a.model}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
