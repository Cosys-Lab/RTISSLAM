"""Offline RTISSLAM over a mapping drive: pose graph + occupancy grid, scored against ground truth.

    python scripts/run_slam.py office [--snap 10]
Writes RTISSLAM_DATA/runs/<drive>/: map.npz (the saved map), trace.npz (per node: odometry, online estimate, GT;
snapshots of all node estimates every --snap nodes, for the video), metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtisslam.system import RTISSLAM, RTISSLAMConfig  # noqa: E402
from sondec.paths import data_dir  # noqa: E402


def align_rmse(est, gt):
    """ATE after aligning the start pose (both trajectories start at the same known pose)."""
    return float(np.sqrt(np.mean(np.sum((est[:, :2] - gt[:, :2]) ** 2, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drive")
    ap.add_argument("--tag", default="map")
    ap.add_argument("--snap", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ao", action="store_true", help="no acoustic odometry factors")
    ap.add_argument("--no-links", action="store_true", help="no place recognition (dead reckoning)")
    ap.add_argument("--no-register", action="store_true", help="BatSLAM links without metric registration")
    ap.add_argument("--ao-sigma", type=float, nargs=3, default=None, help="acoustic odometry sigma (m, m, rad)")
    ap.add_argument("--ao-factors", action="store_true", help="also add acoustic odometry as between factors")
    ap.add_argument("--no-ao-calib", action="store_true", help="no wheel calibration from acoustic odometry")
    ap.add_argument("--imaging", default="deconv", help="energyscape for descriptor, grid and acoustic odometry")
    ap.add_argument("--post", default="none", help="energyscape post-processing: none | ams (azimuthal mean subtraction)")
    ap.add_argument("--ao-imaging", default=None, help="energyscape for the acoustic odometry (default: --imaging)")
    ap.add_argument("--ao-post", default=None, help="post-processing for the acoustic odometry (default: --post)")
    ap.add_argument("--grid-imaging", default=None, help="energyscape for the occupancy grid (default: --imaging)")
    ap.add_argument("--grid-post", default=None, help="post-processing for the occupancy grid (default: --post)")
    ap.add_argument("--register-on", default="descriptor", choices=["grid", "descriptor"],
                    help="energyscape whose echoes are registered for the links")
    ap.add_argument("--evidence", default="office", help="evidence curves fitted on this drive (scripts/fit_evidence.py); "
                                                        "'none' keeps the defaults")
    ap.add_argument("--llr-mid", type=float, default=None, help="evidence curve: similarity at zero evidence")
    ap.add_argument("--llr-slope", type=float, default=None, help="evidence curve: slope")
    ap.add_argument("--min-sim", type=float, default=None, help="candidate similarity floor")
    ap.add_argument("--variant", default="", help="name of the run subfolder (default: the drive)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, dotted: acoustic_odometry.dr_db=35 "
                                                       "ao_calib_sharp=2.0 descriptor.n_sectors=90 grid.det_db=20")
    a = ap.parse_args()
    f = np.load(data_dir() / "drives" / f"{a.drive}_{a.tag}.npz")
    gt, odo = f["gt"], f["odom"]
    # odometry frame = map frame at the start (the robot starts at a known pose)
    cfg = RTISSLAMConfig()
    cfg.use_acoustic_odometry = not a.no_ao
    cfg.register.enabled = not a.no_register
    cfg.ao_factors = a.ao_factors
    # energyscapes are keyed "imaging:post"; descriptor and grid share one, the acoustic odometry may use another
    key = f"{a.imaging}:{a.post}"
    ao_key = f"{a.ao_imaging or a.imaging}:{a.ao_post or a.post}"
    grid_key = f"{a.grid_imaging or a.imaging}:{a.grid_post or a.post}"
    cfg.descriptor.method, cfg.grid.method = key, grid_key
    cfg.register_method = key if a.register_on == "descriptor" else ""
    cfg.acoustic_odometry.method = ao_key
    rc, tc = cfg.batslam.recognition, cfg.batslam.templates
    if a.evidence != "none":
        evs = json.loads((data_dir() / "evidence" / f"{a.evidence}.json").read_text())
        ev = evs[key]
        rc.llr_mid, rc.llr_slope, tc.min_similarity = ev["llr_mid"], ev["llr_slope"], ev["min_similarity"]
        if f"{key}@loc" in evs:                 # localization sessions in this map (fit_evidence.py --query-tag)
            el = evs[f"{key}@loc"]
            cfg.localization = {k: el[k] for k in ("llr_mid", "llr_slope", "min_similarity")}
    if a.llr_mid is not None:
        rc.llr_mid = a.llr_mid
    if a.llr_slope is not None:
        rc.llr_slope = a.llr_slope
    if a.min_sim is not None:
        tc.min_similarity = a.min_sim
    cfg.ao_calibrate = not a.no_ao_calib
    if a.ao_sigma:
        cfg.ao_sigma = tuple(a.ao_sigma)
    if a.no_links:
        cfg.batslam.templates.min_similarity = 2.0
    for kv in a.set:
        k, v = kv.split("=")
        *path, leaf = k.split(".")
        obj = cfg
        for pp in path:
            obj = getattr(obj, pp)
        old = getattr(obj, leaf)
        setattr(obj, leaf, type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true", "yes"))
    slam = RTISSLAM(cfg, f["ranges"], f["azimuths"], initial_pose=tuple(gt[0]))
    meths = {cfg.descriptor.method, cfg.grid.method, cfg.acoustic_odometry.method} | ({cfg.register_method} - {""})
    from rtisslam.imaging_post import POST
    maps_all = {m: f[f"{m.split(':')[0]}_db"] for m in meths}
    posts = {m: POST[m.split(':')[1]] for m in meths}
    online, nodes, snaps, snap_at, n_links, kcal = [], [], [], [], [], []
    t0 = time.time()
    for i in range(len(gt)):
        log = slam.process({m: posts[m](maps_all[m][i]) for m in meths}, odo[i], force=True)
        nodes.append(i)
        online.append(slam.pose())
        n_links.append(len(slam.slam.graph.loops))
        kcal.append((slam.k_s, slam.k_th))
        if (i + 1) % a.snap == 0 or i == len(gt) - 1:
            snaps.append(slam.poses()); snap_at.append(i)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(gt)}: templates {len(slam.slam.templates)}, links {n_links[-1]}, "
                  f"err {np.hypot(*(online[-1][:2] - gt[i, :2])):.2f} m, {time.time() - t0:.0f} s", flush=True)
    online = np.asarray(online)
    final = slam.poses()
    # odometry re-anchored at the start pose
    from batslam.backend.pose_graph import pose2
    T = pose2(gt[0]).compose(pose2(odo[0]).inverse())
    dr = np.array([[q.x(), q.y(), q.theta()] for q in (T.compose(pose2(o)) for o in odo)])
    out = Path(a.out) if a.out else data_dir() / "runs" / (a.drive + (f"_{a.variant}" if a.variant else ""))
    out.mkdir(parents=True, exist_ok=True)
    slam.save(out / "map.npz")
    S = np.zeros((len(snaps), len(gt), 3)); S[:] = np.nan
    for k, s in enumerate(snaps):
        S[k, :len(s)] = s
    links = np.array([(l.query_key, l.anchor_key) for l in slam.slam.graph.loops], int).reshape(-1, 2)
    allk = slam.slam.graph.all_links
    rej = np.array([(l.query_key, l.anchor_key) for l in allk if l.status == "rejected"], int).reshape(-1, 2)
    np.savez_compressed(out / "trace.npz", gt=gt, odom=dr, online=online, final=final, snaps=S.astype(np.float32),
                        snap_at=np.asarray(snap_at), links=links, rejected=rej, n_links=np.asarray(n_links),
                        templates=slam.template_nodes(), kcal=np.asarray(kcal),
                        ao_log=np.asarray(slam.ao_log, float).reshape(-1, 6))
    # correct / false links (GT distance between query and anchor)
    d_link = np.hypot(*(gt[links[:, 0], :2] - gt[links[:, 1], :2]).T) if len(links) else np.zeros(0)
    A = np.array([r[3] for r in slam.ao_log], bool)
    m = {"drive": a.drive, "variant": a.variant, "imaging": a.imaging, "post": a.post, "ao_energyscape": ao_key, "grid_energyscape": grid_key, "register_on": a.register_on, "set": a.set,
         "llr_mid": rc.llr_mid, "llr_slope": rc.llr_slope, "min_similarity": tc.min_similarity, "ao_accepted": float(A.mean()) if len(A) else None, "nodes": len(gt), "templates": len(slam.slam.templates), "links": int(len(links)),
         "links_wrong_gt_gt_1m": int((d_link > 1.0).sum()), "rejected_links": int(len(rej)),
         "ate_odometry_m": align_rmse(dr, gt), "ate_online_m": align_rmse(online, gt), "ate_final_m": align_rmse(final, gt),
         "max_err_final_m": float(np.max(np.hypot(*(final[:, :2] - gt[:, :2]).T))), "runtime_s": time.time() - t0}
    R = np.array([r[2:] for r in slam.reg_log], float).reshape(-1, 4)
    m["wheel_calibration"] = [slam.k_s, slam.k_th]
    m["links_registered"] = int(R[:, 0].sum()) if len(R) else 0
    m["link_candidates"] = len(R)
    m["median_registration_correction_m"] = float(np.median(R[R[:, 0] > 0, 3])) if len(R) and R[:, 0].any() else None
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
