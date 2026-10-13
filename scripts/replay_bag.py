"""Offline localization on a ROS 2 bag (the first thing to run on real data): raw eRTIS pulses (or energyscapes) +
wheel odometry from a bag -> imaging -> RTISSLAM localizer in a saved map; compared with a reference pose topic
(the lidar reference of real.launch.py, or the simulator's ground truth).

    source /opt/ros/jazzy/setup.bash; source ~/ros2_ws/install/setup.bash
    ~/venvs/rtisslam/bin/python scripts/replay_bag.py <bag dir> --map <map.npz> [--localizer mix|pf|graph]
        [--start x,y,yaw | --global] [--pulse-topic /rtis/pulse] [--es-topic /rtis/energyscape] [--odom-topic /odom]
        [--ref-topic /ground_truth/odom] [--device cuda] [--out <prefix>]

Input topics: rtis_msgs/RtisPulse (imaged here with the same code as rtis_imaging_node) or rtis_msgs/Energyscape
(one message per "imaging:post" key, same stamp), nav_msgs/Odometry, and optionally a reference pose
(nav_msgs/Odometry, geometry_msgs/PoseStamped or PoseWithCovarianceStamped). The odometry used for a pulse is the
latest one at or before the pulse stamp. The reference frame of a lidar SLAM differs from the RTISSLAM map frame,
so the error is reported both raw and after the best rigid 2D alignment of the two trajectories.
Writes <out>.npz (per processed pulse: stamp, estimate, covariance, odometry, reference, candidates, ms) and
<out>.json (summary), and prints the summary.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def pose_of(msg):
    """Odometry / PoseStamped / PoseWithCovarianceStamped -> (x, y, yaw)."""
    p = msg.pose
    if hasattr(p, "pose"):                      # Odometry, PoseWithCovarianceStamped: PoseWithCovariance
        p = p.pose
    return np.array([p.position.x, p.position.y, yaw(p.orientation)])


def stamp(msg) -> float:
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def align_2d(A: np.ndarray, B: np.ndarray):
    """Rigid R, t minimising |R a + t - b| (Umeyama without scale), A, B [n, 2]."""
    ma, mb = A.mean(0), B.mean(0)
    H = (A - ma).T @ (B - mb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    return R, mb - R @ ma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--map", required=True)
    ap.add_argument("--localizer", default="mix", choices=["graph", "mix", "pf"])
    ap.add_argument("--start", default=None, help="x,y,yaw in the map (default: the map's first node)")
    ap.add_argument("--global", dest="global_", action="store_true", help="unknown start (mix / pf)")
    ap.add_argument("--pulse-topic", default="/rtis/pulse")
    ap.add_argument("--es-topic", default="/rtis/energyscape")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ref-topic", default="/ground_truth/odom")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    from rtisslam.imaging_post import POST
    from rtisslam.system import RTISSLAM

    slam = RTISSLAM.load(a.map)
    keys = sorted(slam.cfg.energyscape_keys())
    split = {k: (k.split(":") + ["none"])[:2] for k in keys}
    if a.global_:
        start = None
    elif a.start:
        start = [float(v) for v in a.start.split(",")]
    else:
        start = slam.poses()[0].tolist()
    slam.start_localization(start, (0.3, 0.3, 0.1), localizer=a.localizer)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    want = [t for t in (a.pulse_topic, a.es_topic, a.odom_topic, a.ref_topic) if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=want))
    cls = {t: get_message(types[t]) for t in want}
    use_pulse = a.pulse_topic in types
    print(f"bag {a.bag}: topics {want}; input {'raw pulses (imaged here)' if use_pulse else 'energyscapes'}", flush=True)

    sensor = None
    if use_pulse:
        import torch
        from rtissim.sensor import RtisSensor
        empty = {"pos": np.zeros((0, 3), np.float32), "normal": np.zeros((0, 3), np.float32),
                 "area": np.zeros(0, np.float32), "mat": np.zeros((0, 3), np.float32), "obj": np.zeros(0, np.int32),
                 "two": np.zeros(0, bool), "obj_size": np.zeros(0, np.float32), "obj_tilt": np.zeros(0, np.float32),
                 "obj_name": np.array([]), "boxes": np.zeros((0, 7), np.float32), "capsules": np.zeros((0, 7), np.float32)}
        sensor = RtisSensor(empty, device=a.device, methods=sorted({im for im, _ in split.values()}))

    odom, ref = None, None
    es_buf: dict = {}
    rows = []
    t_img = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        m = deserialize_message(data, cls[topic])
        if topic == a.odom_topic:
            odom = pose_of(m)
            continue
        if topic == a.ref_topic:
            ref = pose_of(m)
            continue
        maps = None
        if topic == a.pulse_topic:
            t0 = time.perf_counter()
            x = torch.as_tensor(np.asarray(m.data, np.float32).reshape(m.n_mics, m.n_samples), device=a.device)
            imgs = sensor.image(x)
            maps = {k: POST[p](imgs[im]) for k, (im, p) in split.items()}
            t_img.append(time.perf_counter() - t0)
        elif topic == a.es_topic and m.method in keys:
            st = (m.header.stamp.sec, m.header.stamp.nanosec)
            es_buf.setdefault(st, {})[m.method] = np.asarray(m.data, np.float32).reshape(len(m.ranges), len(m.azimuths))
            if len(es_buf[st]) == len(keys):
                maps = es_buf.pop(st)
        if maps is None or odom is None:
            continue
        t0 = time.perf_counter()
        log = slam.process(maps, odom)
        ms = 1e3 * (time.perf_counter() - t0)
        if log is None:
            continue
        try:
            C = slam.covariance()
        except Exception:                                        # noqa: BLE001
            C = np.full((3, 3), np.nan)
        cands = getattr(getattr(slam, "_rec", None), "last", [])
        rows.append(dict(t=stamp(m), est=slam.pose(), cov=C, odom=odom.copy(),
                         ref=None if ref is None else ref.copy(), n_cand=len(cands),
                         best_p=max((c.p for c in cands), default=0.0), ms=ms))

    if not rows:
        sys.exit("no pulse was processed (check the topics and that the robot moves: pulses are gated by travel)")
    est = np.array([r["est"] for r in rows])
    out = Path(a.out) if a.out else Path(a.bag).with_suffix("")
    summ = {"bag": str(a.bag), "map": str(a.map), "localizer": a.localizer, "pulses_processed": len(rows),
            "ms_localizer_mean": float(np.mean([r["ms"] for r in rows])),
            "ms_imaging_mean": float(1e3 * np.mean(t_img)) if t_img else None,
            "sigma_xy_median": float(np.nanmedian([math.sqrt(max(r["cov"][0, 0] + r["cov"][1, 1], 0)) for r in rows])),
            "share_with_candidates": float(np.mean([r["n_cand"] > 0 for r in rows]))}
    have = [i for i, r in enumerate(rows) if r["ref"] is not None]
    if have:
        R_ = np.array([rows[i]["ref"] for i in have])
        E = est[have]
        e_raw = np.hypot(*(E[:, :2] - R_[:, :2]).T)
        Rm, t = align_2d(E[:, :2], R_[:, :2])
        e_al = np.hypot(*((E[:, :2] @ Rm.T + t) - R_[:, :2]).T)
        summ.update({"err_raw_mean": float(e_raw.mean()), "err_raw_max": float(e_raw.max()),
                     "err_aligned_mean": float(e_al.mean()), "err_aligned_max": float(e_al.max()),
                     "alignment_rotation_deg": float(math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))),
                     "alignment_translation_m": [float(t[0]), float(t[1])]})
    np.savez_compressed(out.with_suffix(".npz"), t=np.array([r["t"] for r in rows]), est=est,
                        cov=np.array([r["cov"] for r in rows]), odom=np.array([r["odom"] for r in rows]),
                        ref=np.array([r["ref"] if r["ref"] is not None else [np.nan] * 3 for r in rows]),
                        n_cand=np.array([r["n_cand"] for r in rows]), best_p=np.array([r["best_p"] for r in rows]),
                        ms=np.array([r["ms"] for r in rows]))
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
