"""Baseline: lidar-style grid SLAM (slam_toolbox, Karto scan matching + Ceres pose graph) on the sonar pseudo-scan.

    source /opt/ros/jazzy/setup.bash
    python3 scripts/baseline_slam_toolbox.py office_map2 [--set key=value ...] [--tag <suffix>]

Input: <project>/data/baseline/<world>_<tag>_scan.npz (scripts/export_scans.py: the RTISSLAM grid's first detection per
azimuth, the same wheel odometry RTISSLAM gets). The scans are replayed in simulated time into
sync_slam_toolbox_node (every scan is processed, none dropped), with TF odom -> base_link from the wheel odometry
and the sensor mount base_link -> laser; the replay waits until slam_toolbox has handled each scan.
Parameters: ros/rtisslam_bringup/config/slam_toolbox_baseline.yaml (tuned on the development drive office_map), or
with --stock slam_toolbox's own mapper_params_online_sync.yaml (+ max range 4.5 m and replay settings); --set overrides.

Output <project>/data/baseline/<world>_<tag>_slamtoolbox[_<suffix>].npz (not in git):
  t, gt, odom (as the input), est_online [N, 3] (map -> odom after scan k, composed with odom k: what the robot
  knows at that time), est_final [N, 3] (each scan placed relative to the latest pose-graph vertex at or before it,
  with the final optimized vertex poses and the wheel odometry in between), vertex_scan [M] (scan index of each
  vertex: the scans after which slam_toolbox published a new map -> odom),
  vertex_final [M, 3], grid [H, W] (int8 OccupancyGrid data, -1 unknown), grid_origin (x, y), grid_resolution,
  params (JSON), and the ATE (start-aligned RMSE of the xy error, as scripts/run_slam.py align_rmse) of the odometry,
  the online and the final estimates; also printed and written as .json. Map frame = the start pose gt[0].
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ros" / "rtisslam_bringup" / "config" / "slam_toolbox_baseline.yaml"


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def between(a, b):
    """a^-1 * b."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


def align_rmse(est, gt):
    """ATE after aligning the start pose (both trajectories start at the same known pose), as scripts/run_slam.py."""
    return float(np.sqrt(np.mean(np.sum((est[:, :2] - gt[:, :2]) ** 2, 1))))


# replay / sensor settings that also apply to slam_toolbox's stock parameters (--stock)
REPLAY = {"max_laser_range": 4.5, "minimum_time_interval": 0.0, "tf_buffer_duration": 3000.0,
          "enable_interactive_mode": True, "use_map_saver": False, "map_update_interval": 2.0}


def load_params(sets, stock=False):
    import yaml
    if stock:
        from ament_index_python.packages import get_package_share_directory
        f = Path(get_package_share_directory("slam_toolbox")) / "config" / "mapper_params_online_sync.yaml"
        p = yaml.safe_load(f.read_text())["slam_toolbox"]["ros__parameters"]
        p.update(REPLAY)
    else:
        p = yaml.safe_load(CONFIG.read_text())["slam_toolbox"]["ros__parameters"]
    for kv in sets:
        k, v = kv.split("=", 1)
        p[k] = yaml.safe_load(v)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drive", help="<world>_<tag>, e.g. office_map2")
    ap.add_argument("--set", nargs="*", default=[], help="slam_toolbox parameter overrides key=value")
    ap.add_argument("--tag", default="", help="output suffix")
    ap.add_argument("--stock", action="store_true", help="slam_toolbox's own online_sync parameters (+ max range, replay)")
    ap.add_argument("--timeout", type=float, default=0.3, help="s to wait for slam_toolbox per scan it should add")
    ap.add_argument("--max-scans", type=int, default=0, help="only the first n scans (debugging)")
    ap.add_argument("--domain", type=int, default=None, help="ROS_DOMAIN_ID (default: from the process id); "
                    "give parallel runs different ones")
    a = ap.parse_args()

    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from tf2_msgs.msg import TFMessage
    from visualization_msgs.msg import MarkerArray

    z = np.load(ROOT / "data" / "baseline" / f"{a.drive}_scan.npz")
    t, gt, odom, ranges, angles = z["t"], z["gt"], z["odom"], z["ranges"], z["angles"]
    mount, rmax = z["mount"], float(z["range_max"])
    if a.max_scans:
        t, gt, odom, ranges = t[:a.max_scans], gt[:a.max_scans], odom[:a.max_scans], ranges[:a.max_scans]
    n = len(t)
    o0 = odom[0]
    odom_rel = np.array([between(o0, o) for o in odom])          # odometry with the start at the origin
    params = load_params(a.set, a.stock)
    params.update({"use_sim_time": True, "odom_frame": "odom", "map_frame": "map", "base_frame": "base_link",
                   "scan_topic": "/scan", "mode": "mapping"})

    # slam_toolbox in its own process, in a private ROS domain so that nothing else on the machine interferes
    domain = str(a.domain if a.domain is not None else 30 + os.getpid() % 70)
    env = dict(os.environ, ROS_DOMAIN_ID=domain)
    os.environ["ROS_DOMAIN_ID"] = domain
    import tempfile
    pfile = Path(tempfile.gettempdir()) / f"slam_toolbox_baseline_{os.getpid()}.yaml"
    import yaml
    pfile.write_text(yaml.safe_dump({"slam_toolbox": {"ros__parameters": params}}))
    log = open(pfile.with_suffix(".log"), "w")
    from ament_index_python.packages import get_package_prefix
    exe = Path(get_package_prefix("slam_toolbox")) / "lib" / "slam_toolbox" / "sync_slam_toolbox_node"
    proc = subprocess.Popen([str(exe), "--ros-args",
                             "--params-file", str(pfile), "-r", "__node:=slam_toolbox"],
                            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    rclpy.init()
    node = Node("baseline_replay", parameter_overrides=[])
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    clock_pub = node.create_publisher(Clock, "/clock", 10)
    tf_pub = node.create_publisher(TFMessage, "/tf", 100)
    tfs_pub = node.create_publisher(TFMessage, "/tf_static", latched)
    scan_pub = node.create_publisher(LaserScan, "/scan", 100)
    state = {"m2o": None, "map": None, "graph": None, "pose_stamp": -1.0, "stamps": []}

    def on_tf(msg):
        for tr in msg.transforms:
            if tr.header.frame_id == "map" and tr.child_frame_id == "odom":
                q = tr.transform.rotation
                state["m2o"] = np.array([tr.transform.translation.x, tr.transform.translation.y,
                                         math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))])
    node.create_subscription(TFMessage, "/tf", on_tf, 100)
    from geometry_msgs.msg import PoseWithCovarianceStamped
    # slam_toolbox publishes /pose (stamped with the scan) for every scan it adds to the pose graph
    def on_pose(m):
        state["pose_stamp"] = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        state["stamps"].append(state["pose_stamp"])
    node.create_subscription(PoseWithCovarianceStamped, "/pose", on_pose, 1000)
    node.create_subscription(OccupancyGrid, "/map", lambda m: state.__setitem__("map", m), latched)
    node.create_subscription(MarkerArray, "/slam_toolbox/graph_visualization",
                             lambda m: state.__setitem__("graph", m), 10)

    def stamp_of(sec):
        from builtin_interfaces.msg import Time
        ns = int(round(sec * 1e9))
        s, nsec = divmod(ns, 1_000_000_000)
        return Time(sec=s, nanosec=nsec)

    def tf(parent, child, p, st):
        m = TransformStamped()
        m.header.stamp = st
        m.header.frame_id, m.child_frame_id = parent, child
        m.transform.translation.x, m.transform.translation.y = float(p[0]), float(p[1])
        m.transform.rotation.z, m.transform.rotation.w = math.sin(p[2] / 2), math.cos(p[2] / 2)
        return m

    def spin(dt):
        end = time.time() + dt
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.005)

    def publish_time(k):
        st = stamp_of(t[k])
        clock_pub.publish(Clock(clock=st))
        tf_pub.publish(TFMessage(transforms=[tf("odom", "base_link", odom_rel[k], st)]))
        return st

    tfs_pub.publish(TFMessage(transforms=[tf("base_link", "laser", (float(mount[0]), float(mount[1]), 0.0),
                                             stamp_of(0.0))]))
    publish_time(0)
    # slam_toolbox (Jazzy) is a lifecycle node: configure, then activate
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState
    cli = node.create_client(ChangeState, "/slam_toolbox/change_state")
    t0 = time.time()
    while not cli.wait_for_service(timeout_sec=0.5):
        if time.time() - t0 > 30 or proc.poll() is not None:
            proc.kill()
            sys.exit(f"slam_toolbox did not start, see {pfile.with_suffix('.log')}")
    for tid in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
        fut = cli.call_async(ChangeState.Request(transition=Transition(id=tid)))
        while not fut.done():
            publish_time(0)
            spin(0.05)
    # wait until it subscribes to /scan
    t0 = time.time()
    while node.count_subscribers("/scan") == 0:
        spin(0.1)
        if time.time() - t0 > 30 or proc.poll() is not None:
            proc.kill()
            sys.exit(f"slam_toolbox did not start, see {pfile.with_suffix('.log')}")
    spin(1.0)

    inc = float(angles[1] - angles[0])
    md, mh = float(params["minimum_travel_distance"]), float(params["minimum_travel_heading"])
    mt = float(params.get("minimum_time_interval", 0.0))
    est_online = np.zeros((n, 3))
    vscan = []                       # scans that slam_toolbox added as pose-graph vertices
    wall = time.time()
    for k in range(n):
        st = publish_time(k)
        spin(0.002)
        msg = LaserScan()
        msg.header.stamp, msg.header.frame_id = st, "laser"
        msg.angle_min, msg.angle_max, msg.angle_increment = float(angles[0]), float(angles[-1]), inc
        msg.time_increment, msg.scan_time = 0.0, 0.0
        msg.range_min, msg.range_max = 0.05, rmax
        msg.ranges = [float(r) for r in ranges[k]]
        scan_pub.publish(msg)
        # sync mode: slam_toolbox handles every scan in order; when it adds one to the graph it publishes /pose
        # stamped with that scan. Wait long only for scans it should add (its travel rule), briefly for the others.
        if vscan:
            d = between(odom_rel[vscan[-1]], odom_rel[k])
            expect = t[k] - t[vscan[-1]] >= mt and (math.hypot(d[0], d[1]) >= md or abs(d[2]) >= mh)
        else:
            expect = True
        t1, limit = time.time(), (a.timeout if expect else 0.02)
        while abs(state["pose_stamp"] - t[k]) > 1e-6 and time.time() - t1 < limit:
            if time.time() - t1 > 0.05:
                publish_time(k)            # keep the clock and odometry alive at this scan's time
            spin(0.005)
        if abs(state["pose_stamp"] - t[k]) <= 1e-6:
            vscan.append(k)                # (progress only; the vertices are taken from all /pose stamps below)
            spin(0.01)                     # the map -> odom of this scan
        m2o = state["m2o"] if state["m2o"] is not None else np.zeros(3)
        est_online[k] = compose(m2o, odom_rel[k])
        if k % 500 == 0 or (a.max_scans and k % 20 == 0):
            print(f"{a.drive}: scan {k}/{n}, vertices {len(vscan)}", flush=True)
    # let slam_toolbox finish its queue (no new /pose for 2 s), then one more map update
    k_last = n - 1
    last = (len(state["stamps"]), time.time())
    while time.time() - last[1] < 2.0:
        publish_time(k_last)
        spin(0.05)
        if len(state["stamps"]) != last[0]:
            last = (len(state["stamps"]), time.time())
    t_end = time.time()
    upd = float(params.get("map_update_interval", 5.0))
    while time.time() - t_end < max(3.0, 2 * upd + 1):
        st = stamp_of(t[k_last] + (time.time() - t_end))
        clock_pub.publish(Clock(clock=st))
        tf_pub.publish(TFMessage(transforms=[tf("odom", "base_link", odom_rel[k_last], st)]))
        spin(0.05)
    graph, grid = state["graph"], state["map"]
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    node.destroy_node()
    rclpy.shutdown()
    log.close()
    if grid is None:
        sys.exit(f"no /map received, see {pfile.with_suffix('.log')}")

    # vertices: graph markers (a sphere per vertex, id = vertex id = order of creation) -> optimized poses
    verts = sorted([m for m in (graph.markers if graph else []) if m.type == m.SPHERE], key=lambda v: v.id)
    vscan = sorted({int(np.argmin(np.abs(t - s_))) for s_ in state["stamps"] if np.min(np.abs(t - s_)) < 1e-6})
    m = min(len(verts), len(vscan))
    vscan = np.array(vscan[:m], int)
    vfin = np.zeros((m, 3))
    for i in range(m):
        q = verts[i].pose.orientation
        th = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        vfin[i] = [verts[i].pose.position.x, verts[i].pose.position.y, th]
    marker_heading = bool(m and np.any(np.abs(vfin[:, 2]) > 1e-9))
    if m and not marker_heading:
        # the markers carry no orientation: heading = online heading at the vertex's scan plus the rotation the
        # optimization applied locally (direction between neighbouring vertices, online -> final)
        for i in range(m):
            j0, j1 = max(0, i - 3), min(m - 1, i + 3)
            on, fi = est_online[vscan[[j0, j1]], :2], vfin[[j0, j1], :2]
            dth = 0.0
            if j1 > j0 and np.hypot(*(on[1] - on[0])) > 0.2:
                dth = wrap(math.atan2(*(fi[1] - fi[0])[::-1]) - math.atan2(*(on[1] - on[0])[::-1]))
            vfin[i, 2] = wrap(est_online[vscan[i], 2] + dth)
    est_final = est_online.copy()
    if m:
        vi = np.searchsorted(vscan, np.arange(n), side="right") - 1
        for k in range(n):
            i = max(vi[k], 0)
            est_final[k] = compose(vfin[i], between(odom_rel[vscan[i]], odom_rel[k]))

    # to the world frame: the map frame is the start pose gt[0]
    g0 = gt[0]
    W = lambda E: np.array([compose(g0, e) for e in E])                              # noqa: E731
    est_online_w, est_final_w, vfin_w = W(est_online), W(est_final), W(vfin) if m else np.zeros((0, 3))
    odom_w = np.array([compose(g0, o) for o in odom_rel])
    info = grid.info
    gridarr = np.array(grid.data, np.int8).reshape(info.height, info.width)
    origin = np.array([info.origin.position.x, info.origin.position.y])
    summ = {"drive": a.drive, "scans": n, "vertices": int(len(verts)), "vertices_matched": int(m),
            "marker_heading": marker_heading,
            "ate_odometry_m": align_rmse(odom_w, gt), "ate_online_m": align_rmse(est_online_w, gt),
            "ate_final_m": align_rmse(est_final_w, gt),
            "ate_final_vertices_m": align_rmse(vfin_w, gt[vscan]) if m else None,
            "wall_s": time.time() - wall, "params": {k: params[k] for k in sorted(params)}}
    out = ROOT / "data" / "baseline" / f"{a.drive}_slamtoolbox{('_' + a.tag) if a.tag else ''}"
    np.savez_compressed(out.with_suffix(".npz"), t=t, gt=gt, odom=odom_w, est_online=est_online_w,
                        est_final=est_final_w, vertex_scan=vscan, vertex_final=vfin_w, grid=gridarr,
                        grid_origin=origin, grid_origin_frame=np.array("map (start pose = gt[0])"),
                        grid_resolution=info.resolution, params=json.dumps(summ["params"]))
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
