"""Record the interactive ROS 2 demo (tools/demo.py: simulator + RTISSLAM localizer + Nav2 + navigator + RViz2) as an
mp4, with scripted goals instead of RViz2's "2D Goal Pose", on a virtual display so that the desktop is not touched.

    python3 tools/record_demo.py [--world office] [--goals 3] [--rtf 2] [--out results/ros_demo.mp4]

Needs Xvfb (apt install xvfb xdotool) and an ffmpeg with x11grab (the one of imageio-ffmpeg in the venv is used).
Starts Xvfb on --display at --size, runs the demo there (private ROS_DOMAIN_ID), maximizes the RViz2 window, sends
the goals one by one on /goal_pose (next goal when the robot is within --reach m of it or after --goal-timeout s of
simulated time), and records the screen from the first goal to the last arrival. Goals: --goal x,y (repeatable),
default: --goals points along the map's mapping path (data/demo/drives/<world>_map.npz), >= 6 m apart.
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def pick_goals(gt: np.ndarray, n: int, seed: int, min_sep: float = 6.0):
    rng = np.random.default_rng(seed)
    goals = []
    for i in rng.permutation(np.linspace(0, len(gt) - 1, 4 * n).astype(int)):
        p = gt[i, :2]
        if np.hypot(*(p - gt[0, :2])) > min_sep and all(np.hypot(*(p - q)) > min_sep for q in goals):
            goals.append(p)
        if len(goals) == n:
            break
    return goals


def grab(display, W, H):
    """One RGB frame of the display."""
    raw = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "0", "-video_size",
                          f"{W}x{H}", "-i", f"{display}.0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)


def collapse_left_dock(env, x, y):
    """RViz2 ignores "Hide Left Dock" when started: click the dock's collapse arrow (its right edge, mid-height; at
    x = 460 for a 1920 px wide window with config/rtisslam_record.rviz)."""
    subprocess.run(["xdotool", "mousemove", str(x), str(y), "click", "1"], env=env)
    time.sleep(1)
    subprocess.run(["xdotool", "mousemove", "1919", "1079"], env=env)
    time.sleep(1)


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:                     # system python (rclpy): use the venv's imageio-ffmpeg binary if there is one
        found = sorted((Path.home() / "venvs" / "rtisslam").glob("lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg*"))
        return str(found[-1]) if found else "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="office")
    ap.add_argument("--data", default=str(PROJECT / "data" / "demo"))
    ap.add_argument("--localizer", default="pf")
    ap.add_argument("--rtf", type=float, default=2.0)
    ap.add_argument("--goals", type=int, default=3)
    ap.add_argument("--goal", action="append", default=[], help="x,y in the map (repeatable)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--reach", type=float, default=0.5)
    ap.add_argument("--goal-timeout", type=float, default=240.0, help="s of simulated time per goal")
    ap.add_argument("--display", default=":99")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dock-arrow", default="460,556", help="x,y of RViz2's left-dock collapse arrow ('' = keep the dock)")
    ap.add_argument("--domain", type=int, default=87)
    ap.add_argument("--out", default=str(PROJECT / "results" / "ros_demo.mp4"))
    ap.add_argument("--rviz-config", default=str(PROJECT / "ros" / "rtisslam_bringup" / "config" / "rtisslam_record.rviz"),
                    help="RViz2 layout for the recording (whole office, no side panels)")
    a = ap.parse_args()

    if a.goal:
        goals = [np.array([float(v) for v in g.split(",")]) for g in a.goal]
    else:
        with np.load(Path(a.data) / "drives" / f"{a.world}_map.npz") as z:
            goals = pick_goals(z["gt"], a.goals, a.seed)
    print("goals:", [tuple(np.round(g, 2)) for g in goals], flush=True)

    W, H = (int(v) for v in a.size.split("x"))
    env = dict(os.environ, DISPLAY=a.display, ROS_DOMAIN_ID=str(a.domain))
    os.environ.update(DISPLAY=a.display, ROS_DOMAIN_ID=str(a.domain))
    procs = []
    if subprocess.run(["xdotool", "getdisplaygeometry"], env=env, capture_output=True).returncode != 0:
        procs.append(subprocess.Popen(["Xvfb", a.display, "-screen", "0", f"{W}x{H}x24", "+extension", "GLX",
                                       "-nolisten", "tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(2)
    log = open(Path(a.out).with_suffix(".log"), "w")
    demo = subprocess.Popen([sys.executable, str(PROJECT / "tools" / "demo.py"), "--world", a.world, "--data", a.data,
                             "--localizer", a.localizer, "--rtf", str(a.rtf), "--rviz-config", a.rviz_config], env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    rec = None
    try:
        # RViz2 window: full screen (no window manager on Xvfb: move and resize it directly)
        wid = None
        t0 = time.time()
        while wid is None and time.time() - t0 < 120:
            r = subprocess.run(["xdotool", "search", "--name", "RViz"], env=env, capture_output=True, text=True)
            wid = r.stdout.split()[-1] if r.stdout.split() else None
            time.sleep(1)
        if wid is None:
            sys.exit("RViz2 did not open")
        subprocess.run(["xdotool", "windowmove", wid, "0", "0", "windowsize", wid, str(W), str(H)], env=env)
        time.sleep(3)
        if a.dock_arrow:
            collapse_left_dock(env, *a.dock_arrow.split(","))

        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rosgraph_msgs.msg import Clock
        rclpy.init()
        node = Node("demo_recorder", parameter_overrides=[])
        st = {"gt": None, "t": 0.0}
        node.create_subscription(Odometry, "/ground_truth/odom", lambda m: st.__setitem__(
            "gt", np.array([m.pose.pose.position.x, m.pose.pose.position.y])), 10)
        node.create_subscription(Clock, "/clock", lambda m: st.__setitem__("t", m.clock.sec + 1e-9 * m.clock.nanosec), 10)
        pub = node.create_publisher(PoseStamped, "/goal_pose", 10)
        while st["gt"] is None or st["t"] < 3.0:              # simulator up, map and localizer settled
            rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(3.0)
        rec = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "0", "-framerate", str(a.fps),
                                "-video_size", f"{W}x{H}", "-i", f"{a.display}.0", "-c:v", "libx264", "-preset",
                                "medium", "-crf", "20", "-pix_fmt", "yuv420p", a.out], env=env, stdin=subprocess.PIPE)
        time.sleep(2.0)
        for i, g in enumerate(goals):
            m = PoseStamped()
            m.header.frame_id = "map"
            m.header.stamp = node.get_clock().now().to_msg()
            m.pose.position.x, m.pose.position.y = float(g[0]), float(g[1])
            m.pose.orientation.w = 1.0
            pub.publish(m)
            t_goal = st["t"]
            print(f"goal {i}: {g.round(2)} at sim t {t_goal:.1f}", flush=True)
            while np.hypot(*(st["gt"] - g)) > a.reach and st["t"] - t_goal < a.goal_timeout:
                rclpy.spin_once(node, timeout_sec=0.1)
            print(f"  {'reached' if np.hypot(*(st['gt'] - g)) <= a.reach else 'TIMEOUT'} after "
                  f"{st['t'] - t_goal:.1f} s simulated", flush=True)
        time.sleep(3.0)
        node.destroy_node()
        rclpy.shutdown()
    finally:
        if rec is not None:
            rec.communicate(b"q", timeout=30)
        os.killpg(demo.pid, signal.SIGINT)
        try:
            demo.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(demo.pid, signal.SIGKILL)
        for p in procs:
            p.terminate()
        log.close()
    print(f"{a.out} written", flush=True)


if __name__ == "__main__":
    main()
