"""The RTISSLAM paper video (Cosys-Lab style): all results in one video.

    python video/make_combined_video.py [--parts intro slam results nav ros end] [--test] [--out results/rtisslam_video.mp4]

Parts (each rendered to RTISSLAM_DATA/video_parts/comb_<part>.mp4, then joined at 30 fps; missing inputs are skipped):
    intro    title, what is shown, the three worlds (paper/figures/worlds.pdf)
    slam     the SLAM system at work on the 2-lap drives (runs/<world>_dm_final_s0, video/make_slam_video.py)
    results  the trajectory errors over five seeds and the grid-SLAM baseline (paper figures, if present)
    nav      autonomous navigation on the RTISSLAM maps (nav/<world>_nav_dm.npz, video/make_nav_video.py)
    ros      the RViz2 / Nav2 demo recorded in ROS 2 (results/ros_demo_2x.mp4)
    end      the end slide
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "video"))
import cosys_video as cv  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

WORLDS = ["office", "warehouse", "atrium"]
PARTS = ["intro", "slam", "results", "nav", "ros", "end"]


def pdf_image(pdf: Path, dpi: int = 200) -> np.ndarray | None:
    """First page of a vector figure as an RGB image (pdftoppm)."""
    if not pdf.exists():
        return None
    out = data_dir() / "video_parts" / f"_{pdf.stem}"
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf), str(out)], check=True)
    import imageio.v2 as iio
    return iio.imread(str(out) + ".png")[..., :3]


def part_intro(wr, test):
    cv.hold(wr, cv.slide_title("RTISSLAM: Sonar-Only SLAM\nWith a 3D Imaging Sonar"), 4.5, fade_in=False)
    cv.hold(wr, cv.slide_text("What Is Shown", [
        "A robot maps its environment with only an eRTIS imaging sonar and its wheel odometry.",
        "The pose graph of BatSLAM 2.0 estimates the trajectory, and an occupancy grid serves as the map.",
        "The energyscapes are formed with third-order DMAS with coherence factor and a range-wise CFAR.",
        "All data is simulated; the ground truth is shown for reference only."], wrap=80), 3 if test else 10)
    fig = pdf_image(ROOT / "paper" / "figures" / "worlds.pdf")
    cv.hold(wr, cv.slide_text("Three Worlds With Specular Walls", [
        "We drive two laps through an office, a warehouse and an atrium."], figure=fig, wrap=80), 3 if test else 7)


def part_slam(wr, test):
    import make_slam_video as sv
    cv.hold(wr, cv.slide_text("SLAM in Three Worlds", [
        "Left, the energyscape of the current pulse.",
        "In the middle, the wheel odometry (red), the RTISSLAM estimate (blue) and the loop closures.",
        "Right, the occupancy grid, which is rendered again after every loop closure."], wrap=80), 3 if test else 8)
    for i, w in enumerate(WORLDS):
        sv.world_frames(wr, w, i + 1, len(WORLDS), test, tag="map2", run_fmt="{world}_dm_final_s0")


def part_results(wr, test):
    img = pdf_image(ROOT / "paper" / "figures" / "slamWorlds.pdf")
    if img is not None:
        cv.hold(wr, cv.slide_text("RTISSLAM Versus Lidar-Style Grid SLAM", [
            "The bottom row shows slam_toolbox on the same sonar echoes and wheel odometry."],
            figure=img, wrap=80), 3 if test else 10)


def part_nav(wr, test):
    import make_nav_video as nv
    runs = [f"{w}_nav_dm" for w in WORLDS if (data_dir() / "nav" / f"{w}_nav_dm.npz").exists()]
    if not runs:
        return
    cv.hold(wr, cv.slide_text("Autonomous Navigation on the Map", [
        "The robot localizes in the saved map with a particle filter on the recognitions of BatSLAM 2.0.",
        "It plans its path on the occupancy grid and follows it with an acoustic-flow controller."], wrap=80), 3 if test else 7)
    for i, r in enumerate(runs):
        nv.run_frames(wr, r, i + 1, len(runs), test, stride=4)


def part_ros(wr, test):
    """A title slide here; the recording itself is joined in main() (its own frame rate)."""
    if (ROOT / "results" / "ros_demo_2x.mp4").exists():
        cv.hold(wr, cv.slide_text("ROS 2 Demo", [
            "The same system runs as ROS 2 nodes, with Nav2 and RViz2, shown here at twice the real speed."], wrap=80), 3 if test else 6)


def part_end(wr, test):
    cv.hold(wr, cv.slide_end(), 3.5, fade_out=False)


def anonymize():
    """Replace the slide and frame helpers of cosys_video by versions without logos."""
    import matplotlib.pyplot as plt

    def slide_title(title: str):
        fig = cv.new_fig(cv.MAGENTA)
        fig.text(0.5, 0.56, title, ha="center", va="center", fontsize=46, color="white", fontweight="bold",
                 linespacing=1.35, **cv.SANS)
        img = cv.grab(fig); plt.close(fig); return img

    def slide_end():
        fig = cv.new_fig()
        fig.text(0.5, 0.5, "Supplementary Video", ha="center", va="center", fontsize=36)
        img = cv.grab(fig); plt.close(fig); return img

    cv.logos = lambda fig, white=False: None
    cv.slide_title, cv.slide_end = slide_title, slide_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=PARTS)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--rerender", action="store_true", help="render parts that already exist again")
    ap.add_argument("--out", default=None, help="default: results/rtisslam_video[_anonymous].mp4")
    ap.add_argument("--anonymous", action="store_true", help="no lab or partner logos (double-anonymous review)")
    a = ap.parse_args()
    if a.anonymous:
        anonymize()
    a.out = a.out or str(ROOT / "results" / f"rtisslam_video{'_anonymous' if a.anonymous else ''}.mp4")
    parts_dir = data_dir() / "video_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    fns = {"intro": part_intro, "slam": part_slam, "results": part_results, "nav": part_nav, "ros": part_ros,
           "end": part_end}
    files = []
    for p in PARTS:
        f = parts_dir / f"comb_{p}{'_anon' if a.anonymous else ''}{'_test' if a.test else ''}.mp4"
        if p in a.parts and (a.rerender or not f.exists()):
            wr = cv.writer(str(f))
            fns[p](wr, a.test)
            wr.close()
            print(f"part {p} -> {f}", flush=True)
        if f.exists() and f.stat().st_size > 10_000:
            files.append(f)
        if p == "ros" and (ROOT / "results" / "ros_demo_2x.mp4").exists():
            files.append(ROOT / "results" / "ros_demo_2x.mp4")
    # join at 30 fps (the ROS recording has its own frame rate), 1920 x 1080
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    for f in files:
        cmd += ["-i", str(f)]
    chains = "".join(f"[{i}:v]fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,"
                     f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white,setsar=1[v{i}];" for i in range(len(files)))
    cmd += ["-filter_complex", chains + "".join(f"[v{i}]" for i in range(len(files))) +
            f"concat=n={len(files)}:v=1:a=0[out]", "-map", "[out]", "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p", a.out]
    subprocess.run(cmd, check=True)
    print("->", a.out, f"({len(files)} parts)")


if __name__ == "__main__":
    main()
