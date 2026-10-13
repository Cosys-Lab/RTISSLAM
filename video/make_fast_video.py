"""Shorter cut of the combined RTISSLAM video, edited from the rendered parts (no re-rendering of frames).

    python video/make_fast_video.py [--anonymous]
Slides play at normal speed. Every data sequence starts at normal speed for a few seconds, after which it is sped up,
with a fast-forward symbol and the playback speed (relative to real time) in the top right corner:
    SLAM, per world           first 8 s at 1x real time, then 4x
    navigation, per world     first 8 s at 4x real time (as rendered), then 16x; the "Playback" status line is
                              covered while sped up, as the symbol gives the speed
    ROS 2 demo                first 10 s at 2x real time (as recorded), then 6x
Inputs: RTISSLAM_DATA/video_parts/comb_<part>[_anon].mp4 (video/make_combined_video.py) and results/ros_demo_2x.mp4.
Writes results/rtisslam_video_fast[_anonymous].mp4 and copies it to paper/video/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sondec.paths import data_dir  # noqa: E402

# part boundaries (s) of the rendered parts: slide, then one sequence per world (frame counts / 10 fps)
SLAM_WORLDS = [8.0, 107.8, 182.5, 243.8]          # slide 0-8 s, office, warehouse, atrium
NAV_WORLDS = [7.0, 163.0, 288.4, 473.2]           # slide 0-7 s, office, warehouse, atrium
HEAD = 8.0                                         # seconds at normal speed at the start of every sequence


def ff_icon(path: Path, label: str):
    """Fast-forward symbol (two triangles) and the playback speed, white on a dark rounded box (PNG with alpha)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon
    fig = plt.figure(figsize=(2.55, 0.68), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 3.4); ax.set_ylim(0, 0.9); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.05, 0.05), 3.3, 0.8, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=(0, 0, 0, 0.6), ec="none"))
    for x0 in (0.25, 0.62):
        ax.add_patch(Polygon([(x0, 0.22), (x0, 0.68), (x0 + 0.37, 0.45)], closed=True, fc="white", ec="none"))
    ax.text(1.15, 0.45, label, color="white", fontsize=16.5, va="center", ha="left", family="serif")
    fig.savefig(path, transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anonymous", action="store_true")
    a = ap.parse_args()
    sfx = "_anon" if a.anonymous else ""
    parts = data_dir() / "video_parts"
    icons = {}
    for k, lab in (("slam", "4× real time"), ("nav", "16× real time"), ("ros", "6× real time")):
        icons[k] = parts / f"_ff_{k}.png"
        ff_icon(icons[k], lab)
    ros = ROOT / "results" / "ros_demo_2x.mp4"
    # segments: (input file, start, end, speed-up, icon key or None, cover the playback status line)
    segs = [(parts / f"comb_intro{sfx}.mp4", 0, None, 1, None, False),
            (parts / f"comb_slam{sfx}.mp4", 0, SLAM_WORLDS[0], 1, None, False)]
    for t0, t1 in zip(SLAM_WORLDS[:-1], SLAM_WORLDS[1:]):
        segs += [(parts / f"comb_slam{sfx}.mp4", t0, t0 + HEAD, 1, None, False),
                 (parts / f"comb_slam{sfx}.mp4", t0 + HEAD, t1, 4, "slam", False)]
    segs += [(parts / f"comb_results{sfx}.mp4", 0, None, 1, None, False),
             (parts / f"comb_nav{sfx}.mp4", 0, NAV_WORLDS[0], 1, None, False)]
    for t0, t1 in zip(NAV_WORLDS[:-1], NAV_WORLDS[1:]):
        segs += [(parts / f"comb_nav{sfx}.mp4", t0, t0 + HEAD, 1, None, False),
                 (parts / f"comb_nav{sfx}.mp4", t0 + HEAD, t1, 4, "nav", True)]
    segs += [(parts / f"comb_ros{sfx}.mp4", 0, None, 1, None, False),
             (ros, 0, 10.0, 1, None, False), (ros, 10.0, None, 3, "ros", False),
             (parts / f"comb_end{sfx}.mp4", 0, None, 1, None, False)]
    files = sorted({str(s[0]) for s in segs})
    idx = {f: i for i, f in enumerate(files)}
    icon_idx = {k: len(files) + i for i, k in enumerate(icons)}
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    for f in files:
        cmd += ["-i", f]
    for k in icons:
        cmd += ["-loop", "1", "-i", str(icons[k])]
    chains, labels = [], []
    for n, (f, t0, t1, sp, ic, cover) in enumerate(segs):
        trim = f"trim=start={t0}" + (f":end={t1}" if t1 is not None else "")
        c = (f"[{idx[str(f)]}:v]{trim},setpts=(PTS-STARTPTS)/{sp},fps=30,"
             f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white,setsar=1")
        if cover:                                          # the "Playback: 4x real time" status line of the nav frames
            c += ",drawbox=x=20:y=392:w=480:h=44:color=white:t=fill"
        if ic is not None:
            c += f"[b{n}];[b{n}][{icon_idx[ic]}:v]overlay=W-w-40:150:shortest=1"
        c += f"[s{n}]"
        chains.append(c)
        labels.append(f"[s{n}]")
    fc = ";".join(chains) + ";" + "".join(labels) + f"concat=n={len(segs)}:v=1:a=0[out]"
    out = ROOT / "results" / f"rtisslam_video_fast{'_anonymous' if a.anonymous else ''}.mp4"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    dst = ROOT / "paper" / "video" / f"RTISSLAM_video_short{'_anonymous' if a.anonymous else ''}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, dst)
    print("->", out, "and", dst)


if __name__ == "__main__":
    main()
