"""Final RTISSLAM runs and ablation on the 2-lap mapping drives (seeds: drives map2 = seed 0, map2s1..map2s4), with
the final configuration (rtisslam/system.py: DMAS3 + CF + smoothed RW-CFAR 2 for every role), all in parallel.

    python scripts/run_final.py [--worlds office warehouse atrium] [--seeds 0 1 2 3 4] [--jobs 12] [--only final]
Writes RTISSLAM_DATA/runs/<world>_dm_<variant>_s<seed> (seed 0 of "final" with graph snapshots every 4 nodes, for
the video).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sondec.paths import data_dir  # noqa: E402

BASE = ["--imaging", "dmas3cf", "--post", "ams2s", "--grid-imaging", "dmas3cf", "--grid-post", "ams2s",
        "--ao-imaging", "dmas3cf", "--ao-post", "ams2s", "--register-on", "descriptor"]
VARIANTS = {
    "final": [],
    "final_wheel": ["--no-links", "--no-ao-calib"],                  # wheel odometry only
    "final_calibdr": ["--no-links"],                                  # calibrated dead reckoning
    "final_aofactors": ["--no-links", "--no-ao-calib", "--ao-factors"],   # acoustic odometry as factors, no links
    "final_links_noreg_nocal": ["--no-register", "--no-ao-calib"],    # BatSLAM 2.0 links only
    "final_links_nocal": ["--no-ao-calib"],                           # + registration
    "final_links_noreg": ["--no-register"],                           # + calibration, no registration
    "final_full_aofactors": ["--ao-factors"],                         # full + acoustic odometry factors
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["office", "warehouse", "atrium"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    logs = data_dir() / "runs" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads))
    jobs = []
    for s in a.seeds:
        tag = "map2" if s == 0 else f"map2s{s}"
        for v, extra in VARIANTS.items():
            if a.only is not None and v not in a.only:
                continue
            snap = ["--snap", "4"] if (v == "final" and s == 0) else ["--snap", "1000000"]
            for w in a.worlds:
                name = f"dm_{v}_s{s}"
                jobs.append((f"{w}_{name}", [sys.executable, str(ROOT / "scripts" / "run_slam.py"), w, "--tag", tag,
                                             "--variant", name] + BASE + extra + snap))
    # the final runs (with snapshots) take longest: start them first
    jobs.sort(key=lambda j: ("_wheel_" in j[0] or "_calibdr_" in j[0] or "_aofactors_" in j[0], "_final_s" not in j[0]))

    def run(job):
        name, cmd = job
        with open(logs / f"{name}.log", "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {name}", flush=True)
        return r.returncode

    with ThreadPoolExecutor(a.jobs) as ex:
        codes = list(ex.map(run, jobs))
    print(f"done: {codes.count(0)}/{len(codes)} ok", flush=True)


if __name__ == "__main__":
    main()
