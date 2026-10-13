"""Run scripts/run_slam.py over imaging x post-processing variants and worlds in parallel.

    python scripts/sweep_imaging.py --variants deconv:none dmas3cf:ams2s --worlds office warehouse atrium [--ao own]
Runs are written to RTISSLAM_DATA/runs/<world>_<prefix>_<imaging>_<post> (prefix ev / evao, + 2 for --tag map2); --ao fixed keeps the acoustic odometry on
deconv:none (isolates place recognition), --ao own feeds it the same energyscape as the descriptor.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--worlds", nargs="+", default=["office", "warehouse", "atrium"])
    ap.add_argument("--ao", default="fixed", choices=["fixed", "own"])
    ap.add_argument("--tag", default="map", help="drive tag: map (1 lap) | map2 (2 laps)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed on to run_slam.py")
    a = ap.parse_args()
    prefix = ("ev" if a.ao == "fixed" else "evao") + ("" if a.tag == "map" else a.tag.replace("map", ""))
    logs = data_dir() / "runs" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads),
               MKL_NUM_THREADS=str(a.threads))
    jobs = []
    for w in a.worlds:
        for v in a.variants:
            im, post = v.split(":")
            name = f"{prefix}_{im}_{post}"
            cmd = [sys.executable, str(ROOT / "scripts" / "run_slam.py"), w, "--imaging", im, "--post", post,
                   "--variant", name, "--snap", "1000000", "--tag", a.tag] + a.extra
            if a.ao == "fixed":
                cmd += ["--ao-imaging", "deconv", "--ao-post", "none"]
            jobs.append((f"{w}_{name}", cmd))

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
