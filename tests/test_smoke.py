"""End-to-end smoke test (no dataset, no ROS): simulate a short drive in the small "loop" floor plan with the
eRTIS (DMAS3 + coherence factor), build an RTISSLAM map, save and load it, and localize in it with the particle filter.

    python -m pytest tests/test_smoke.py -s

Runs on the GPU when torch sees one (seconds), otherwise on the CPU (about 3 s per pulse, 2-3 minutes in total).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_PULSES = 40            # mapping pulses (gated every GATE_M of travel)
GATE_M = 0.15


def _drive(n, seed):
    """Pure pursuit along the plan's driving line; one DMAS3-CF energyscape every GATE_M of odometry travel."""
    import torch

    from rtisslam.imaging_post import POST
    from rtissim.robot import PurePursuit
    from rtissim.sim import Simulator
    sim = Simulator("loop", device="cuda" if torch.cuda.is_available() else "cpu", methods=("dmas3cf",), seed=seed)
    pp = PurePursuit(sim.line, v_max=0.3)
    out, last = [], None
    while len(out) < n:
        v, w = pp.command(sim.robot.pose)
        obs = sim.step(v, w, fire=False)
        o = obs["odom"]
        if last is not None and math.hypot(*(o[:2] - last[:2])) < GATE_M:
            continue
        last = o.copy()
        m, _ = sim.sensor.render(sim.robot.pose)
        out.append(({"dmas3cf:ams2s": POST["ams2s"](m["dmas3cf"])}, o.copy(), obs["gt"].copy()))
    return sim, out


@pytest.fixture(scope="module")
def mapped(tmp_path_factory):
    from rtisslam.system import RTISSLAM, RTISSLAMConfig
    sim, drive = _drive(N_PULSES, seed=0)
    cfg = RTISSLAMConfig()
    assert cfg.descriptor.method == cfg.grid.method == "dmas3cf:ams2s"
    slam = RTISSLAM(cfg, sim.sensor.ranges, sim.sensor.azimuths, initial_pose=tuple(drive[0][2]))
    for maps, odom, _ in drive:
        slam.process(maps, odom, force=True)
    path = tmp_path_factory.mktemp("map") / "map.npz"
    slam.save(path)
    return slam, drive, path


def test_mapping(mapped):
    slam, drive, _ = mapped
    P = slam.poses()
    gt = np.array([d[2] for d in drive])
    assert len(P) == len(drive) and np.all(np.isfinite(P))
    err = np.hypot(*(P[:, :2] - gt[:, :2]).T)
    assert err.max() < 1.0, f"mapping error {err.max():.2f} m on a 6 m drive"
    assert len(slam.slam.templates) > 0


def test_save_load_localize(mapped):
    from rtisslam.system import RTISSLAM
    _, drive, path = mapped
    loc = RTISSLAM.load(path)
    assert len(loc.poses()) == len(drive)
    # a second pass over the same route with different sensor and odometry noise
    _, again = _drive(15, seed=1)
    loc.start_localization(tuple(again[0][2]), (0.3, 0.3, 0.1), localizer="pf")
    for maps, odom, _ in again:
        loc.process(maps, odom)
    est, gt = loc.pose(), again[-1][2]
    assert np.all(np.isfinite(est)) and np.all(np.isfinite(loc.covariance()))
    assert math.hypot(est[0] - gt[0], est[1] - gt[1]) < 1.0
