"""The eRTIS on the robot: one pulse at a 2D robot pose -> 32 microphone signals -> 2D energyscapes.

Synthesis: sondec.sim.echoes.ArraySynth (coherent per-microphone spectra, true array errors, realistic emitter v2,
white sensor noise 5e-4 as in the SPL dataset). The sensor sits on the robot at MOUNT = (x, y, z) in base_link,
facing forward, level: R_WS = Rz(yaw), t_WS = robot position + Rz(yaw) MOUNT.

Imaging, all on the same range x azimuth grid (240 cells of 2 cm from 0.2 m, 181 azimuths over -90..90 deg, el = 0),
10 log10 of the mean envelope power per cell (dB, the SPL dataset's units):
    das       delay-and-sum (sondec.das, NOMINAL array)
    dmas2     delay-multiply-and-sum, order 2               } Jansen, Daems, Steckel (arXiv 2511.09165),
    dmas2cf   DMAS2 x coherence factor                       } sondec.dmas
    dmas3cf   DMAS3 x coherence factor                       }
    deconv    the self-supervised deconvolution network (SPL model, spl_main_s0) applied to the DAS map
              (Model.infer_db: reflectivity seen through the range/beam attenuation with a delta PSF)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from sondec.das import DAS, RangeGrid
from sondec.dmas import DMAS
from sondec.paths import data_dir
from sondec.sim.echoes import ArraySynth
from sondec.sphere import Circle

METHODS = ("das", "dmas2", "dmas2cf", "dmas3cf", "deconv")
MOUNT = (0.10, 0.0, 0.40)          # m, sensor origin in base_link (sensor plane 0.4 m above the floor)
NOISE = 5e-4
D_AZ = 181


def default_ckpt() -> Path:
    return data_dir() / "models" / "spl_main_s0.pt"


class RtisSensor:
    def __init__(self, world: dict, device: str = "cpu", methods=METHODS, noise: float = NOISE, mount=MOUNT,
                 emitter: str = "v2", ckpt: str | Path | None = None, seed: int = 0):
        self.dev, self.methods, self.mount = device, tuple(methods), np.asarray(mount, float)
        self.synth = ArraySynth(world, device=device, noise=noise, emitter=emitter)
        self.circle = Circle(D_AZ, device=device)
        self.grid = RangeGrid()
        self.das = DAS(self.circle.dirs, self.grid, device=device)
        self.dmas_variants = tuple(v for v in (("dmas2", 2, False), ("dmas2cf", 2, True), ("dmas3cf", 3, True))
                                   if v[0] in self.methods)
        self.dmas = DMAS(self.circle.dirs, self.grid, device=device) if self.dmas_variants else None
        self.model = None
        if "deconv" in self.methods:
            from sondec.train_lib import load_model
            self.model, meta = load_model(ckpt or default_ckpt(), device)
            assert meta["D"] == D_AZ and meta.get("dim") == 2 and tuple(meta["grid"]) == (self.grid.r_min, self.grid.dr, self.grid.n_r)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.timing: dict[str, float] = {}

    # ------------------------------------------------------------------ geometry
    @property
    def ranges(self) -> np.ndarray:
        return self.grid.r("cpu").numpy()

    @property
    def azimuths(self) -> np.ndarray:
        return self.circle.az.astype(np.float32)

    def sensor_pose(self, pose) -> tuple[torch.Tensor, torch.Tensor]:
        """Robot pose (x, y, yaw) -> (R_WS [3, 3], t_WS [3])."""
        x, y, th = (float(v) for v in pose)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([x, y, 0.0]) + R @ self.mount
        T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.dev)  # noqa: E731
        return T(R), T(t)

    # ------------------------------------------------------------------ pulse + imaging
    @torch.no_grad()
    def pulse(self, pose, top_k: int = 64):
        """-> (x [M, NSIG] noisy microphone signals, contributions of the strongest echoes: EVALUATION ONLY)."""
        R, t = self.sensor_pose(pose)
        x, contrib = self.synth.pulse(R, t, top_k)
        return self.synth.add_noise(x, self.gen), contrib

    @torch.no_grad()
    def image(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        """x [M, NSIG] -> {method: [R, D] float32 dB}."""
        out, tm = {}, {}
        t0 = time.perf_counter()
        das_db = 10 * torch.log10(self.das(x[None])[0] + 1e-30)
        tm["das"] = time.perf_counter() - t0
        if "das" in self.methods:
            out["das"] = das_db
        if self.dmas is not None:
            t0 = time.perf_counter()
            for nm, P in self.dmas(x, variants=self.dmas_variants).items():
                out[nm] = 10 * torch.log10(P + 1e-30)
            tm["dmas"] = time.perf_counter() - t0
        if self.model is not None:
            t0 = time.perf_counter()
            out["deconv"] = self.model.infer_db(das_db[None])[0]
            tm["deconv"] = time.perf_counter() - t0
        self.timing.update(tm)
        return {k: out[k].float().cpu().numpy() for k in self.methods}

    def render(self, pose, top_k: int = 64):
        """Robot pose -> ({method: [R, D] dB}, contributions)."""
        t0 = time.perf_counter()
        x, contrib = self.pulse(pose, top_k)
        self.timing["synth"] = time.perf_counter() - t0
        return self.image(x), {k: v.cpu().numpy() for k, v in contrib.items()}
