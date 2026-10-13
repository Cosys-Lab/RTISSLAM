"""Classical PSF-based deconvolution baselines on the DAS energyscape (2D or 3D direction sets):

  DAMAS   non-negative least squares  min ||A E - P||^2, E >= 0 (Brooks & Humphreys 2006), solved with the
          multiplicative ISRA update  E <- E * A^T P / A^T A E  (same minimiser as DAMAS's Gauss-Seidel sweeps)
  RL      Richardson-Lucy / DAMAS2-type multiplicative update for the Poisson (I-divergence) fit
          E <- E * A^T (P / A E) / A^T 1      (DAMAS2 is RL with a shift-invariant PSF, Dougherty 2005)
  CLEAN   Hogbom-type CLEAN with the PSF: repeatedly remove a fraction (loop gain) of the PSF at the residual
          maximum and store the removed component. (CLEAN-SC proper needs per-frequency cross-spectral
          matrices, which a single pulse-echo snapshot does not provide.)

A is the imaging operator of model.ForwardModel (PSF K[dr, u, v] times the range attenuation), with either
  - the NOMINAL PSF: the forward model at its initialisation (nominal array and emitter position, flat
    directivity, the call's spectrum), i.e. what can be computed from the design; or
  - the TRUE PSF (oracle, "perfect calibration"): measured in the simulator, one point reflector per source
    direction through the true array, the true emitter and DAS (true_psf()).
P is the measured energyscape minus the noise floor (clipped at 0). The outputs get the same uniform output
blur as the network (model.clean) for a fair comparison.
"""
from __future__ import annotations

import numpy as np
import torch

from .das import DAS
from .paths import data_dir
from .sim.echoes import ArraySynth

EMPTY = {k: np.zeros((0, 3)) for k in ["pos", "normal", "mat"]}
EMPTY.update(area=np.zeros(0), two=np.zeros(0, bool), obj=np.zeros(0, int), obj_size=np.zeros(0),
             obj_tilt=np.zeros(0), boxes=np.zeros((0, 7)), capsules=np.zeros((0, 7)))


@torch.no_grad()
def true_psf(S, grid, emitter: str, n_dr: int, r0: float = 2.0, n_off: int = 4) -> torch.Tensor:
    """ORACLE: K_true [2 n_dr + 1, D_u, D_v], normalised like the model (mean diagonal peak 1). Cached."""
    cache = data_dir() / f"true_psf_{getattr(S, 'dim', 3)}d_D{S.D}_{emitter}_n{n_dr}.npy"
    if cache.exists():
        return torch.as_tensor(np.load(cache), device=S.device)
    syn, das = ArraySynth(EMPTY, emitter=emitter, noise=0.0), DAS(S.dirs, grid)
    i0 = int((r0 - grid.r_min) / grid.dr)
    K = torch.zeros(2 * n_dr + 1, S.D, S.D, device=S.device)
    for j in range(S.D):
        xs = []
        for off in (np.arange(n_off) + 0.5) / n_off:
            rr = grid.r_min + (i0 + off) * grid.dr
            xs.append(syn.pulse(None, None, points={"p": (rr * S.dirs_np[j])[None], "amp": np.array([0.1])})[0])
        P = das(torch.stack(xs), chunk=n_off).mean(0)                     # [R, D_u]
        K[:, :, j] = P[i0 - n_dr:i0 + n_dr + 1] * (r0 ** 4)              # undo the spreading at r0
    K = K / torch.diagonal(K, dim1=1, dim2=2).amax(0).mean()
    np.save(cache, K.cpu().numpy())
    return K


class Operator:
    """A E = fwd(E, K) (PSF + range attenuation of the forward model), and its adjoint."""

    def __init__(self, fwd, K: torch.Tensor):
        self.fwd, self.K = fwd, K
        self.fwd_n = fwd.n_dr
        fwd.n_dr = (K.shape[0] - 1) // 2                                  # the oracle may have another width

    def __call__(self, E):
        return self.fwd(E, self.K)

    def T(self, X):
        return self.fwd.adjoint(X, self.K)


@torch.no_grad()
def damas(P: torch.Tensor, A: Operator, iters: int = 2000) -> torch.Tensor:
    AtP = A.T(P).clamp_min(0)
    E = torch.full_like(P, float(AtP.mean()) * 1e-2 + 1e-12)
    for _ in range(iters):
        E = E * AtP / A.T(A(E)).clamp_min(1e-30)
    return E


@torch.no_grad()
def richardson_lucy(P: torch.Tensor, A: Operator, iters: int = 2000, floor: float = 0.0) -> torch.Tensor:
    At1 = A.T(torch.ones_like(P)).clamp_min(1e-30)
    E = torch.full_like(P, float((A.T(P) / At1).mean()) + 1e-12)
    for _ in range(iters):
        E = E * A.T(P / (A(E) + floor).clamp_min(1e-30)) / At1
    return E


@torch.no_grad()
def clean(P: torch.Tensor, A: Operator, iters: int = 1500, gain: float = 0.2, stop_rel: float = 1e-4) -> torch.Tensor:
    """P [B, R, D] -> clean components [B, R, D]."""
    B, R, D = P.shape
    res = P.clone()
    comp = torch.zeros_like(P)
    # response of a unit component at (r, v), peak value at its own cell, for the scaling of each step
    unit = torch.zeros(1, R, D, device=P.device)
    pk = torch.zeros(R, D, device=P.device)
    rs = torch.arange(R, device=P.device)
    for v in range(D):                                                   # peak of A delta_(r, v) over r (cheap: diagonal)
        unit.zero_(); unit[0, rs, v] = 1.0
        pk[:, v] = A(unit)[0, rs, v]
    pk = pk.clamp_min(1e-30)
    start = res.flatten(1).amax(1)
    for _ in range(iters):
        m, idx = res.flatten(1).max(1)
        if bool((m < stop_rel * start).all()):
            break
        r_i, v_i = idx // D, idx % D
        c = gain * m / pk[r_i, v_i]
        unit = torch.zeros_like(P)
        unit[torch.arange(B), r_i, v_i] = c * (m >= stop_rel * start)
        comp += unit
        res = res - A(unit)
    return comp
