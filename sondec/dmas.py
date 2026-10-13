"""Delay-Multiply-and-Sum (DMAS) beamforming with optional Coherence Factor, as in
Jansen, Daems, Steckel, "Delay-Multiply-And-Sum Beamforming for Real-Time In-Air Acoustic Imaging"
(arXiv 2511.09165). A BASELINE here (no learning), computed from the raw microphone signals.

Pipeline (their Fig. 1): matched filter -> pre-steering delays (NOMINAL array, far field; exact fractional
delays in the frequency domain) -> beamformer -> envelope (|.| then low-pass).
    signed n-th roots   s_i = sgn(x_i) |x_i|^(1/n)                                      (their eq. 4)
    DMAS order n        S_n = e_n(s_1..s_N), the n-th elementary symmetric polynomial    (eq. 5),
                        from the power sums P_k = sum_i s_i^k (Newton-Girard, eqs. 9-10):
                        S_2 = (P1^2 - P2) / 2,   S_3 = (P1^3 + 2 P3 - 3 P1 P2) / 6
    Coherence Factor    CF = (sum_i x_i)^2 / (N sum_i x_i^2 + eps)                       (eq. 13)
    output              S_n (x CF)                                                       (eq. 14)
Envelope: |.|, then a low-pass (FFT brick wall at LP_HZ); the energyscape is the mean envelope power per
range cell, as for DAS (sondec/das.py), so the images are directly comparable.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .das import BAND, NFFT, RangeGrid
from .sensor import C_SOUND, FS, chirp, load_array

LP_HZ = 30e3


class DMAS:
    def __init__(self, dirs: torch.Tensor, grid: RangeGrid = RangeGrid(), device="cuda"):
        self.dev, self.grid = device, grid
        p = torch.as_tensor(np.asarray(load_array()["nominal"]), dtype=torch.float32, device=device)
        self.M, self.D = len(p), dirs.shape[0]
        f = np.fft.rfftfreq(NFFT, 1 / FS)
        self.k0, self.k1 = int(np.ceil(BAND[0] / f[1])), int(np.floor(BAND[1] / f[1])) + 1
        fb = torch.as_tensor(f[self.k0:self.k1], dtype=torch.float32, device=device)
        C = np.fft.rfft(chirp(), NFFT)[self.k0:self.k1]
        self.MF = torch.as_tensor(np.conj(C) / np.sum(np.abs(C) ** 2) * NFFT, dtype=torch.complex64, device=device)
        pu = p @ dirs.T.float()                                                     # [M, D]
        self.steer = torch.polar(torch.ones_like(pu)[None], -2 * math.pi * fb[:, None, None] * pu[None] / C_SOUND)  # [K, M, D]
        self.lp = torch.as_tensor(f <= LP_HZ, dtype=torch.float32, device=device)  # [NFFT/2+1]
        edges = grid.r_min + grid.dr * np.arange(grid.n_r + 1)
        self.edges = torch.as_tensor(np.round(edges * 2 / C_SOUND * FS).astype(np.int64), device=device)

    def _cells(self, env: torch.Tensor) -> torch.Tensor:
        p = env ** 2
        cs = torch.nn.functional.pad(torch.cumsum(p.double(), -1), (1, 0))
        a, b = self.edges[:-1], self.edges[1:]
        return ((cs[..., b] - cs[..., a]) / (b - a).double()).float()

    def _envelope(self, y: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft(torch.fft.rfft(y.abs(), dim=-1) * self.lp, NFFT, dim=-1).clamp_min(0)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, variants=(("dmas2", 2, False), ("dmas2cf", 2, True), ("dmas3cf", 3, True)),
                 chunk: int = 24) -> dict:
        """x [M, NSIG] one pulse -> {name: [R, D] energyscape (linear power)}."""
        X = torch.fft.rfft(x, NFFT)[:, self.k0:self.k1] * self.MF                   # [M, K]
        out = {nm: [] for nm, _, _ in variants}
        for d0 in range(0, self.D, chunk):
            st = self.steer[:, :, d0:d0 + chunk]                                    # [K, M, d]
            Z = torch.zeros(st.shape[2], self.M, NFFT // 2 + 1, dtype=torch.complex64, device=self.dev)
            Z[..., self.k0:self.k1] = (X.T[:, :, None] * st).permute(2, 1, 0)
            xd = torch.fft.irfft(Z, NFFT, dim=-1)                                   # [d, M, T] delayed real signals
            del Z
            cf = None
            for nm, n, use_cf in variants:
                s = torch.sign(xd) * xd.abs().pow(1.0 / n)
                P1 = s.sum(1); P2 = (s * s).sum(1)
                if n == 2:
                    S = 0.5 * (P1 ** 2 - P2)
                else:
                    P3 = (s ** 3).sum(1)
                    S = (P1 ** 3 + 2 * P3 - 3 * P1 * P2) / 6
                del s
                if use_cf:
                    if cf is None:
                        cf = xd.sum(1) ** 2 / (self.M * (xd ** 2).sum(1) + 1e-30)
                    S = S * cf
                out[nm].append(self._cells(self._envelope(S)))                       # [d, R]
        return {nm: torch.cat(v).T.contiguous() for nm, v in out.items()}            # [R, D]
