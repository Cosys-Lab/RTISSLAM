"""Self-supervised deconvolution of the eRTIS energyscape.

    observed level  l_t[r, u]  (DAS energyscape, compressed)  --Deconvolver-->  E_t[r, v] >= 0  (reflectivity)
    E_t  --warp by the robot's own motion (acoustic flow)-->  E_{t->t+k}
    E    --ForwardModel (learned PSF of the whole imaging chain)-->  predicted level at t, t+k
    loss = || render(E_t) - l_t ||^2 + mean_k || render(warp_k E_t) - l_{t+k} ||^2 + lambda_1 |E|_1

As in HRTF SSL: only observed data are compared (never two encoder outputs); the motion is the teacher.
A DAS ghost (sidelobe, grating lobe) is not a reflector: it moves with the true reflector's
direction, not along its own acoustic flow, so explaining it as a reflector makes the prediction of
the next pulses wrong. Explaining it through the PSF does not.

ForwardModel (fixed physics, learned sensor): the DAS energyscape of a point reflector at (r0, v),
steered at u, around range r0 + x:

    y_{u,v}(x) = sum_f S_f G(v, f) B_f(u, v) exp(i 2 pi f (2 x + e . v) / c)      (e: emitter position)
    B_f(u, v)  = (1/M) sum_m g_m exp(i 2 pi f (p_m . v - p^nom_m . u) / c)

p^nom: the NOMINAL array the beamformer steers with (known: it is our own processing). Learned: the
TRUE microphone positions p_m (init nominal), complex gains g_m (init 1), the pulse spectrum S_f (init
|CALL|^2, the matched-filter output), the sensor beam G(v, f) (emitter x microphone directivity, dB, an
MLP of the direction, zero mean over the hemisphere: the overall level is a fixed gauge). The power
|y|^2 is averaged over range cells (a triangle kernel: a reflector anywhere within its cell) ->
K[dr, u, v] for dr = -n..n cells. Then

    P_hat[r, u] = gain0 sum_{dr, v} K[dr, u, v] E[r - dr, v] r^-4 10^(-2 alpha r / 10)

an incoherent sum (interference between reflectors in one cell is not modelled: speckle stays in
the residual).

Deconvolver (the network used at inference, ONNX-exportable: Conv, MatMul, GroupNorm, Softplus only):
a mixer over [range x direction]: alternating dense direction mixing (a learned D x D matrix per block:
the PSF is global, sidelobes reach the whole hemisphere) and dilated range convolutions, with channels.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .das import RangeGrid
from .sensor import C_SOUND, EMITTER_POS, F_HI, F_LO, chirp, load_array
from .sphere import Sphere


# ------------------------------------------------------------------------------------------ deconvolver
class ChebSphere(nn.Module):
    """Local spherical convolution: Chebyshev filter of order k_cheb on the mesh's graph Laplacian
    (DeepSphere-style), isotropic, per channel pair: y = sum_k W_k T_k(L~) x. Only MatMul + Conv (ONNX)."""

    def __init__(self, width: int, Lt: torch.Tensor, k_cheb: int = 3, groups: int = 8):
        super().__init__()
        self.k = k_cheb
        self.register_buffer("Lt", Lt.contiguous(), persistent=False)           # [D, D], symmetric
        self.norm = nn.GroupNorm(groups, width)
        self.mix = nn.Conv2d(width * k_cheb, width, 1)
        self.out = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        T = [h, torch.matmul(h, self.Lt)]
        for _ in range(2, self.k):
            T.append(2 * torch.matmul(T[-1], self.Lt) - T[-2])
        return self.out(F.gelu(self.mix(torch.cat(T[:self.k], 1))))


class SphereGaussBlur(nn.Module):
    """E = range Gaussian x spherical Gaussian (geodesic angle, width sigma, the SAME everywhere on the
    hemisphere) applied to E_raw, mass-preserving per source direction. Gives the clean map uniform,
    smooth blobs (user request 2026-10-06). sigma fixed (design choice) unless learn=True."""

    def __init__(self, sphere: Sphere, sigma_deg: float = 3.0, range_sigma_cells: float = 0.7, learn: bool = False):
        super().__init__()
        self.register_buffer("cosang", sphere.cos_angles(), persistent=False)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(math.radians(sigma_deg))), requires_grad=learn)
        i = torch.arange(-2, 3, dtype=torch.float32)
        k = torch.exp(-0.5 * (i / range_sigma_cells) ** 2)
        self.register_buffer("kr", (k / k.sum()).view(1, 1, 5, 1), persistent=False)

    def matrix(self) -> torch.Tensor:
        """[D_src, D_dst]: row v spreads source v over its neighbours (rows sum to 1)."""
        ang2 = torch.acos(self.cosang.clamp(-1 + 1e-7, 1 - 1e-7)) ** 2
        H = torch.exp(-0.5 * ang2 / torch.exp(2 * self.log_sigma))
        return H / H.sum(1, keepdim=True)

    def forward(self, E_raw: torch.Tensor) -> torch.Tensor:
        E = torch.matmul(E_raw, self.matrix())
        return F.conv2d(E[:, None], self.kr, padding=(2, 0))[:, 0]


class Deconvolver(nn.Module):
    def __init__(self, D: int, n_r: int, width: int = 48, n_blocks: int = 6, groups: int = 8,
                 Lt: torch.Tensor | None = None, k_cheb: int = 3):
        super().__init__()
        self.D, self.n_r = D, n_r
        self.cheb = nn.ModuleList([ChebSphere(width, Lt, k_cheb, groups) for _ in range(n_blocks)]) if Lt is not None else None
        self.inp = nn.Conv2d(2, width, 1)
        self.dir_norm = nn.ModuleList([nn.GroupNorm(groups, width) for _ in range(n_blocks)])
        # direction mixing starts as the identity (+ small noise): the network starts local, from the raw image
        self.dir_mix = nn.ParameterList([nn.Parameter(torch.eye(D) + 0.1 * torch.randn(D, D) / math.sqrt(D))
                                         for _ in range(n_blocks)])
        self.dir_out = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_blocks)])
        self.rng = nn.ModuleList()
        for i in range(n_blocks):
            d = 2 ** (i % 3)
            self.rng.append(nn.Sequential(nn.GroupNorm(groups, width), nn.Conv2d(width, width, (5, 1), padding=(2 * d, 0), dilation=(d, 1)),
                                          nn.GELU(), nn.Conv2d(width, width, 1)))
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.head.weight, std=0.01); nn.init.zeros_(self.head.bias)
        self.bias = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer("rpos", torch.linspace(-1, 1, n_r)[None, None, :, None], persistent=False)

    def forward(self, level: torch.Tensor) -> torch.Tensor:
        """level [B, R, D] -> E_raw [B, R, D] >= 0."""
        B = level.shape[0]
        x = torch.cat([level[:, None], self.rpos.expand(B, 1, self.n_r, self.D)], 1)
        x = self.inp(x)
        for i, (nrm, W, out, rng) in enumerate(zip(self.dir_norm, self.dir_mix, self.dir_out, self.rng)):
            x = x + out(F.gelu(torch.matmul(nrm(x), W)))                     # global direction mixing
            if self.cheb is not None:
                x = x + self.cheb[i](x)                                     # local spherical convolution
            x = x + rng(x)                                                  # range convolution
        return F.softplus(self.head(x)[:, 0] + self.bias)


# --------------------------------------------------------------------------------------- forward model
class SensorBeam(nn.Module):
    """G(v, f) in dB: an MLP on random Fourier features of the unit vector, n_f frequency outputs."""

    def __init__(self, n_f: int, n_feat: int = 64, hidden: int = 128, scale: float = 3.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("Bf", scale * torch.randn(3, n_feat, generator=g))
        self.mlp = nn.Sequential(nn.Linear(2 * n_feat + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_f))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        z = u @ self.Bf
        return 10 * self.mlp(torch.cat([torch.sin(z), torch.cos(z), u], -1))     # [D, n_f] dB


class ForwardModel(nn.Module):
    def __init__(self, sphere: Sphere, grid: RangeGrid, n_f: int = 64, n_dr: int = 3, h_fine: float = 1.25e-3,
                 learn_array: bool = True, init_positions: str = "nominal"):
        super().__init__()
        self.S, self.grid, self.n_dr = sphere, grid, n_dr
        arr = load_array()
        nom = torch.as_tensor(np.asarray(arr["nominal"])[:, 1:], dtype=torch.float32)     # [M, 2] (y, z)
        self.register_buffer("p_nom", nom)
        if init_positions == "nominal":
            p0 = nom.clone()
        else:                                                                             # "jitter<mm>"
            mm = float(init_positions.replace("jitter", "") or 1.0)
            p0 = nom + mm * 1e-3 * torch.randn(nom.shape, generator=torch.Generator().manual_seed(5))
        self.p = nn.Parameter(p0, requires_grad=learn_array)
        # emitter position: nominal (known design) + learned correction. The two-way path to a point at
        # (r, v) is ~ 2 r - e . v, so the DAS image puts it at range r - e . v / 2 (up to a cell here).
        self.register_buffer("e_nom", torch.tensor(EMITTER_POS, dtype=torch.float32))
        self.de = nn.Parameter(torch.zeros(3), requires_grad=learn_array)
        M = nom.shape[0]
        self.g_re = nn.Parameter(torch.ones(M), requires_grad=learn_array)
        self.g_im = nn.Parameter(torch.zeros(M), requires_grad=learn_array)
        f = torch.linspace(F_LO, F_HI, n_f)
        self.register_buffer("f", f)
        C = np.fft.rfft(chirp(), 8192)
        fc = np.fft.rfftfreq(8192, 1 / 450e3)
        s0 = np.interp(f.numpy(), fc, np.abs(C) ** 2)
        self.register_buffer("S0_db", torch.as_tensor(10 * np.log10(s0 / s0.max()), dtype=torch.float32))
        self.dS = nn.Parameter(torch.zeros(n_f))                                        # learned spectrum (dB)
        self.beam = SensorBeam(n_f)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1.5)))                        # dB/m one way
        # fine range grid around the reflector (+-(n_dr + 1) cells) and the cell triangle weights
        nx = int(round((n_dr + 1) * grid.dr / h_fine))
        x = torch.arange(-nx, nx + 1, dtype=torch.float32) * h_fine
        self.register_buffer("ph_x", 4 * math.pi * f[:, None] * x[None] / C_SOUND)       # [F, T]
        dr = torch.arange(-n_dr, n_dr + 1, dtype=torch.float32) * grid.dr
        tri = (1 - (x[None] - dr[:, None]).abs() / grid.dr).clamp_min(0)                 # [2n+1, T]
        self.register_buffer("tri", tri / tri.sum(1, keepdim=True), persistent=False)
        self.register_buffer("r", grid.r("cpu"), persistent=False)
        self.register_buffer("gain_db", torch.tensor(0.0))
        self.register_buffer("norm0", torch.tensor(1.0))

    def beam_db(self) -> torch.Tensor:
        """Directional part of the sensor beam [D, F] dB: zero mean over the hemisphere per frequency."""
        g = self.beam(self.S.dirs)
        return g - g.mean(0, keepdim=True)

    def beam_prior(self, over_db: float = 3.0):
        """Priors on the sensor beam (no ground truth): (1) spherical total variation over the mesh edges
        (Charbonnier, dB), per frequency; (2) nowhere louder than straight ahead + over_db (a squared
        hinge). Both gauge-free. Without them the beam blew up (+33 dB) where no reflectors are ever seen."""
        g = self.beam_db()
        e = self.S.edges
        tv = (torch.sqrt((g[e[:, 0]] - g[e[:, 1]]) ** 2 + 1.0) - 1.0).mean()
        b = int(self.S.dirs[:, 0].argmax())                                               # boresight
        over = F.relu(g - g[b:b + 1] - over_db).pow(2).mean()
        return tv, over

    def psf(self) -> torch.Tensor:
        """K [2n+1, D (u: steering), D (v: source)], normalised (norm0): mean peak of K[:, v, v] = 1 at init."""
        u = self.S.dirs                                                                     # [D, 3]
        k = 2 * math.pi * self.f / C_SOUND                                                  # [F]
        pv = u[:, 1:] @ self.p.T                                                            # [D, M] p_m . v
        pu = u[:, 1:] @ self.p_nom.T                                                        # [D, M] p_nom_m . u
        g = torch.complex(self.g_re, self.g_im)
        g = g / torch.sqrt((g.real ** 2 + g.imag ** 2).mean())                             # gauge: mean mic power 1
        A = torch.polar(torch.ones_like(pv)[None].expand(len(k), -1, -1), k[:, None, None] * pv[None]) * g   # [F, Dv, M]
        Wn = torch.polar(torch.ones_like(pu)[None].expand(len(k), -1, -1), -k[:, None, None] * pu[None])     # [F, Du, M]
        Bf = torch.bmm(Wn, A.transpose(1, 2)) / pu.shape[1]                                 # [F, Du, Dv]
        amp = torch.pow(10.0, (self.S0_db + self.dS - self.dS.mean())[None] / 20 + self.beam_db() / 20)   # [Dv, F]
        Bf = Bf * amp.T[:, None, :].to(Bf.dtype)
        ev = u @ (self.e_nom + self.de)                                                     # [Dv]
        Bf = Bf * torch.polar(torch.ones_like(ev)[None, None], (k[:, None] * ev[None])[:, None])   # range shift -e.v/2
        E = torch.polar(torch.ones_like(self.ph_x), self.ph_x)                              # [F, T]
        y = torch.einsum("fuv,ft->uvt", Bf, E)                                              # [Du, Dv, T]
        p = y.real ** 2 + y.imag ** 2
        K = torch.einsum("uvt,nt->nuv", p, self.tri)
        return K / self.norm0

    @torch.no_grad()
    def init_norm(self):
        """Fix the scale once at init: the mean (over v) peak of K[:, v, v] is 1."""
        self.norm0.fill_(1.0)
        K = self.psf()
        self.norm0.fill_(float(torch.diagonal(K, dim1=1, dim2=2).amax(0).mean()))

    def forward(self, E: torch.Tensor, K: torch.Tensor | None = None) -> torch.Tensor:
        """E [B, R, D] reflectivity -> P_hat [B, R, D] linear power (data units)."""
        if K is None:
            K = self.psf()
        alpha = self.log_alpha.exp()
        att = torch.pow(10.0, (self.gain_db - 40 * torch.log10(self.r) - 2 * alpha * self.r) / 10)   # [R]
        Ea = E * att[None, :, None]
        n = self.n_dr
        Ep = F.pad(Ea, (0, 0, n, n))
        out = 0
        for j in range(2 * n + 1):                                                          # dr = j - n
            out = out + torch.matmul(Ep[:, 2 * n - j: 2 * n - j + E.shape[1]], K[j].T)
        return out

    def att(self) -> torch.Tensor:
        """[R] range attenuation: fixed gauge gain, r^-4, learned air absorption."""
        return torch.pow(10.0, (self.gain_db - 40 * torch.log10(self.r) - 2 * self.log_alpha.exp() * self.r) / 10)

    def adjoint(self, X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """A^T X: X [B, R, D] (image space) -> [B, R, D] (reflectivity space)."""
        n = self.n_dr
        Xp = F.pad(X, (0, 0, n, n))
        out = 0
        for j in range(2 * n + 1):                                                          # dr = j - n
            out = out + torch.matmul(Xp[:, j: j + X.shape[1]], K[j])
        return out * self.att()[None, :, None]


# ------------------------------------------------------------------------------------ unrolled solver
class Unrolled(nn.Module):
    """Deconvolver = T unrolled multiplicative updates for the speckle likelihood (Itakura-Saito NMF /
    Richardson-Lucy type) through the learned forward model A:

        q = A E + N0,     E <- E * [ A^T (P / q^2) / (A^T (1 / q) + lambda_t) ]^gamma_t

    A = the forward model (PSF K and range attenuation; K is a constant in the ONNX graph). Learned here:
    step exponents gamma_t, sparsity weights lambda_t and the start value. A itself is learned
    self-supervised (the same operator renders the predictions)."""

    def __init__(self, T: int = 20):
        super().__init__()
        self.T = T
        self.log_gamma = nn.Parameter(torch.zeros(T))
        self.log_lam = nn.Parameter(torch.full((T,), -10.0))
        self.log_e0 = nn.Parameter(torch.tensor(math.log(1e-3)))

    def forward(self, P: torch.Tensor, fwd: "ForwardModel", K: torch.Tensor, N0: torch.Tensor) -> torch.Tensor:
        E = torch.exp(self.log_e0) * torch.ones_like(P)
        for t in range(self.T):
            q = fwd(E, K) + N0
            num = fwd.adjoint(P / q ** 2, K)
            den = fwd.adjoint(1 / q, K) + torch.exp(self.log_lam[t])
            E = E * torch.pow((num / den).clamp(1e-6, 1e6), torch.exp(self.log_gamma[t]))
        return E


# ------------------------------------------------------------------------------------------ levels
class LevelMap(nn.Module):
    """dB -> compressed level: l = 10 log10(1 + 10^((P_dB - floor) / 10)) / span. Constants from DATA."""

    def __init__(self, floor_db: float, span_db: float):
        super().__init__()
        self.register_buffer("floor_db", torch.tensor(float(floor_db)))
        self.register_buffer("span", torch.tensor(float(span_db)))

    def from_db(self, db: torch.Tensor) -> torch.Tensor:
        return F.softplus((db - self.floor_db) * (math.log(10) / 10)) * (10 / math.log(10)) / self.span

    def from_power(self, P: torch.Tensor) -> torch.Tensor:
        return 10 * torch.log10(1 + P / torch.pow(10.0, self.floor_db / 10)) / self.span


class Model(nn.Module):
    def __init__(self, sphere: Sphere, grid: RangeGrid, floor_db: float, span_db: float, width: int = 48,
                 n_blocks: int = 6, learn_array: bool = True, init_positions: str = "nominal", arch: str = "mixer",
                 T: int = 20, cheb: int = 0, blur_deg: float = 0.0, n_dr: int = 3):
        """cheb: Chebyshev order of the local spherical convolutions in each mixer block (0 = none);
        blur_deg: width of the uniform spherical Gaussian applied to E_raw (0 = none): the clean map
        E = blur(E_raw) is what is rendered, warped and output."""
        super().__init__()
        self.S, self.grid, self.arch = sphere, grid, arch
        self.lv = LevelMap(floor_db, span_db)
        Lt = sphere.cheb_laplacian() if cheb else None
        self.net = Deconvolver(sphere.D, grid.n_r, width, n_blocks, Lt=Lt, k_cheb=cheb) if arch == "mixer" else Unrolled(T)
        self.fwd = ForwardModel(sphere, grid, n_dr=n_dr, learn_array=learn_array, init_positions=init_positions)
        self.blur = SphereGaussBlur(sphere, blur_deg) if blur_deg > 0 else None

    def clean(self, E_raw: torch.Tensor) -> torch.Tensor:
        return self.blur(E_raw) if self.blur is not None else E_raw

    def encode(self, es_db: torch.Tensor, K: torch.Tensor | None = None) -> torch.Tensor:
        """-> the clean map E (blurred if blur_deg > 0)."""
        if self.arch == "mixer":
            return self.clean(self.net(self.lv.from_db(es_db)))
        if K is None:
            K = self.fwd.psf()
        return self.net(torch.pow(10.0, es_db / 10), self.fwd, K, torch.pow(10.0, self.lv.floor_db / 10))

    def render(self, E: torch.Tensor, K=None) -> torch.Tensor:
        return self.lv.from_power(self.fwd(E, K))

    @torch.no_grad()
    def infer_db(self, es_db: torch.Tensor, with_render: bool = False):
        """Deconvolved energyscape in dB, in the input's units: the reflectivity map E seen through the
        same range/beam attenuation but with a delta PSF (each reflector in its own cell and direction)."""
        E = self.encode(es_db)
        f = self.fwd
        att = torch.pow(10.0, (f.gain_db - 40 * torch.log10(f.r) - 2 * f.log_alpha.exp() * f.r) / 10)
        out = 10 * torch.log10(E * att[None, :, None] + 1e-30)
        if with_render:
            return out, 10 * torch.log10(f(E) + 1e-30)
        return out


@torch.no_grad()
def calibrate_gain_db(es_db: torch.Tensor, grid: RangeGrid, floor_db: float) -> float:
    """Fixed gauge from DATA statistics: the 99.9th percentile of (P - floor) r^4 over 0.5-2 m is what a
    reflector with E = 1 produces (the PSF peaks are normalised to ~1)."""
    r = grid.r(es_db.device)
    sel = (r >= 0.5) & (r <= 2.0)
    x = es_db[::7][:, sel].float()
    P = (torch.pow(10.0, x / 10) - 10 ** (floor_db / 10)).clamp_min(0) * r[sel][None, :, None] ** 4
    q = torch.quantile(P.flatten()[::17], 0.999).item()
    return 10 * math.log10(q)
