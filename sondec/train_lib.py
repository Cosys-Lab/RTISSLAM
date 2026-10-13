"""Data, checkpoints and evaluation for the deconvolution model."""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import torch

from .das import RangeGrid
from .model import Model
from .paths import data_dir
from .sensor import load_array
from .sphere import Sphere, directions
from .warp import head_motion_from_sensors, warp_apply, warp_coords


class Sequences:
    """All pulses of a split on the GPU (fp16), windows (t, t+1..t+K) inside one world."""

    def __init__(self, ds: str, split: str, K: int, device="cuda", max_worlds: int | None = None):
        files = sorted(glob.glob(str(data_dir() / "ds" / ds / split / "*.npz")))
        files = [f for f in files if not f.endswith(".tmp.npz")][:max_worlds]
        es, inc, servo, starts, self.names, self.gt = [], [], [], [], [], []
        n0 = 0
        for f in files:
            z = np.load(f)
            n = len(z["es_db"])
            es.append(torch.as_tensor(z["es_db"]))
            inc.append(np.c_[z["odo_xy"], z["gyro_yaw"]]); servo.append(z["servo"][:, :2])
            starts += list(range(n0, n0 + n - K))
            self.names.append(Path(f).stem)
            self.gt.append({k: z[k] for k in ("top_r", "top_u", "top_pow", "gt_pose", "head")})
            n0 += n
            meta = z
        self.es = torch.cat(es).to(device)                                          # [N, R, D] fp16 dB
        self.inc = torch.as_tensor(np.concatenate(inc), dtype=torch.float32, device=device)
        self.servo = torch.as_tensor(np.concatenate(servo), dtype=torch.float32, device=device)
        self.starts = torch.as_tensor(starts, device=device)
        self.K, self.dev = K, device
        self.grid = RangeGrid(float(meta["r_min"]), float(meta["dr"]), int(meta["n_r"]))
        self.D = self.es.shape[2]
        self.dim = int(meta["dim"]) if "dim" in meta.files else 3
        self.top_r = np.concatenate([g["top_r"] for g in self.gt]); self.top_u = np.concatenate([g["top_u"] for g in self.gt])
        self.top_pow = np.concatenate([g["top_pow"] for g in self.gt])

    def batch(self, idx: torch.Tensor):
        """idx [B] window starts -> es [B, K+1, R, D] (float dB), inc [B, K, 3], servo [B, K+1, 2]."""
        k = torch.arange(self.K + 1, device=self.dev)
        t = idx[:, None] + k
        return self.es[t].float(), self.inc[t[:, :-1]], self.servo[t]

    def sample(self, B: int, gen: torch.Generator):
        return self.starts[torch.randint(len(self.starts), (B,), device=self.dev, generator=gen)]


def data_stats(es_db: torch.Tensor):
    """Noise floor and span (dB) from the data: floor = 20th percentile of all cells (most cells hold
    only noise and sidelobes), span = 99.95th percentile of the level above the floor."""
    x = es_db[::11, ::3].float().flatten()[::7]
    floor = torch.quantile(x, 0.2).item()
    lv = 10 * torch.log10(1 + torch.pow(10.0, (x - floor) / 10))
    return floor, torch.quantile(lv, 0.9995).item()


def build_model(meta: dict, device="cuda") -> Model:
    S = directions(meta["D"], meta.get("dim", 3), device)
    grid = RangeGrid(*meta["grid"])
    m = Model(S, grid, meta["floor_db"], meta["span_db"], meta.get("width", 48), meta.get("n_blocks", 6),
              learn_array=meta.get("learn_array", True), init_positions=meta.get("init_positions", "nominal"),
              arch=meta.get("arch", "mixer"), T=meta.get("T", 20), cheb=meta.get("cheb", 0),
              blur_deg=meta.get("blur_deg", 0.0), n_dr=meta.get("n_dr", 3))
    m = m.to(device)
    m.fwd.gain_db.fill_(meta["gain_db"])
    m.fwd.init_norm()
    return m


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(ck["meta"], device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["meta"]


def nll_power(mu: torch.Tensor, P: torch.Tensor, N0: torch.Tensor) -> torch.Tensor:
    """Speckle likelihood per cell: the observed DAS power P is exponentially distributed around the
    incoherent mean mu + N0 (noise floor). -log p = log q + P / q, offset so a perfect fit is 0 (the
    Itakura-Saito divergence)."""
    q = mu + N0
    return torch.log(q / P) + P / q - 1


def robust_nll_power(mu: torch.Tensor, P: torch.Tensor, N0: torch.Tensor, nu: float = 1.0) -> torch.Tensor:
    """Heavy-tailed speckle likelihood: exponential speckle whose mean is itself uncertain (gamma-
    distributed precision with nu degrees of freedom) -> -log p = log q + (1 + nu) log(1 + P / (nu q)).
    Minimum at q = P (offset to 0 there); grows only logarithmically for unexplained echoes (e.g. the
    sidelobes of a reflector entering the view, which no warp mask can hide). nu -> inf: nll_power."""
    q = mu + N0
    return torch.log(q / P) + (1 + nu) * (torch.log1p(P / (nu * q)) - math.log1p(1 / nu))


def step_losses(model: Model, es, inc, servo, l1: float, K_psf=None, loss: str = "nll", sparsity: str = "l1",
                w_beam_tv: float = 0.0, w_beam_over: float = 0.0, w_pred: float = 1.0):
    """-> dict of losses for one batch of windows. loss: "nll" (speckle likelihood on power) or "mse"
    (squared error of compressed levels)."""
    lv = model.lv.from_db(es)                                                         # [B, K+1, R, D]
    P = torch.pow(10.0, es / 10)
    N0 = torch.pow(10.0, model.lv.floor_db / 10)

    def err(E, k):
        if loss == "nll":
            return nll_power(model.fwd(E, Kp), P[:, k], N0)
        if loss == "robust":
            return robust_nll_power(model.fwd(E, Kp), P[:, k], N0)
        return (model.render(E, Kp) - lv[:, k]) ** 2
    Kp = model.fwd.psf() if K_psf is None else K_psf
    if model.arch == "mixer":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            E_raw = model.net(lv[:, 0])
        E_raw = E_raw.float()
        E = model.clean(E_raw)
    else:
        E = E_raw = model.encode(es[:, 0], Kp)
    rec = err(E, 0).mean()
    R, t = head_motion_from_sensors(inc, servo)                                       # [B, K, 3, 3], [B, K, 3]
    preds = []
    ones = torch.ones_like(E[:, None])
    for k in range(inc.shape[1]):
        idx, w = warp_coords(R[:, k], t[:, k], model.S, model.grid)
        Ew = warp_apply(E[:, None], idx, w)[:, 0]
        with torch.no_grad():
            mask = (warp_apply(ones, idx, w)[:, 0] > 0.5).float()
        e = err(Ew, k + 1)
        preds.append((e * mask).sum() / mask.sum().clamp_min(1))
    pred = torch.stack(preds)
    # sparsity on E_raw. "l1" costs the same for spread and concentrated energy; "sqrt" (an L_1/2-type
    # penalty) favours concentration, i.e. removes the PSF's natural widening towards the rim.
    sp = torch.sqrt(E_raw + 1e-4).mean() if sparsity == "sqrt" else E_raw.mean()
    total = rec + w_pred * pred.mean() + l1 * sp
    out = {"rec": rec, "pred": pred.mean(), "pred_k": pred.detach(), "l1": sp}
    if w_beam_tv or w_beam_over:
        tv, over = model.fwd.beam_prior()
        total = total + w_beam_tv * tv + w_beam_over * over
        out.update(beam_tv=tv.detach(), beam_over=over.detach())
    out["loss"] = total
    return out


def truth_target(seq: "Sequences", idx: torch.Tensor, sphere, grid) -> torch.Tensor:
    """SUPERVISED UPPER BOUND ONLY: the true reflectivity map [B, R, D] of pulses idx: the (up to 512)
    strongest true echoes, range-compensated power p r^4, splatted into their range cell and (linearly) onto
    the direction grid. 2D: echoes within |u_z| < 0.2 of the sensor plane, projected onto it."""
    out = torch.zeros(len(idx), grid.n_r, sphere.D, device=sphere.device)
    for b, i in enumerate(idx.tolist()):
        r = torch.as_tensor(seq.top_r[i], device=sphere.device); u = torch.as_tensor(seq.top_u[i], device=sphere.device)
        p = torch.as_tensor(seq.top_pow[i], device=sphere.device)
        ok = (r > grid.r_min) & (r < grid.r_min + grid.n_r * grid.dr) & (p > 0)
        if getattr(sphere, "dim", 3) == 2:
            ok &= u[:, 2].abs() < 0.2
            u = u * torch.tensor([1.0, 1.0, 0.0], device=u.device)
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        r, u, p = r[ok], u[ok], p[ok]
        ri = ((r - grid.r_min) / grid.dr).long()
        vid, w = sphere.interp_weights(u)
        for k in range(3):
            out[b].index_put_((ri, vid[:, k]), p * r ** 4 * w[:, k], accumulate=True)
    return out


# ----------------------------------------------------------------------------------- evaluation (GT)
def array_errors(model: Model) -> dict:
    """EVALUATION ONLY: learned vs true microphone positions and gains (the gauge: mean position offset,
    a common phase and the mean power are not identifiable from the images and are removed)."""
    a = load_array()
    p = model.fwd.p.detach().cpu().numpy()
    true = np.asarray(a["true"])[:, 1:]; nom = np.asarray(a["nominal"])[:, 1:]
    dl = p - true; dn = nom - true
    dl -= dl.mean(0); dn -= dn.mean(0)
    g = torch.complex(model.fwd.g_re, model.fwd.g_im).detach().cpu().numpy()
    gt = np.asarray(a["gain"]) * np.exp(1j * np.asarray(a["phase"]))
    def ph_err(x):
        d = np.angle(x / gt); d -= np.angle(np.mean(np.exp(1j * d)))
        return float(np.degrees(np.sqrt(np.mean(np.angle(np.exp(1j * d)) ** 2))))
    def g_err(x):
        d = 20 * np.log10(np.abs(x) / np.abs(gt)); return float(np.sqrt(np.mean((d - d.mean()) ** 2)))
    return {"pos_rms_mm": float(np.sqrt((dl ** 2).sum(1).mean()) * 1e3), "pos_rms_mm_nominal": float(np.sqrt((dn ** 2).sum(1).mean()) * 1e3),
            "gain_rms_db": g_err(g), "gain_rms_db_nominal": g_err(np.ones_like(gt)),
            "phase_rms_deg": ph_err(g), "phase_rms_deg_nominal": ph_err(np.ones_like(gt))}


def truth_mask(top_r, top_u, sphere: Sphere, grid: RangeGrid, n_top: int = 64, ang_cells: float = 1.5, r_cells: int = 1):
    """EVALUATION ONLY: [R, D] bool, cells within +-r_cells and ang_cells x spacing of the n_top strongest
    true echoes inside the image."""
    ok = (top_r > grid.r_min) & (top_r < grid.r_min + grid.n_r * grid.dr)
    r, u = top_r[ok][:n_top], top_u[ok][:n_top]
    m = torch.zeros(grid.n_r, sphere.D, dtype=torch.bool, device=sphere.device)
    if len(r) == 0:
        return m
    if getattr(sphere, "dim", 3) == 2:                 # 2D: azimuth only (project onto the sensor plane)
        u = u * np.array([1.0, 1.0, 0.0]); u = u / np.linalg.norm(u, axis=1, keepdims=True).clip(1e-9)
    # window: 1.5 direction spacings, at least 3 deg (2D spacing is 1 deg, far below the array resolution)
    cth = math.cos(math.radians(max(ang_cells * sphere.spacing_deg, 3.0)))
    near = (torch.as_tensor(u, dtype=torch.float32, device=sphere.device) @ sphere.dirs.T) > cth           # [n, D]
    ri = ((torch.as_tensor(r, device=sphere.device) - grid.r_min) / grid.dr).long()
    for d in range(-r_cells, r_cells + 1):
        rr = (ri + d).clamp(0, grid.n_r - 1)
        m[rr] |= near
    return m


@torch.no_grad()
def concentration(model: Model, seq: Sequences, n: int = 200, seed: int = 0) -> dict:
    """EVALUATION ONLY: fraction of the (range-compensated) image energy inside the neighbourhoods of the
    strongest true echoes, raw DAS vs deconvolved; and the same as a 'contrast' in dB (energy per cell
    inside / outside)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seq.es), size=min(n, len(seq.es)), replace=False)
    r = seq.grid.r(seq.dev)
    fr, fd, cr, cd = [], [], [], []
    for i in idx:
        es = seq.es[i].float()
        m = truth_mask(seq.top_r[i], seq.top_u[i], model.S, seq.grid)
        if m.sum() == 0:
            continue
        raw = torch.pow(10.0, es / 10) * r[:, None] ** 4
        raw = (raw - torch.pow(10.0, model.lv.floor_db / 10) * r[:, None] ** 4).clamp_min(0)
        dec = model.encode(es[None])[0]
        for x, fl, cl in ((raw, fr, cr), (dec, fd, cd)):
            fl.append((x[m].sum() / x.sum()).item())
            cl.append(10 * math.log10(x[m].mean().item() / max(x[~m].mean().item(), 1e-30)))
    return {"frac_raw": float(np.mean(fr)), "frac_deconv": float(np.mean(fd)), "contrast_raw_db": float(np.mean(cr)),
            "contrast_deconv_db": float(np.mean(cd)), "mask_frac": float(m.float().mean())}
