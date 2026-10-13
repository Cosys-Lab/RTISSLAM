"""2D views of a 3D energyscape E[r, u] (u on the scattered direction set):

- horizontal fan (top view): max over elevation |el| <= band of the barycentric interpolant, on a regular
  azimuth grid, drawn in x-y (forward up, the sensor's left on the left);
- vertical fan (side view): the same for elevation (|az| <= band), drawn in x-z;
- LAEAP disk: max over range of the (range-compensated) map, seen from behind the sensor.

Range compensation for display: + 40 log10(r) dB (two-way spherical spreading), so far echoes show.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .sphere import RHO_MAX, Sphere, chart_inv


def _azel_dirs(az_deg, el_deg):
    A, E = np.meshgrid(np.radians(az_deg), np.radians(el_deg), indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)


class Views:
    def __init__(self, sphere: Sphere, r: np.ndarray, band_deg: float = 12.0, n_ang: int = 121, n_disk: int = 161):
        self.S, self.r = sphere, r
        self.ang = np.linspace(-89, 89, n_ang)
        band = np.linspace(-band_deg, band_deg, 7)
        dev = sphere.device
        self.wh = sphere.interp_weights(torch.as_tensor(_azel_dirs(self.ang, band), dtype=torch.float32, device=dev))
        self.wv = sphere.interp_weights(torch.as_tensor(_azel_dirs(band, self.ang).transpose(1, 0, 2), dtype=torch.float32, device=dev))
        g = np.linspace(-RHO_MAX, RHO_MAX, n_disk)
        GX, GY = np.meshgrid(g, g, indexing="xy")
        self.disk_in = GX ** 2 + GY ** 2 < 2.0
        u = chart_inv(GX, GY)
        self.wd = sphere.interp_weights(torch.as_tensor(u, dtype=torch.float32, device=dev))
        self.g = g

    @staticmethod
    def _apply(E, w):
        vid, b = w
        return (E[..., vid] * b).sum(-1)                       # [R, *grid]

    def all(self, E_db: torch.Tensor, tvg: bool = True):
        """E_db [R, D] (dB) -> dict of numpy images in dB: horiz [R, n_ang], vert [R, n_ang], disk [n, n]."""
        E = E_db.float()
        if tvg:
            E = E + 40 * torch.log10(torch.as_tensor(self.r, dtype=torch.float32, device=E.device))[:, None]
        lin = torch.pow(10.0, (E - E.max()) / 10)                # interpolate in power
        h = self._apply(lin, self.wh).amax(-1)
        v = self._apply(lin, self.wv).amax(-1)
        d = self._apply(lin.amax(0), self.wd)
        f = lambda x: (10 * torch.log10(x + 1e-12) + E.max()).cpu().numpy()   # noqa: E731
        dd = f(d); dd[~self.disk_in] = np.nan
        return {"horiz": f(h), "vert": f(v), "disk": dd}


def fan_mesh(ang_deg: np.ndarray, r: np.ndarray, vertical: bool = False):
    """Cell edges for pcolormesh of a fan: horizontal -> (X = -y (left on the left), Y = x);
    vertical -> (X = x, Y = z)."""
    a = np.radians(np.r_[ang_deg[0] - (ang_deg[1] - ang_deg[0]) / 2, (ang_deg[1:] + ang_deg[:-1]) / 2,
                         ang_deg[-1] + (ang_deg[1] - ang_deg[0]) / 2])
    dr = r[1] - r[0]
    re = np.r_[r - dr / 2, r[-1] + dr / 2]
    A, Rr = np.meshgrid(a, re)
    if vertical:
        return Rr * np.cos(A), Rr * np.sin(A)
    return -Rr * np.sin(A), Rr * np.cos(A)


def truth_points(top_r, top_u, top_pow, band_deg: float = 12.0, n: int = 80):
    """The strongest ground-truth echoes (EVALUATION ONLY) for overlays: -> dict of (X, Y, level dB)
    for the horizontal fan, vertical fan and disk."""
    ok = top_r > 0
    r, u, p = top_r[ok][:n], top_u[ok][:n], top_pow[ok][:n]
    az = np.degrees(np.arctan2(u[:, 1], u[:, 0])); el = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
    from .sphere import chart
    X, Y = chart(u)
    lv = 10 * np.log10(p * r ** 4 + 1e-30)
    hb, vb = np.abs(el) <= band_deg, np.abs(az) <= band_deg
    return {"horiz": (-r[hb] * np.sin(np.radians(az[hb])), r[hb] * np.cos(np.radians(az[hb])), lv[hb]),
            "vert": (r[vb] * np.cos(np.radians(el[vb])), r[vb] * np.sin(np.radians(el[vb])), lv[vb]),
            "disk": (X, Y, lv)}


class FanRaster:
    """Anti-aliased raster of a 2D (range x azimuth) map as a top-view fan: each output pixel averages the
    bilinearly interpolated POWER of ss x ss sub-pixel samples, then converts to dB. Avoids the jagged,
    broken arcs of drawing 1-pixel-wide range cells as polygons (pcolormesh). Image coordinates: X = -y
    (the sensor's left on the left), Y = x (forward up), extent [-r_max, r_max] x [0, r_max]."""

    def __init__(self, r: np.ndarray, az_deg: np.ndarray, width: int = 720, ss: int = 3, device="cuda"):
        self.r_max = float(r[-1] + (r[1] - r[0]) / 2)
        self.W, self.H = width, width // 2
        self.extent = [-self.r_max, self.r_max, 0, self.r_max]
        px = 2 * self.r_max / self.W
        o = (np.arange(ss) + 0.5) / ss - 0.5
        xs = -self.r_max + (np.arange(self.W)[:, None] + 0.5 + o[None]) * px          # [W, ss]
        ys = (np.arange(self.H)[:, None] + 0.5 + o[None]) * px                      # [H, ss]
        X = xs[None, :, None, :]; Y = ys[:, None, :, None]                         # [H, W, ss, ss]
        rr = np.sqrt(X ** 2 + Y ** 2); aa = np.degrees(np.arctan2(-X, Y))
        fr = (rr - r[0]) / (r[1] - r[0]); fa = (aa - az_deg[0]) / (az_deg[1] - az_deg[0])
        ok = (fr >= 0) & (fr <= len(r) - 1) & (fa >= 0) & (fa <= len(az_deg) - 1)
        i0 = np.clip(np.floor(fr), 0, len(r) - 2).astype(np.int64); j0 = np.clip(np.floor(fa), 0, len(az_deg) - 2).astype(np.int64)
        wr, wa = np.clip(fr - i0, 0, 1), np.clip(fa - j0, 0, 1)
        nA = len(az_deg)
        idx = np.stack([i0 * nA + j0, i0 * nA + j0 + 1, (i0 + 1) * nA + j0, (i0 + 1) * nA + j0 + 1], -1)
        w = np.stack([(1 - wr) * (1 - wa), (1 - wr) * wa, wr * (1 - wa), wr * wa], -1) * ok[..., None] / ss ** 2
        T = lambda x, dt: torch.as_tensor(x.reshape(self.H * self.W, -1), dtype=dt, device=device)  # noqa: E731
        self.idx, self.w = T(idx, torch.long), T(w, torch.float32)
        self.inside = (np.sqrt(((np.arange(self.W) + 0.5) * px - self.r_max)[None] ** 2
                               + ((np.arange(self.H) + 0.5) * px)[:, None] ** 2) <= self.r_max)
        self.device = device

    def __call__(self, M_db: np.ndarray) -> np.ndarray:
        """M_db [R, D] (dB) -> [H, W] dB image (NaN outside the fan)."""
        x = torch.as_tensor(M_db, dtype=torch.float32, device=self.device).flatten()
        mx = x.max()
        p = torch.pow(10.0, (x - mx) / 10)
        img = (p[self.idx] * self.w).sum(1).view(self.H, self.W)
        out = (10 * torch.log10(img + 1e-12) + mx).cpu().numpy()
        out[~self.inside] = np.nan
        return out
