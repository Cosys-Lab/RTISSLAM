"""Directions of interest: D points on the frontal hemisphere (x > 0 in the sensor frame), equal area.

Spiral (Fibonacci) sampling around the forward axis: x_j = 1 - (j + 1/2) / D is uniform in x, i.e. equal
solid angle 2 pi / D per point; the roll angle about x advances by the golden angle. Works for any D
(400, 1000, ...).

Chart: the Lambert azimuthal equal-area projection (LAEAP) centred on straight ahead, as in BatSLAM2's
figures, seen from BEHIND the sensor (the sensor's left on the left):
    rho = 2 sin(theta / 2) (theta = angle from +x),  X = -y rho / s,  Y = z rho / s   (s = |(y, z)|)
The hemisphere maps to the disk of radius sqrt(2). A Delaunay triangulation of the points in this chart
gives the mesh for barycentric interpolation (warp) and a lookup table from a fine chart grid to the
containing triangle.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import Delaunay

GOLDEN = math.pi * (3 - math.sqrt(5))
RHO_MAX = math.sqrt(2)


def spiral_hemisphere(D: int) -> np.ndarray:
    j = np.arange(D)
    x = 1 - (j + 0.5) / D
    s = np.sqrt(1 - x ** 2)
    a = j * GOLDEN
    return np.c_[x, s * np.cos(a), s * np.sin(a)]


def chart(u):
    """[..., 3] -> X, Y (LAEAP, seen from behind). numpy or torch."""
    lib = torch if isinstance(u, torch.Tensor) else np
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    s = lib.sqrt(y * y + z * z)
    rho = lib.sqrt(lib.clip(2 * (1 - x), 0, None) if lib is np else (2 * (1 - x)).clamp_min(0))
    k = rho / (s + 1e-12)
    return -y * k, z * k


def chart_inv(X, Y):
    rho2 = X ** 2 + Y ** 2
    x = 1 - rho2 / 2
    k = np.sqrt(np.clip(1 - rho2 / 4, 0, None))
    return np.stack([x, -X * k, Y * k], -1)


def azel(u):
    lib = torch if isinstance(u, torch.Tensor) else np
    az = lib.arctan2(u[..., 1], u[..., 0])
    el = lib.arctan2(u[..., 2], lib.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2))
    return az, el


class Sphere:
    def __init__(self, D: int = 1000, device="cuda", lut: int = 512):
        self.D, self.device = D, device
        d = spiral_hemisphere(D)
        self.dirs_np = d
        self.dirs = torch.as_tensor(d, dtype=torch.float32, device=device)
        self.w = torch.full((D,), 2 * math.pi / D, device=device)              # sr per point
        X, Y = chart(d)
        self.XY = np.c_[X, Y]
        tri = Delaunay(self.XY)
        self.tri = tri
        self.faces = torch.as_tensor(tri.simplices, dtype=torch.long, device=device)   # [F, 3]
        # affine maps chart -> barycentric per simplex
        self.T = torch.as_tensor(tri.transform, dtype=torch.float32, device=device)    # [F, 3, 2]
        # LUT: fine chart grid -> simplex (-1 outside)
        self.lut_n = lut
        g = (np.arange(lut) + 0.5) / lut * 2 * RHO_MAX - RHO_MAX
        GX, GY = np.meshgrid(g, g, indexing="ij")
        s = tri.find_simplex(np.c_[GX.ravel(), GY.ravel()])
        # cells whose centre is outside the hull but near it: nearest simplex of a neighbour sample
        self.lut = torch.as_tensor(s.reshape(lut, lut), dtype=torch.long, device=device)
        # neighbours (mesh edges) for plots / regularisers
        e = np.concatenate([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [2, 0]]])
        e = np.unique(np.sort(e, 1), axis=0)
        self.edges = torch.as_tensor(e, dtype=torch.long, device=device)
        self.spacing_deg = math.degrees(math.sqrt(2 * math.pi / D))

    def cos_angles(self) -> torch.Tensor:
        """[D, D] cosine of the great-circle angle between directions."""
        return (self.dirs @ self.dirs.T).clamp(-1, 1)

    def cheb_laplacian(self) -> torch.Tensor:
        """[D, D] rescaled normalised graph Laplacian of the mesh, L~ = L_sym - I (eigenvalues in [-1, 1]),
        for Chebyshev filters. Edge weights: Gaussian in the edge's angle (scale = mean edge angle)."""
        e = self.edges
        ang = torch.acos((self.dirs[e[:, 0]] * self.dirs[e[:, 1]]).sum(-1).clamp(-1, 1))
        w = torch.exp(-(ang / ang.mean()) ** 2 / 2)
        W = torch.zeros(self.D, self.D, device=self.device)
        W[e[:, 0], e[:, 1]] = w; W[e[:, 1], e[:, 0]] = w
        d = W.sum(1).clamp_min(1e-9).rsqrt()
        return -(d[:, None] * W * d[None])                        # L_sym - I = -D^-1/2 W D^-1/2

    def interp_weights(self, u: torch.Tensor):
        """u [..., 3] (unit) -> vertex ids [..., 3], barycentric weights [..., 3] (zero outside the mesh
        or behind the sensor)."""
        X, Y = chart(u)
        n = self.lut_n
        i = ((X + RHO_MAX) / (2 * RHO_MAX) * n).long().clamp(0, n - 1)
        j = ((Y + RHO_MAX) / (2 * RHO_MAX) * n).long().clamp(0, n - 1)
        s = self.lut[i, j]
        ok = (s >= 0) & (u[..., 0] > 0)
        s = s.clamp_min(0)
        T = self.T[s]                                                            # [..., 3, 2]
        p = torch.stack([X, Y], -1) - T[..., 2, :]
        b01 = (T[..., :2, :] @ p[..., None])[..., 0]                            # [..., 2]
        b = torch.cat([b01, 1 - b01.sum(-1, keepdim=True)], -1)
        # a LUT cell may straddle a triangle edge: clamp tiny negatives and renormalise
        b = b.clamp_min(0)
        b = b / b.sum(-1, keepdim=True).clamp_min(1e-9)
        return self.faces[s], b * ok[..., None]

    def interp(self, values: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """values [D, ...] -> at u [N, 3]: [N, ...]."""
        vid, b = self.interp_weights(u)
        return (values[vid] * b.view(*b.shape, *([1] * (values.dim() - 1)))).sum(1)


class Circle:
    """2D direction set (user request 2026-10-06: "focus on 2D"): D azimuths, uniform in
    [-AZ_MAX, AZ_MAX], in the horizontal plane of the sensor (el = 0). Same interface as Sphere, so the
    DAS, the warp, the forward model and the networks run unchanged; interpolation is linear in azimuth
    (the elevation of a query is ignored: in 2D everything stays in the plane)."""

    def __init__(self, D: int = 181, az_max_deg: float = 90.0, device="cuda"):
        self.D, self.device = D, device
        self.az = np.radians(np.linspace(-az_max_deg, az_max_deg, D))
        self.daz = self.az[1] - self.az[0]
        d = np.c_[np.cos(self.az), np.sin(self.az), np.zeros(D)]
        self.dirs_np = d
        self.dirs = torch.as_tensor(d, dtype=torch.float32, device=device)
        self.w = torch.full((D,), float(self.daz), device=device)           # rad per direction
        e = np.c_[np.arange(D - 1), np.arange(1, D)]
        self.edges = torch.as_tensor(e, dtype=torch.long, device=device)
        self.spacing_deg = math.degrees(self.daz)
        self.dim = 2

    def cos_angles(self) -> torch.Tensor:
        return (self.dirs @ self.dirs.T).clamp(-1, 1)

    def cheb_laplacian(self) -> torch.Tensor:
        W = torch.zeros(self.D, self.D, device=self.device)
        e = self.edges
        W[e[:, 0], e[:, 1]] = 1.0; W[e[:, 1], e[:, 0]] = 1.0
        d = W.sum(1).clamp_min(1e-9).rsqrt()
        return -(d[:, None] * W * d[None])

    def interp_weights(self, u: torch.Tensor):
        """u [..., 3] -> ids [..., 3], weights [..., 3] (linear in azimuth; third slot unused)."""
        az = torch.atan2(u[..., 1], u[..., 0])
        x = (az - float(self.az[0])) / float(self.daz)
        ok = (x >= 0) & (x <= self.D - 1) & (u[..., 0] > 0)
        i0 = x.floor().long().clamp(0, self.D - 2)
        a = (x - i0).clamp(0, 1)
        vid = torch.stack([i0, i0 + 1, i0], -1)
        b = torch.stack([1 - a, a, torch.zeros_like(a)], -1) * ok[..., None]
        return vid, b

    def interp(self, values: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        vid, b = self.interp_weights(u)
        return (values[vid] * b.view(*b.shape, *([1] * (values.dim() - 1)))).sum(1)


def directions(D: int, dim: int = 3, device="cuda"):
    """The direction set of a dataset: Sphere (3D hemisphere) or Circle (2D, azimuth only)."""
    return Circle(D, device=device) if dim == 2 else Sphere(D, device)
