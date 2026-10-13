"""(Copied from HRTF SSL bat3d/warp.py, commit 8dc57cb; mirror-plane variant removed.)

Acoustic flow in 3D: the exact, parameter-free warp of a head-frame map Z(r, u) from the pose of
pulse t to the pose of pulse t+k.

For every target cell (r_i, u_j) (head frame at t+k), the same world point in the source head frame
(t) is p = r_i R u_j + t_rel, with (R, t_rel) the pose of head t+k in head t (body motion from wheel
odometry + gyro, composed with the pan/tilt change and the lever arm, kinematics.relative_head_pose).
The source map is sampled there: linearly in range, barycentrically on the icosphere. Points outside
the range grid or the field of view give zero (nothing enters from outside the map).

Small-step limit (tested): dr/dt = -v.u,  du/dt = -(v - (v.u) u) / r - w x u.
"""
from __future__ import annotations

import torch

from .kinematics import integrate_se2, relative_head_pose
from .das import RangeGrid
from .sphere import Sphere


def head_motion_from_sensors(inc: torch.Tensor, servo: torch.Tensor):
    """inc [B, K, 3] (odometry x, y + gyro yaw per step), servo [B, K+1, 2 or 3] (pan, tilt[, roll]) ->
    R [B, K, 3, 3], t [B, K, 3]: pose of head t+k in head t (k = 1..K)."""
    body = integrate_se2(inc.double())                                     # [B, K, 3]
    s = servo.double()
    K = inc.shape[1]
    roll0 = s[:, :1, 2].expand(-1, K) if s.shape[-1] > 2 else None
    roll1 = s[:, 1:, 2] if s.shape[-1] > 2 else None
    R, t = relative_head_pose(body, s[:, :1, 0].expand(-1, K), s[:, :1, 1].expand(-1, K), s[:, 1:, 0], s[:, 1:, 1],
                              roll0, roll1)
    return R.float(), t.float()


def source_coords(R: torch.Tensor, t: torch.Tensor, sphere: Sphere, grid: RangeGrid):
    """R [B, 3, 3], t [B, 3] -> (range index i0 [B, R, D], frac [B, R, D], vertex ids [B, R, D, 3],
    bary weights [B, R, D, 3] (zero outside the field of view / range))."""
    r = grid.r(R.device)
    ud = sphere.dirs @ R.transpose(-1, -2)                                  # [B, D, 3] = R u_j
    p = r[None, :, None, None] * ud[:, None] + t[:, None, None]            # [B, R, D, 3]
    rs = p.norm(dim=-1)
    us = p / rs[..., None].clamp_min(1e-6)
    fi = (rs - grid.r_min) / grid.dr - 0.5
    ok = (fi >= -1e-3) & (fi <= grid.n_r - 1 + 1e-3)                      # tolerance: exact cell centres
    i0 = fi.floor().clamp(0, grid.n_r - 2).long()
    a = (fi - i0).clamp(0, 1) * ok
    vid, b = sphere.interp_weights(us)
    b = b * ok[..., None]
    return i0, a, vid, b


@torch.no_grad()
def warp_coords(R: torch.Tensor, t: torch.Tensor, sphere: Sphere, grid: RangeGrid):
    """Sampling indices [B, R*D*6] (into the flattened [R*D] source map) and weights [B, R*D, 6].
    They depend only on the motion sensors (no gradient)."""
    B, D = R.shape[0], sphere.D
    i0, a, vid, b = source_coords(R, t, sphere, grid)
    idx = torch.stack([(i0 + dr) * D + vid[..., k] for dr in (0, 1) for k in range(3)], -1)
    w = torch.stack([(1 - a if dr == 0 else a) * b[..., k] for dr in (0, 1) for k in range(3)], -1)
    return idx.view(B, -1), w.view(B, -1, 6)


def warp_apply(Z: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Z [B, M, R, D] -> [B, M, R, D] with precomputed (idx, w) from warp_coords. Channels-last
    gather with a stride-0 index view (no index copy kept for the backward pass)."""
    B, M, NR, D = Z.shape
    Zp = Z.permute(0, 2, 3, 1).reshape(B, NR * D, M)
    g = torch.gather(Zp, 1, idx[:, :, None].expand(B, idx.shape[1], M)).view(B, NR * D, 6, M)
    return (g * w[..., None]).sum(2).view(B, NR, D, M).permute(0, 3, 1, 2)


def warp(Z: torch.Tensor, R: torch.Tensor, t: torch.Tensor, sphere: Sphere, grid: RangeGrid) -> torch.Tensor:
    """Z [B, M, R, D] in the head frame of pulse t -> [B, M, R, D] in the head frame of pulse t+k
    ((R, t): pose of head t+k in head t)."""
    return warp_apply(Z, *warp_coords(R, t, sphere, grid))
