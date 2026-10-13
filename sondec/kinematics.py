"""Robot kinematics: planar body + pan-tilt(-roll) head with a lever arm. Known robot geometry (not world
knowledge): both the simulator and the model use it.

Frames (all right-handed, x forward, y left, z up):
    world W   floor plane z = 0
    body B    origin on the floor under the robot centre, yaw about z
    head H    the sensor: emitter at its origin, ears at +-EAR_Y along its y axis.
              Directions u in the head frame: x = cos(el) cos(az), y = cos(el) sin(az), z = sin(el)
              (az + = left, el + = up; BatSLAM2's convention).

Head in body:  R_BH = Rz(pan) Ry(-tilt) Rx(roll)   (pan + = left, tilt + = up, roll + = left ear up)
               t_BH = PIVOT + R_BH LEVER           (pivot of the head unit, lever arm pivot -> sensor)
Joint order pan -> tilt -> roll (roll about the sensor's own forward axis). Roll is optional
(simulator v2 variant "roll"); without it the formulas reduce to the pan-tilt head.

Continuous acoustic flow (docs/acoustic_flow_3d.md; head_twist; tests/test_3d_flow_ode.py):
a static point p = r u in the head frame moves as  dp/dt = -v - w x p,  i.e.
    dr/dt = -v.u,     du/dt = -(v - (v.u) u) / r - w x u
with the head's own velocities (head frame; W = body yaw rate, v_b = body velocity in the body frame)
    w = (pan' + W) R_BH^T e_z - tilt' Rx(roll)^T e_y + roll' e_x
    v = R_BH^T (v_b + W e_z x t_BH) + w_joint x LEVER,   w_joint = w without the W term.
The warp (warp.py) uses the exact finite pose instead (relative_head_pose); the ODE is its
small-step limit.
"""
from __future__ import annotations

import torch

PIVOT = (0.05, 0.0, 0.35)       # m, pan-tilt pivot in the body frame
LEVER = (0.04, 0.0, 0.05)       # m, pivot -> sensor origin in the head frame (sensor at 0.40 m at zero tilt)
EAR_Y = 0.012                   # m, ears at +-EAR_Y (left = +y)
PAN_LIMIT = (-60.0, 60.0)       # deg
TILT_LIMIT = (-20.0, 40.0)      # deg (simulator v1: -30..40)
ROLL_LIMIT = (-15.0, 15.0)      # deg (variant "roll" only)


def rot_z(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def rot_x(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([torch.stack([o, z, z], -1), torch.stack([z, c, -s], -1), torch.stack([z, s, c], -1)], -2)


def rot_y(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)


def split_head(h):
    """Head angles [..., 2 or 3] (pan, tilt[, roll]) -> pan, tilt, roll (None without roll)."""
    return h[..., 0], h[..., 1], (h[..., 2] if h.shape[-1] > 2 else None)


def head_in_body(pan: torch.Tensor, tilt: torch.Tensor, roll: torch.Tensor | None = None):
    """pan, tilt[, roll] [...] (rad) -> R_BH [..., 3, 3], t_BH [..., 3]."""
    R = rot_z(pan) @ rot_y(-tilt)
    if roll is not None:
        R = R @ rot_x(roll)
    piv = torch.tensor(PIVOT, dtype=R.dtype, device=R.device)
    lev = torch.tensor(LEVER, dtype=R.dtype, device=R.device)
    return R, piv + R @ lev


def body_in_world(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pose [..., 3] (x, y, yaw) -> R_WB [..., 3, 3], t_WB [..., 3]."""
    R = rot_z(pose[..., 2])
    t = torch.stack([pose[..., 0], pose[..., 1], torch.zeros_like(pose[..., 0])], -1)
    return R, t


def head_in_world(pose: torch.Tensor, pan: torch.Tensor, tilt: torch.Tensor, roll: torch.Tensor | None = None):
    Rb, tb = body_in_world(pose)
    Rh, th = head_in_body(pan, tilt, roll)
    return Rb @ Rh, tb + (Rb @ th[..., None])[..., 0]


def integrate_se2(inc: torch.Tensor) -> torch.Tensor:
    """[..., K, 3] body-frame increments (dx, dy, dyaw) -> [..., K, 3] poses of pulses 1..K in the
    body frame of pulse 0."""
    x = torch.zeros_like(inc[..., 0, 0]); y = torch.zeros_like(x); a = torch.zeros_like(x)
    out = []
    for k in range(inc.shape[-2]):
        dx, dy, da = inc[..., k, 0], inc[..., k, 1], inc[..., k, 2]
        c, s = torch.cos(a), torch.sin(a)
        x, y, a = x + c * dx - s * dy, y + s * dx + c * dy, a + da
        out.append(torch.stack([x, y, a], -1))
    return torch.stack(out, -2)


def relative_head_pose(body_rel: torch.Tensor, pan0, tilt0, pan1, tilt1, roll0=None, roll1=None):
    """Pose of head 1 in the frame of head 0.
    body_rel [..., 3]: body pose 1 in body frame 0 (x, y, yaw); pan/tilt/roll [...] rad.
    Returns R [..., 3, 3], t [..., 3] with  p_H0 = R p_H1 + t."""
    R0, t0 = head_in_body(pan0, tilt0, roll0)
    R1, t1 = head_in_body(pan1, tilt1, roll1)
    Rb, tb = body_in_world(body_rel)              # body 1 in body 0
    R01 = Rb @ R1                                 # head 1 in body 0
    t01 = tb + (Rb @ t1[..., None])[..., 0]
    R0t = R0.transpose(-1, -2)
    return R0t @ R01, (R0t @ (t01 - t0)[..., None])[..., 0]


def head_twist(pan, tilt, roll, dpan, dtilt, droll, v_body, yaw_rate):
    """Analytical head velocities in the head frame (the continuous acoustic flow's v and w).
    Angles and rates [...] (rad, rad/s), v_body [..., 3] body velocity in the body frame, yaw_rate [...]
    (body yaw rate W). -> v [..., 3], w [..., 3]: a static point p (head frame) moves as dp/dt = -v - w x p."""
    Rbh, tbh = head_in_body(pan, tilt, roll)
    Rx = rot_x(roll)
    ez = torch.zeros_like(tbh); ez[..., 2] = 1.0
    ey = torch.zeros_like(tbh); ey[..., 1] = 1.0
    ex = torch.zeros_like(tbh); ex[..., 0] = 1.0
    RbhT = Rbh.transpose(-1, -2)
    up_h = (RbhT @ ez[..., None])[..., 0]                               # body vertical axis, head frame
    w_joint = dpan[..., None] * up_h - dtilt[..., None] * (Rx.transpose(-1, -2) @ ey[..., None])[..., 0] + droll[..., None] * ex
    w = w_joint + yaw_rate[..., None] * up_h
    lev = torch.tensor(LEVER, dtype=tbh.dtype, device=tbh.device).expand_as(tbh)
    v_b = v_body + yaw_rate[..., None] * torch.cross(ez, tbh, dim=-1)    # sensor origin velocity from the body
    v = (RbhT @ v_b[..., None])[..., 0] + torch.cross(w_joint, lev, dim=-1)
    return v, w


def dir_to_azel(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., 3] -> az, el (rad)."""
    az = torch.atan2(u[..., 1], u[..., 0])
    el = torch.atan2(u[..., 2], torch.hypot(u[..., 0], u[..., 1]))
    return az, el


def azel_to_dir(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], -1)
