"""Sparse scenes and smooth drives (user request 2026-10-06: "way less points, each measurement maybe
20-30 reflections max"; "smooth trajectories").

Scene: the Recognizer's point world (worlds.build_points: isotropic point reflectors, no floor, walls or
ceiling; strength log-normal, size 1-10 cm with Rayleigh roll-off, random spectral tilt), at a density
chosen so that ~20-30 points lie in the sensor's frontal hemisphere within the 5 m image range.
Points closer than CLEAR to the sensor's path are removed (the robot does not drive through them).

Drive: constant speed (one pulse every DS m) along a smooth path: the curvature is a smoothed random
process (Ornstein-Uhlenbeck, |kappa| <= KMAX), plus a steering term back into the arena near its edges.
No turns on the spot.
"""
from __future__ import annotations

import math

import numpy as np

from .worlds import build_points

DS = 0.08            # m per pulse
KMAX = 1 / 0.8       # 1/m, tightest turn radius 0.8 m
CLEAR = 0.5          # m, minimum distance from a reflector to the sensor path
SENSOR_Z = 0.4


def smooth_drive(n: int, size: float, rng: np.random.Generator) -> np.ndarray:
    """-> [n, 3] (x, y, yaw), inside [1.5, size - 1.5]^2 most of the time."""
    x, y = rng.uniform(0.35, 0.65, 2) * size
    th = rng.uniform(-math.pi, math.pi)
    k, out = 0.0, []
    tau = 1.5                                                     # m, correlation length of the curvature
    c = size / 2
    for _ in range(n):
        out.append((x, y, th))
        # steer back when heading out near the edge
        d_edge = min(x, y, size - x, size - y)
        to_c = math.atan2(c - y, c - x)
        err = math.atan2(math.sin(to_c - th), math.cos(to_c - th))
        pull = max(0.0, (2.5 - d_edge) / 2.5) * 1.5 * err
        k += -(k * DS / tau) + 0.5 * math.sqrt(2 * DS / tau) * rng.standard_normal()
        kk = float(np.clip(k + pull, -KMAX, KMAX))
        th_mid = th + kk * DS / 2
        x, y = x + DS * math.cos(th_mid), y + DS * math.sin(th_mid)
        th = th + kk * DS
    return np.asarray(out)


def build_sparse(name: str, seed: int, n: int, density: float = 0.25):
    """-> world (dict arrays + World), ground-truth poses [n, 3]."""
    w = build_points(name, seed, density=density, height=3.0)
    rng = np.random.default_rng(seed + 99)
    S = w.bounds[1]
    gt = smooth_drive(n, S, rng)
    arr = w.arrays()
    sp = np.c_[gt[:, :2], np.full(len(gt), SENSOR_Z)]
    p = arr["pos"]
    dmin = np.full(len(p), np.inf)
    for i0 in range(0, len(sp), 200):
        dmin = np.minimum(dmin, np.linalg.norm(p[:, None] - sp[None, i0:i0 + 200], axis=-1).min(1))
    keep = dmin >= CLEAR
    for k in ("pos", "normal", "area", "mat", "obj", "two"):
        arr[k] = arr[k][keep]
    return arr, gt


def _walls_2d(S: float, gt: np.ndarray, rng: np.random.Generator, n_part=(2, 4), H: float = 1.0):
    """Room walls + L-shaped partitions (user request 2026-10-06: "add walls ... and also corners").
    -> World with: the enclosing room [0, S]^2 (specular walls, height H, no floor/ceiling; strong vertical
    concave-corner lines: a single-bounce stand-in for the dihedral retro-reflection, which the simulator
    has no multipath for), and n_part L-shaped interior partitions (thin occluding boxes, 0.1 m thick; the
    joint's inner side gets a concave-corner line, the box edges are convex diffracting edges), placed at
    least 0.7 m from the robot's path."""
    from .worlds import MAT, World
    MAT.setdefault("corner2d", (0.3, 0.0, 0.0))          # concave corner: strong isotropic line (retro-reflector)
    w = World("walls", "sparse2d", (0, S, 0, S), rng=rng)
    o = w.new_obj("walls", tilt=rng.normal(0, 1.5))
    dens = 150.0                                        # dense: smooth specular returns
    w.rect((0, S / 2, H / 2), (0, S / 2, 0), (0, 0, H / 2), "wall", o, density=dens)
    w.rect((S, S / 2, H / 2), (0, 0, H / 2), (0, S / 2, 0), "wall", o, density=dens)
    w.rect((S / 2, 0, H / 2), (0, 0, H / 2), (S / 2, 0, 0), "wall", o, density=dens)
    w.rect((S / 2, S, H / 2), (S / 2, 0, 0), (0, 0, H / 2), "wall", o, density=dens)
    oc = w.new_obj("corners")
    for x in (0, S):
        for y in (0, S):
            w.line((x, y, 0), (x, y, H), "corner2d", oc)
    path = gt[:, :2]

    def clear(a, b, m=0.7):
        t = np.linspace(0, 1, 40)[:, None]
        seg = a[None] + t * (b - a)[None]
        return np.min(np.linalg.norm(seg[:, None] - path[None], axis=-1)) >= m

    k_target = int(rng.integers(n_part[0], n_part[1] + 1))
    placed = 0
    for _ in range(400):
        if placed >= k_target:
            break
        c = rng.uniform(1.0, S - 1.0, 2)
        sx, sy = rng.choice([-1, 1], 2)
        L1, L2 = rng.uniform(1.2, 3.0, 2)
        a1 = np.clip(c + np.array([sx * L1, 0]), 0.3, S - 0.3); a2 = np.clip(c + np.array([0, sy * L2]), 0.3, S - 0.3)
        if not (clear(c, a1) and clear(c, a2)):
            continue
        for a in (a1, a2):
            mid = (c + a) / 2; d = a - c; L = np.linalg.norm(d)
            w.box((mid[0], mid[1], H / 2), (L / 2 + 0.05, 0.05, H / 2), float(np.arctan2(d[1], d[0])), "wall", "partition",
                  edges=True, edge_mat="edge")
        inner = c + 0.05 * np.array([sx, sy])                # concave side of the L joint
        w.line((inner[0], inner[1], 0), (inner[0], inner[1], H), "corner2d", w.new_obj("corner"))
        placed += 1
    return w


def build_sparse2d(name: str, seed: int, n: int, density: float = 0.9, z_range=(0.25, 0.55), walls: bool = False):
    """2D variant (user request 2026-10-06: "focus on 2D"): isotropic point reflectors near the sensor's
    horizontal plane (height uniform in z_range; the sensor is at SENSOR_Z = 0.4 m, so within ~+-9 deg
    elevation at 1 m, less further away), `density` per m^2 of floor -> ~20-30 within 5 m in the frontal
    half plane. Strength log-normal (sigma 5 dB in power), size 1-10 cm (Rayleigh roll-off), random
    spectral tilt, as worlds.build_points. Same smooth drives; points within CLEAR of the path removed.
    -> world arrays (dict), ground-truth poses [n, 3]."""
    rng = np.random.default_rng(seed)
    S = rng.uniform(12, 16)
    M = 3.0
    L = S + 2 * M
    k = rng.poisson(density * L * L)
    p = np.stack([rng.uniform(-M, S + M, k), rng.uniform(-M, S + M, k), rng.uniform(*z_range, k)], 1)
    d = 0.05 * 10 ** (rng.normal(0, 5, k) / 10)
    gt = smooth_drive(n, S, np.random.default_rng(seed + 99))
    if walls:                                            # points only inside the room
        inside = (p[:, 0] > 0.2) & (p[:, 0] < S - 0.2) & (p[:, 1] > 0.2) & (p[:, 1] < S - 0.2)
        p, d, k = p[inside], d[inside], int(inside.sum())
    sp = np.c_[gt[:, :2], np.full(len(gt), SENSOR_Z)]
    dmin = np.full(k, np.inf)
    for i0 in range(0, len(sp), 200):
        dmin = np.minimum(dmin, np.linalg.norm(p[:, None] - sp[None, i0:i0 + 200], axis=-1).min(1))
    keep = dmin >= CLEAR
    p, d = p[keep], d[keep]
    k = len(p)
    arr = {"pos": p.astype(np.float32), "normal": np.zeros((k, 3), np.float32), "area": np.ones(k, np.float32),
           "mat": np.c_[d, np.zeros(k), np.zeros(k)].astype(np.float32), "obj": np.arange(k, dtype=np.int32),
           "two": np.zeros(k, bool), "obj_size": rng.uniform(0.01, 0.1, k).astype(np.float32),
           "obj_tilt": rng.normal(0, 3, k).astype(np.float32), "obj_name": np.array(["point"] * k),
           "boxes": np.zeros((0, 7), np.float32), "capsules": np.zeros((0, 7), np.float32),
           "bounds": np.array([0, S, 0, S], np.float32)}
    if walls:
        wa = _walls_2d(S, gt, np.random.default_rng(seed + 7)).arrays()
        off = len(arr["obj_size"])
        for key in ("pos", "normal", "area", "mat", "two"):
            arr[key] = np.concatenate([arr[key], wa[key].astype(arr[key].dtype)])
        arr["obj"] = np.concatenate([arr["obj"], wa["obj"] + off]).astype(np.int32)
        arr["obj_size"] = np.concatenate([arr["obj_size"], wa["obj_size"]]).astype(np.float32)
        arr["obj_tilt"] = np.concatenate([arr["obj_tilt"], wa["obj_tilt"]]).astype(np.float32)
        arr["obj_name"] = np.concatenate([arr["obj_name"], wa["obj_name"]])
        arr["boxes"] = wa["boxes"]
    return arr, gt
