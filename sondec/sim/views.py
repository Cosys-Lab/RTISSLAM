"""Training views for the place recognizer: anchors on a jittered grid over the drivable area of a
world, and around each anchor views at a random position offset and a uniformly random heading.

Pairs of views then cover "same place, any heading" (views of one anchor), near misses and near
places (neighbouring anchors, ~1 m apart) and aliased places (the repeated rooms of a floor).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

CLEAR_ANCHOR = 0.45      # m from any obstacle (or the outer walls) for an anchor
CLEAR_VIEW = 0.30        # m for a view (the robot's half width plus a margin)
ROBOT_R = 0.20           # m, the robot's half width (what must pass through a doorway)


def free_space(w, res: float = 0.05):
    """-> occupancy grid, origin, res, reachable (bool): free cells connected to the largest free region."""
    occ, origin, res = w.occupancy(res)
    occ = occ.copy()
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True                    # outer walls
    lab, n = label(~binary_dilation(occ, iterations=int(math.ceil(ROBOT_R / res))))
    if n == 0:
        return occ, origin, res, np.zeros_like(occ)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return occ, origin, res, lab == sizes.argmax()


def sample_views(w, rng: np.random.Generator, spacing: float = 1.0, per_anchor: int = 12, sigma: float = 0.3):
    """-> poses [N, 3] (x, y, yaw), anchor index [N], anchors [K, 2]."""
    occ, (x0, y0), res, reach = free_space(w)
    clear = distance_transform_edt(~occ) * res
    ok_anchor = reach & (clear >= CLEAR_ANCHOR)
    ok_view = reach & (clear >= CLEAR_VIEW)
    nx, ny = occ.shape
    gx = np.arange(x0 + spacing / 2, x0 + nx * res, spacing)
    gy = np.arange(y0 + spacing / 2, y0 + ny * res, spacing)
    A = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    A = A + rng.uniform(-0.25, 0.25, A.shape) * spacing
    cell = lambda p: (np.clip(((p[..., 0] - x0) / res).astype(int), 0, nx - 1),  # noqa: E731
                      np.clip(((p[..., 1] - y0) / res).astype(int), 0, ny - 1))
    A = A[ok_anchor[cell(A)]]
    poses, anchor = [], []
    for k, a in enumerate(A):
        for _ in range(per_anchor):
            for _try in range(50):
                p = a + rng.normal(0, sigma, 2)
                if ok_view[cell(p)]:
                    break
            else:
                p = a
            poses.append((p[0], p[1], rng.uniform(-math.pi, math.pi))); anchor.append(k)
    return np.asarray(poses, np.float64), np.asarray(anchor, np.int32), A
