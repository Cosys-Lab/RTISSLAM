"""Long drives through a floor for SLAM: a random tour of goals spread over the whole floor (each at
least `min_hop` m from the current position), A* paths between them, and a differential-drive robot that
DRIVES that path with pure pursuit: constant speed (one pulse every `ds` m), a lookahead of `lookahead` m,
a minimum turning radius `r_min` (corners are rounded), and turns in place (`turn_step` per pulse) only
when the path doubles back (target more than `turn_in_place` to the side, e.g. a U-turn at a goal).
Where rounding a corner would come closer than `r_robot` to an obstacle, it steers with a shorter
lookahead. Places are revisited often, from all directions.
"""
from __future__ import annotations

import math

import numpy as np

from .motion import Planner, _wrap
from .views import free_space


def _goal_path(P, rng, ok, x0, y0, res, length_m, min_hop):
    """The tour as one dense polyline [M, 2] (A* paths between random goals, resampled every 5 cm)."""
    def free_point():
        i, j = ok[rng.integers(len(ok))]
        return x0 + (i + 0.5) * res, y0 + (j + 0.5) * res

    pos = free_point()
    pts, total, fails = [np.asarray(pos)], 0.0, 0
    while total < length_m * 1.05 and fails < 200:
        goal = free_point()
        if math.hypot(goal[0] - pos[0], goal[1] - pos[1]) < min_hop:
            continue
        path = P.astar(pos, goal)
        if path is None or len(path) < 2:
            fails += 1
            continue
        for a, b in zip(path[:-1], path[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(math.ceil(L / 0.05)))
            for k in range(1, n + 1):
                pts.append(np.asarray(a) + k / n * (np.asarray(b) - np.asarray(a)))
            total += L
        pos = goal
    return np.asarray(pts)


def plan_tour(world, length_m: float, seed: int, ds: float = 0.15, lookahead: float = 0.45, r_min: float = 0.25,
              turn_step: float = math.radians(12), turn_in_place: float = math.radians(100), r_robot: float = 0.2,
              min_hop: float = 5.0) -> np.ndarray:
    """-> [N, 3] ground-truth poses (x, y, yaw), one per pulse."""
    rng = np.random.default_rng(seed)
    P = Planner(world, rng)
    occ, (x0, y0), res, reach = free_space(world)                          # largest connected free region
    ok = np.argwhere(reach & (P.clear >= 0.55))
    path = _goal_path(P, rng, ok, x0, y0, res, length_m, min_hop)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]       # arc length along the path
    x, y = path[0]
    th = math.atan2(*(path[min(10, len(path) - 1)] - path[0])[::-1])
    poses, idx, travelled = [(x, y, th)], 0, 0.0
    k_max = 1.0 / r_min
    while travelled < length_m and idx < len(path) - 2:
        # progress: the nearest path point in a window ahead (the path may cross itself)
        w = slice(idx, min(idx + 60, len(path)))
        idx = idx + int(np.argmin(np.hypot(path[w, 0] - x, path[w, 1] - y)))
        cand = None
        for la in (lookahead, lookahead / 2, 0.15):
            j = min(int(np.searchsorted(seg, seg[idx] + la)), len(path) - 1)
            tx, ty = path[j]
            alpha = _wrap(math.atan2(ty - y, tx - x) - th)
            if abs(alpha) > turn_in_place:                                   # doubles back: turn on the spot
                cand = (x, y, th + math.copysign(min(turn_step, abs(alpha)), alpha))
                break
            k = max(-k_max, min(k_max, 2 * math.sin(alpha) / max(la, 1e-3)))  # pure-pursuit curvature, limited
            th_mid = th + k * ds / 2
            nx, ny = x + ds * math.cos(th_mid), y + ds * math.sin(th_mid)
            if P.clearance(nx, ny) >= r_robot:
                cand = (nx, ny, th + k * ds)
                break
        if cand is None:                                                     # no free step: turn toward the path
            cand = (x, y, th + math.copysign(min(turn_step, abs(alpha)), alpha))
        moved = math.hypot(cand[0] - x, cand[1] - y)
        x, y, th = cand
        travelled += moved
        poses.append((x, y, th))
    return np.asarray(poses, np.float64)
