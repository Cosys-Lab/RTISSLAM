"""Grid path planning on an occupancy grid: A* on a clearance costmap, then elastic smoothing.

Used twice: by the simulator to turn a tour of waypoints into a smooth driving line on the ground-truth floor plan,
and by the navigation stack on the RTISSLAM occupancy grid (the global planner, Nav2-style).

    cost per step = length x (1 + K_INFL exp(-(clearance - R_ROBOT) / DECAY))      cells with clearance < R_ROBOT
                                                                                    (+ margin) are lethal
Smoothing: x_i <- x_i + a (x_{i-1} + x_{i+1} - 2 x_i) + b grad(clearance) where clearance < DESIRED; endpoints fixed;
then resampled every ds.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

R_ROBOT = 0.25          # m, robot radius
MARGIN = 0.07           # m, lethal = clearance < R_ROBOT + MARGIN
DESIRED = 0.9           # m, smoothing pushes away from obstacles closer than this
K_INFL, DECAY = 6.0, 0.35


class GridPlanner:
    def __init__(self, occ: np.ndarray, origin, res: float, coarse: float = 0.1):
        """occ [nx, ny] bool (True = blocked; index [i, j] = cell centred at origin + (i + 0.5, j + 0.5) res)."""
        self.origin, self.res = np.asarray(origin, float), float(res)
        free = ~occ.copy()
        free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
        self.clear = distance_transform_edt(free) * self.res                         # m
        self.gy, self.gx = None, None
        gx, gy = np.gradient(self.clear, self.res)
        self.grad = (gx, gy)
        self.cs = max(1, int(round(coarse / self.res)))
        cc = self.clear[::self.cs, ::self.cs]
        self.cres = self.res * self.cs
        self.lethal = cc < R_ROBOT + MARGIN
        self.cost = 1.0 + K_INFL * np.exp(-np.clip(cc - R_ROBOT, 0, None) / DECAY)

    # ------------------------------------------------------------------ queries
    def clearance(self, p) -> np.ndarray:
        p = np.atleast_2d(np.asarray(p, float))
        ij = ((p - self.origin) / self.res - 0.5).T
        return map_coordinates(self.clear, ij, order=1, mode="constant", cval=0.0)

    def _grad(self, p):
        ij = ((p - self.origin) / self.res - 0.5).T
        return np.stack([map_coordinates(g, ij, order=1, mode="nearest") for g in self.grad], 1)

    def _to_c(self, p):
        return tuple(int(v) for v in np.floor((np.asarray(p) - self.origin) / self.cres))

    def _nearest_free(self, c, rmax=6):
        nx, ny = self.lethal.shape
        for r in range(rmax + 1):
            best = None
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    v = (c[0] + di, c[1] + dj)
                    if 0 <= v[0] < nx and 0 <= v[1] < ny and not self.lethal[v]:
                        d = di * di + dj * dj
                        if best is None or d < best[0]:
                            best = (d, v)
            if best:
                return best[1]
        return None

    # ------------------------------------------------------------------ A*
    def astar(self, start, goal):
        """-> [K, 2] waypoints (coarse cell centres, start and goal exact) or None."""
        s, g = self._nearest_free(self._to_c(start)), self._nearest_free(self._to_c(goal))
        if s is None or g is None:
            return None
        nx, ny = self.lethal.shape
        cost, lethal = self.cost, self.lethal
        dist = {s: 0.0}
        prev = {}
        pq = [(0.0, 0.0, s)]
        nb = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        while pq:
            _, d, u = heapq.heappop(pq)
            if u == g:
                break
            if d > dist.get(u, 1e18) + 1e-12:
                continue
            for di, dj, L in nb:
                v = (u[0] + di, u[1] + dj)
                if not (0 <= v[0] < nx and 0 <= v[1] < ny) or lethal[v]:
                    continue
                nd = d + L * 0.5 * (cost[u] + cost[v])
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd + math.hypot(g[0] - v[0], g[1] - v[1]), nd, v))
        if g != s and g not in prev:
            return None
        cells, u = [g], g
        while u != s:
            u = prev[u]
            cells.append(u)
        pts = self.origin + (np.array(cells[::-1], float) + 0.5) * self.cres
        pts[0], pts[-1] = start, goal
        return pts

    # ------------------------------------------------------------------ smoothing
    def smooth(self, P: np.ndarray, ds: float = 0.05, iters: int = 60, a: float = 0.2, b: float = 0.08,
               closed: bool = False) -> np.ndarray:
        P = resample(P, 0.1, closed)
        for _ in range(iters):
            if closed:
                lap = np.roll(P, 1, 0) + np.roll(P, -1, 0) - 2 * P
            else:
                lap = np.zeros_like(P)
                lap[1:-1] = P[:-2] + P[2:] - 2 * P[1:-1]
            c = self.clearance(P)
            push = self._grad(P) * np.clip(DESIRED - c, 0, None)[:, None]
            if not closed:
                push[0] = push[-1] = 0
            Pn = P + a * lap + b * push
            ok = self.clearance(Pn) >= np.minimum(c, R_ROBOT + MARGIN)               # never move into an obstacle
            P[ok] = Pn[ok]
        return resample(P, ds, closed)

    def plan(self, start, goal, ds: float = 0.05):
        P = self.astar(start, goal)
        return None if P is None else self.smooth(P, ds)

    def tour(self, waypoints, ds: float = 0.05, closed: bool = True) -> np.ndarray:
        """A* between consecutive waypoints (closing the loop if closed), concatenated and smoothed."""
        W = list(waypoints) + ([waypoints[0]] if closed else [])
        parts = []
        for a_, b_ in zip(W[:-1], W[1:]):
            seg = self.astar(a_, b_)
            if seg is None:
                raise RuntimeError(f"no path {a_} -> {b_}")
            parts.append(seg[:-1] if parts or closed else seg[:-1])
        P = np.vstack(parts + ([] if closed else [np.asarray(W[-1], float)[None]]))
        return self.smooth(P, ds, closed=closed)


def resample(P: np.ndarray, ds: float, closed: bool = False) -> np.ndarray:
    Q = np.vstack([P, P[:1]]) if closed else P
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    t = np.arange(0, s[-1] - (1e-9 if closed else -1e-9), ds)
    if not closed and t[-1] < s[-1] - 1e-6:
        t = np.r_[t, s[-1]]
    return np.c_[np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])]
