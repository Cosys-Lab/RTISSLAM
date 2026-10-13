"""Metric refinement of BatSLAM view links by registering eRTIS echo submaps (the RTIS-specific part).

BatSLAM's link says "the robot is at the anchor pose (shifted along its heading)" with a loose sigma. The eRTIS sees
WHERE its echoes are, so a recognised revisit can be registered: the query submap (detections of the last
SUBMAP_M of travel, in the query node's frame, odometry-composed) against the anchor submap (detections within
+-SUBMAP_M of the anchor node), by correlative scan matching on a Gaussian likelihood field: coarse grid search
(+-0.5 m, +-15 deg) around BatSLAM's measurement, then a fine search. A registration is accepted if its score is high
and sharp (well above the search-window median); the link then carries the registered (dx, dy, dtheta) with a tight
sigma, otherwise BatSLAM's own measurement is kept.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass
class RegisterConfig:
    enabled: bool = True
    submap_m: float = 1.2           # m of travel per submap side
    res: float = 0.02               # likelihood field resolution
    sigma_field: float = 0.05       # m
    search_xy: float = 0.5          # m
    step_xy: float = 0.04
    search_th: float = math.radians(15)
    step_th: float = math.radians(1.5)
    max_points: int = 500
    min_points: int = 25
    min_score: float = 0.25         # mean field value of the query points at the optimum
    min_contrast: float = 2.0       # optimum / median score over the coarse search
    sigma: tuple = (0.06, 0.06, 0.03)


def _T(x, y, th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def _v(T):
    return np.array([T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])])


class Field:
    def __init__(self, pts: np.ndarray, cfg: RegisterConfig):
        self.cfg = cfg
        m = cfg.search_xy + 1.0
        self.x0, self.y0 = pts[:, 0].min() - m, pts[:, 1].min() - m
        nx = int((pts[:, 0].max() + m - self.x0) / cfg.res) + 1
        ny = int((pts[:, 1].max() + m - self.y0) / cfg.res) + 1
        F = np.zeros((nx, ny))
        i = ((pts[:, 0] - self.x0) / cfg.res).astype(int)
        j = ((pts[:, 1] - self.y0) / cfg.res).astype(int)
        np.add.at(F, (i, j), 1.0)
        s = cfg.sigma_field / cfg.res
        F = gaussian_filter(F, s) * (2 * math.pi * s * s)                   # a single point peaks at ~1
        self.F = np.minimum(F, 1.0)

    def score(self, P: np.ndarray) -> np.ndarray:
        """P [..., n, 2] -> mean field value over n."""
        ij = np.stack([(P[..., 0] - self.x0) / self.cfg.res, (P[..., 1] - self.y0) / self.cfg.res])
        v = map_coordinates(self.F, ij.reshape(2, -1), order=1, mode="constant", cval=0.0)
        return v.reshape(P.shape[:-1]).mean(-1)


def register(Q: np.ndarray, A: np.ndarray, init, cfg: RegisterConfig, rng=None):
    """Q [n, 2] query points (query frame), A [m, 2] anchor points (anchor frame), init = query pose in the anchor
    frame (x, y, th) -> (pose (x, y, th) or None, score, contrast)."""
    if len(Q) < cfg.min_points or len(A) < cfg.min_points:
        return None, 0.0, 0.0
    if len(Q) > cfg.max_points:
        rng = rng or np.random.default_rng(0)
        Q = Q[rng.choice(len(Q), cfg.max_points, replace=False)]
    F = Field(A, cfg)
    x0, y0, t0 = init

    def search(xc, yc, tc, rx, sx, rt, st):
        ths = tc + np.arange(-rt, rt + 1e-9, st)
        dx = np.arange(-rx, rx + 1e-9, sx)
        best = (-1.0, None)
        allsc = []
        for th in ths:
            c, s = math.cos(th), math.sin(th)
            R = Q @ np.array([[c, s], [-s, c]])                             # rotate (row vectors)
            P = R[None, None] + np.stack(np.meshgrid(xc + dx, yc + dx, indexing="ij"), -1)[:, :, None, :]
            sc = F.score(P)                                                 # [nx, ny]
            allsc.append(sc)
            k = np.unravel_index(np.argmax(sc), sc.shape)
            if sc[k] > best[0]:
                best = (float(sc[k]), (xc + dx[k[0]], yc + dx[k[1]], th))
        return best, np.concatenate([a.ravel() for a in allsc])
    (sc, p), allsc = search(x0, y0, t0, cfg.search_xy, cfg.step_xy, cfg.search_th, cfg.step_th)
    contrast = sc / max(float(np.median(allsc)), 1e-6)
    (sc2, p2), _ = search(p[0], p[1], p[2], 1.5 * cfg.step_xy, 0.01, 1.5 * cfg.step_th, math.radians(0.25))
    if sc2 >= sc:
        sc, p = sc2, p2
    ok = sc >= cfg.min_score and contrast >= cfg.min_contrast
    return (np.array(p) if ok else None), sc, contrast


def compose_submap(key: int, pts: list, odom: list, s: np.ndarray, half_m: float, causal: bool) -> np.ndarray:
    """Points of the nodes within half_m of travel of `key` (causal: only earlier nodes), in key's frame."""
    T0inv = np.linalg.inv(_T(*odom[key]))
    out = []
    k = key
    while k >= 0 and s[key] - s[k] <= half_m:
        if len(pts[k]):
            M = T0inv @ _T(*odom[k])
            out.append(pts[k] @ M[:2, :2].T + M[:2, 2])
        k -= 1
    if not causal:
        k = key + 1
        while k < len(pts) and s[k] - s[key] <= half_m:
            if len(pts[k]):
                M = T0inv @ _T(*odom[k])
                out.append(pts[k] @ M[:2, :2].T + M[:2, 2])
            k += 1
    return np.vstack(out) if out else np.zeros((0, 2))
