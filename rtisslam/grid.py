"""Occupancy grid from eRTIS energyscapes, anchored to pose-graph nodes.

Keyframes (every KF_SPACING m or KF_TURN rad of travel) store their evidence in the SENSOR frame:
    occupied points   every energyscape cell whose range-compensated level is above the view's noise floor + DET_DB
                      and within PEAK_DB of the view peak (log-odds +L_OCC each, weighted by level)
    free points       along each azimuth up to the first detection minus FREE_GAP (capped at R_FREE); where an
                      azimuth has no detection only up to R_EMPTY: specular walls are invisible at grazing incidence,
                      so "no echo" does NOT mean "free" far out
    footprint         the robot disk (free): where the robot is, nothing is
The global grid is re-rendered from the CURRENT node poses (log-odds sums, clipped), so loop closures that move the
pose graph move the evidence with it. Rendering is a scatter-add of all keyframe points (numpy bincount).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridConfig:
    method: str = "deconv"
    res: float = 0.05
    kf_spacing: float = 0.25        # m
    kf_turn: float = math.radians(15)
    det_db: float = 22.0            # detection threshold above the view's median (range-compensated) level
    peak_db: float = 30.0           # ...and within this of the view peak
    max_range: float = 4.5          # m
    free_gap: float = 0.12          # m
    r_free: float = 3.5             # m
    r_empty: float = 0.6            # m
    free_step: float = 0.04         # m along a ray
    free_az_step: int = 2           # every n-th azimuth for free rays
    l_occ: float = 0.9
    l_free: float = 0.25
    l_min: float = -3.0
    l_max: float = 6.0
    robot_r: float = 0.25
    mount: tuple = (0.10, 0.0)      # sensor position in base_link
    occ_thresh: float = 1.2         # log-odds: occupied (OccupancyGrid 100)
    free_thresh: float = -0.6       # log-odds: free (OccupancyGrid 0); between = unknown (-1)


@dataclass
class Keyframe:
    node: int
    occ: np.ndarray                 # [n, 3] x, y (base_link frame), log-odds
    free: np.ndarray                # [m, 2] x, y (base_link frame)


@dataclass
class KeyframeGrid:
    cfg: GridConfig
    ranges: np.ndarray
    azimuths: np.ndarray
    keyframes: list = field(default_factory=list)

    def __post_init__(self):
        c = self.cfg
        self.r = np.asarray(self.ranges, float)
        self.az = np.asarray(self.azimuths, float)
        self.keep = self.r <= c.max_range
        self.tvg = 40 * np.log10(self.r)[:, None]
        g = np.arange(-c.robot_r, c.robot_r + 1e-9, c.res)
        X, Y = np.meshgrid(g, g)
        m = X ** 2 + Y ** 2 <= c.robot_r ** 2
        self.foot = np.c_[X[m], Y[m]]
        self._last = None

    # ------------------------------------------------------------------ evidence
    def want_keyframe(self, odom) -> bool:
        if self._last is None:
            return True
        d = np.hypot(odom[0] - self._last[0], odom[1] - self._last[1])
        return d >= self.cfg.kf_spacing or abs(math.remainder(odom[2] - self._last[2], 2 * math.pi)) >= self.cfg.kf_turn

    def evidence(self, es_db: np.ndarray):
        """-> occupied points [n, 3] (x, y, l) and free points [m, 2], in base_link."""
        c = self.cfg
        v = (np.asarray(es_db, np.float32) + self.tvg)[self.keep]                  # [R', D]
        r = self.r[self.keep]
        floor = np.median(v)
        thr = max(floor + c.det_db, v.max() - c.peak_db)
        det = v > thr
        ri, ai = np.nonzero(det)
        w = np.clip((v[ri, ai] - thr) / 10.0, 0, 1) * 0.6 + 0.4
        mx, my = c.mount
        ox = mx + r[ri] * np.cos(self.az[ai])
        oy = my + r[ri] * np.sin(self.az[ai])
        occ = np.c_[ox, oy, c.l_occ * w].astype(np.float32)
        first = np.full(len(self.az), np.inf)
        if len(ri):
            np.minimum.at(first, ai, r[ri])
        reach = np.where(np.isfinite(first), np.minimum(first - c.free_gap, c.r_free), c.r_empty)
        pts = []
        for j in range(0, len(self.az), c.free_az_step):
            s = np.arange(0.05, reach[j], c.free_step)
            if len(s):
                pts.append(np.c_[mx + s * math.cos(self.az[j]), my + s * math.sin(self.az[j])])
        free = np.vstack(pts + [self.foot]).astype(np.float32)
        return occ, free

    def add(self, node: int, odom, es_db: np.ndarray) -> Keyframe:
        occ, free = self.evidence(es_db)
        kf = Keyframe(node, occ, free)
        self.keyframes.append(kf)
        self._last = np.asarray(odom, float).copy()
        return kf

    # ------------------------------------------------------------------ rendering
    def render(self, poses: np.ndarray, bounds=None, keyframes=None):
        """poses [N, 3] current node estimates -> (log-odds grid [nx, ny], origin (x0, y0), res).
        Index [i, j] = cell centred at origin + (i + 0.5, j + 0.5) res."""
        c = self.cfg
        kfs = self.keyframes if keyframes is None else keyframes
        P = np.asarray(poses)[[k.node for k in kfs]] if kfs else np.zeros((0, 3))
        if bounds is None:
            if not len(P):
                bounds = (-5, 5, -5, 5)
            else:
                m = c.max_range + 0.5
                bounds = (P[:, 0].min() - m, P[:, 0].max() + m, P[:, 1].min() - m, P[:, 1].max() + m)
        x0, x1, y0, y1 = bounds
        nx, ny = int(math.ceil((x1 - x0) / c.res)), int(math.ceil((y1 - y0) / c.res))
        L = np.zeros(nx * ny)
        occ_xy, occ_l, free_xy = [], [], []
        for kf, (x, y, th) in zip(kfs, P):
            ct, st = math.cos(th), math.sin(th)
            R = np.array([[ct, -st], [st, ct]])
            if len(kf.occ):
                occ_xy.append(kf.occ[:, :2] @ R.T + (x, y)); occ_l.append(kf.occ[:, 2])
            free_xy.append(kf.free @ R.T + (x, y))

        def idx(xy):
            i = np.floor((xy[:, 0] - x0) / c.res).astype(np.int64)
            j = np.floor((xy[:, 1] - y0) / c.res).astype(np.int64)
            ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
            return i[ok] * ny + j[ok], ok
        if free_xy:
            k, _ = idx(np.vstack(free_xy))
            L -= c.l_free * np.bincount(k, minlength=nx * ny)
        if occ_xy:
            k, ok = idx(np.vstack(occ_xy))
            L += np.bincount(k, weights=np.concatenate(occ_l)[ok], minlength=nx * ny)
        return np.clip(L.reshape(nx, ny), c.l_min, c.l_max), (x0, y0), c.res

    def ternary(self, L: np.ndarray) -> np.ndarray:
        """log-odds -> nav_msgs/OccupancyGrid values: 100 occupied, 0 free, -1 unknown (int8)."""
        out = np.full(L.shape, -1, np.int8)
        out[L >= self.cfg.occ_thresh] = 100
        out[L <= self.cfg.free_thresh] = 0
        return out

    # ------------------------------------------------------------------ persistence
    def state(self) -> dict:
        return {"kf_node": np.array([k.node for k in self.keyframes], np.int64),
                "kf_occ_n": np.array([len(k.occ) for k in self.keyframes], np.int64),
                "kf_free_n": np.array([len(k.free) for k in self.keyframes], np.int64),
                "kf_occ": np.vstack([k.occ for k in self.keyframes]) if self.keyframes else np.zeros((0, 3), np.float32),
                "kf_free": np.vstack([k.free for k in self.keyframes]) if self.keyframes else np.zeros((0, 2), np.float32)}

    def load_state(self, d: dict) -> None:
        self.keyframes = []
        oo = np.r_[0, np.cumsum(d["kf_occ_n"])]
        ff = np.r_[0, np.cumsum(d["kf_free_n"])]
        for i, n in enumerate(d["kf_node"]):
            self.keyframes.append(Keyframe(int(n), d["kf_occ"][oo[i]:oo[i + 1]], d["kf_free"][ff[i]:ff[i + 1]]))
