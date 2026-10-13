"""Differential-drive robot with realistic odometry errors, and a smooth path follower.

Robot: unicycle kinematics x' = v cos(th), y' = v sin(th), th' = w, integrated at dt. The odometry (what the robot
reports) has a scale error on distance, a heading bias (a fraction of the rotation, as the BatSLAM2 paper's b),
and white noise per step; the gyro reports the yaw rate with noise. Ground truth is kept separately.

Driver: a smooth closed path through the waypoints of a driving line (straight segments with circular fillets,
arc-length resampling), followed by pure pursuit with a curvature-limited angular rate and a speed that slows in turns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OdomNoise:
    scale: float = 0.02            # relative distance scale error
    heading_bias: float = 0.03     # extra rotation per rotation (BatSLAM2's b x 0.03 rad/rad)
    sigma_d: float = 0.01          # m per sqrt(m) random distance noise
    sigma_th: float = 0.002        # rad per sqrt(m) random heading noise
    gyro_sigma: float = 0.002      # rad/s gyro noise


@dataclass
class DiffDrive:
    pose: np.ndarray = field(default_factory=lambda: np.zeros(3))     # true x, y, yaw
    odom: np.ndarray = field(default_factory=lambda: np.zeros(3))     # odometry pose (drifts)
    noise: OdomNoise = field(default_factory=OdomNoise)
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.odom = self.pose.copy()

    def step(self, v: float, w: float, dt: float):
        """Integrate the true pose and the odometry over dt; returns the gyro yaw rate."""
        x, y, th = self.pose
        thm = th + w * dt / 2
        self.pose = np.array([x + v * dt * math.cos(thm), y + v * dt * math.sin(thm), _wrap(th + w * dt)])
        d = abs(v) * dt
        n = self.noise
        dv = v * dt * (1 + n.scale) + n.sigma_d * math.sqrt(d) * self.rng.standard_normal()
        dth = w * dt * (1 + n.heading_bias) + n.sigma_th * math.sqrt(d) * self.rng.standard_normal()
        ox, oy, oth = self.odom
        othm = oth + dth / 2
        self.odom = np.array([ox + dv * math.cos(othm), oy + dv * math.sin(othm), _wrap(oth + dth)])
        return w + n.gyro_sigma * self.rng.standard_normal()


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def smooth_closed_path(waypoints, radius: float = 0.9, ds: float = 0.05) -> np.ndarray:
    """Closed polyline -> straight segments joined by circular fillets of the given radius (shrunk where a segment
    is too short), resampled every ds metres -> [K, 2]. Keeps the path on the corridor centre lines."""
    P = np.asarray(waypoints, float)
    n = len(P)
    pts = []
    for k in range(n):
        a, b, c = P[k - 1], P[k], P[(k + 1) % n]
        u, v = b - a, c - b
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        u, v = u / lu, v / lv
        phi = math.acos(float(np.clip(u @ v, -1, 1)))                  # turn angle
        if phi < 1e-3:
            pts.append(b)
            continue
        t = min(radius * math.tan(phi / 2), 0.5 * lu, 0.5 * lv)       # tangent length
        r = t / math.tan(phi / 2)
        p0 = b - t * u
        nrm = np.array([-u[1], u[0]]) * (1 if u[0] * v[1] - u[1] * v[0] > 0 else -1)
        ctr = p0 + r * nrm
        a0 = math.atan2(*(p0 - ctr)[::-1])
        sgn = 1 if u[0] * v[1] - u[1] * v[0] > 0 else -1
        for th in np.linspace(0, phi, max(3, int(r * phi / 0.02))):
            pts.append(ctr + r * np.array([math.cos(a0 + sgn * th), math.sin(a0 + sgn * th)]))
    Q = np.vstack(pts + [pts[0]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    t = np.arange(0, s[-1], ds)
    return np.c_[np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])]


@dataclass
class PurePursuit:
    """Pure pursuit along a path; the closest-point search only looks WINDOW points ahead (hairpins: never skips to the
    return leg); heading errors above TURN_IN_PLACE rotate on the spot (differential drive)."""
    path: np.ndarray                    # [K, 2], closed (or open with closed=False)
    lookahead: float = 0.6
    v_max: float = 0.4                  # m/s
    w_max: float = 0.8                  # rad/s
    closed: bool = True
    window: int = 12
    turn_in_place: float = math.radians(60)
    _i: int = 0
    _turn: float = 0.0                  # turning on the spot: direction (+-1), 0 = driving

    @property
    def index(self) -> int:
        return self._i

    def done(self, pose, tol: float = 0.15) -> bool:
        return not self.closed and self._i >= len(self.path) - 2 and np.linalg.norm(self.path[-1] - pose[:2]) < tol

    def command(self, pose) -> tuple[float, float]:
        x, y, th = pose
        K = len(self.path)
        idx = self._i + np.arange(0, self.window)
        idx = idx % K if self.closed else np.clip(idx, 0, K - 1)
        d = np.linalg.norm(self.path[idx] - [x, y], axis=1)
        k = int(np.argmin(d))
        reached = np.nonzero(d < 0.25)[0]                                  # also pass points already reached (tips)
        if len(reached):
            k = max(k, int(reached[-1]))
        self._i = int(idx[k])
        ds = float(np.linalg.norm(self.path[1] - self.path[0])) or 0.05
        j = self._i + max(1, int(round(self.lookahead / ds)))             # carrot: lookahead in ARC LENGTH (hairpins)
        if not self.closed:
            j = min(j, K - 1)
        gx, gy = self.path[j % K]
        a = _wrap(math.atan2(gy - y, gx - x) - th)
        if self._turn == 0.0 and abs(a) > self.turn_in_place:
            self._turn = math.copysign(1.0, a)
        if self._turn != 0.0:
            if abs(a) > math.radians(30):
                return 0.0, self._turn * self.w_max * 0.7
            self._turn = 0.0
        curv = 2 * math.sin(a) / max(math.hypot(gx - x, gy - y), 1e-3)
        v = self.v_max / (1 + 2.0 * abs(curv))
        if not self.closed:
            v = min(v, 0.6 * np.linalg.norm(self.path[-1] - [x, y]) + 0.03)
        w = float(np.clip(v * curv, -self.w_max, self.w_max))
        return v, w
