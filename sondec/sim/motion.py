"""Inspection-style planar routes, pan-tilt head scanning, and the robot's own (noisy) sensors.

Route: a sequence of behaviours on the floor's free space (clearance >= ROBOT_R + margin):
  goto      A* to a random goal, shortcut-smoothed; turn on the spot at corners > 35 deg (smaller
            heading changes: arc while driving), straight runs
  scan      turn on the spot to +-(30..80) deg and back
  approach  drive towards the nearest object until 0.45 m clearance, then back off (reversing)
  sidestep  turn ~90 deg, drive 0.3-0.8 m, turn back (lateral offset)
Pulse spacing: 0.07-0.13 m of driving (median ~0.1 m), 8-15 deg per pulse when turning on the spot.

Head: smooth random scanning (two sinusoids per axis) plus occasional fast moves to a random target
(up to 10 deg per pulse) and holds. Rate-limited to 10 deg per pulse. Pan: periods 25-120 pulses.
Tilt (simulator v2): -20..+40 deg, periods 8-45 pulses, overdriven so that it sweeps the whole range
and dwells at the limits (v1: -30..+40 deg, periods 25-120, std 18 deg, ~1.7 deg per pulse).
Variants: "full" (pan + tilt), "pan" (tilt fixed at 0), "fixed" (pan = tilt = 0), "roll" (pan + tilt as
in "full" + roll -15..+15 deg, periods 15-60 pulses; returns [n, 3]).

Sensors (what training may use): wheel odometry increments (x, y in the body frame; unbiased, 10 %
noise of each step), gyro yaw increment (0.15 deg noise), servo pan/tilt (0.2 deg noise).
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..kinematics import PAN_LIMIT, ROLL_LIMIT, TILT_LIMIT

ROBOT_R = 0.25
CLEAR = ROBOT_R + 0.12


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Planner:
    def __init__(self, world, rng):
        occ, (self.x0, self.y0), self.res = world.occupancy(0.05)
        free = ~occ
        free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
        self.clear = distance_transform_edt(free) * self.res
        self.rng = rng
        # coarse grid for A*
        self.cs = 2
        c = self.clear[::self.cs, ::self.cs]
        self.cclear = c

    def clearance(self, x, y):
        i = int((x - self.x0) / self.res); j = int((y - self.y0) / self.res)
        if i < 0 or j < 0 or i >= self.clear.shape[0] or j >= self.clear.shape[1]:
            return 0.0
        return float(self.clear[i, j])

    def random_free(self, min_clear=0.55):
        idx = np.argwhere(self.clear >= min_clear)
        if len(idx) == 0:
            idx = np.argwhere(self.clear >= CLEAR)
        i, j = idx[self.rng.integers(len(idx))]
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res

    def segment_free(self, a, b, min_clear=CLEAR):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 0.03))
        return all(self.clearance(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) >= min_clear
                   for t in np.linspace(0, 1, n))

    def astar(self, start, goal):
        c, r = self.cclear, self.res * self.cs
        to_ij = lambda p: (int((p[0] - self.x0) / r), int((p[1] - self.y0) / r))  # noqa: E731
        s, g = to_ij(start), to_ij(goal)
        nx, ny = c.shape
        if not (0 <= g[0] < nx and 0 <= g[1] < ny) or c[g] < CLEAR:
            return None
        dist = {s: 0.0}; prev = {}; pq = [(0.0, 0.0, s)]
        while pq:
            _, d, u = heapq.heappop(pq)
            if u == g:
                break
            if d > dist.get(u, 1e18) + 1e-9:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == dj == 0:
                        continue
                    v = (u[0] + di, u[1] + dj)
                    if not (0 <= v[0] < nx and 0 <= v[1] < ny) or c[v] < CLEAR:
                        continue
                    step = math.hypot(di, dj) * r * (1 + 1.5 * math.exp(-(c[v] - CLEAR) / 0.3))
                    h = math.hypot(g[0] - v[0], g[1] - v[1]) * r
                    nd = d + step
                    if nd < dist.get(v, 1e18):
                        dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd + h, nd, v))
        if g not in prev and g != s:
            return None
        path, u = [], g
        while u != s:
            path.append(u); u = prev[u]
        path.append(s)
        pts = [(self.x0 + (i + 0.5) * r, self.y0 + (j + 0.5) * r) for i, j in path[::-1]]
        pts[0], pts[-1] = start, goal
        out = [pts[0]]; i = 0                                              # shortcut smoothing
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.segment_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j]); i = j
        return out


def plan_route(world, n_pulses: int, seed: int) -> np.ndarray:
    """-> [n_pulses, 3] ground-truth body poses (x, y, yaw)."""
    rng = np.random.default_rng(seed)
    P = Planner(world, rng)
    x, y = P.random_free(0.7)
    th = rng.uniform(-math.pi, math.pi)
    poses = [(x, y, th)]

    def push(p):
        poses.append(p)

    def turn_to(target):
        nonlocal th
        step = math.radians(rng.uniform(8, 15))
        d = _wrap(target - th)
        n = int(abs(d) / step)
        for _ in range(n):
            th = th + math.copysign(step, d); push((x, y, th))
        th = th + (d - math.copysign(step * n, d)) if n else target
        if abs(d) > 1e-6:
            push((x, y, th))

    def drive(dist, backwards=False, turn=0.0):
        """Straight run (optionally arcing through `turn` rad over the first steps)."""
        nonlocal x, y, th
        ds = rng.uniform(0.07, 0.13)
        n = max(1, int(round(abs(dist) / ds)))
        n_arc = min(n, max(1, int(abs(turn) / math.radians(8)))) if turn else 0
        sgn = -1 if backwards else 1
        for i in range(n):
            th_n = th + (turn / n_arc if i < n_arc else 0.0)
            hm = 0.5 * (th + th_n)
            nx_, ny_ = x + sgn * math.cos(hm) * dist / n, y + sgn * math.sin(hm) * dist / n
            if P.clearance(nx_, ny_) < CLEAR:
                return False
            x, y, th = nx_, ny_, th_n; push((x, y, th))
        return True

    while len(poses) < n_pulses:
        a = rng.random()
        if a < 0.7:                                                           # goto
            for _ in range(30):
                gx, gy = P.random_free(0.5)
                if 1.5 < math.hypot(gx - x, gy - y) < 7.0:
                    break
            path = P.astar((x, y), (gx, gy))
            if not path:
                continue
            for (ax_, ay_), (bx_, by_) in zip(path[:-1], path[1:]):
                L = math.hypot(bx_ - ax_, by_ - ay_)
                if L < 0.05:
                    continue
                hd = math.atan2(by_ - ay_, bx_ - ax_)
                d = _wrap(hd - th)
                if abs(d) > math.radians(35):
                    turn_to(hd); d = 0.0                                      # turn on the spot
                if not drive(L, turn=d):                                      # small changes: arc while driving
                    break
        elif a < 0.78:                                                        # scan
            h0 = th; amp = math.radians(rng.uniform(30, 80))
            turn_to(h0 + amp); turn_to(h0 - amp); turn_to(h0)
        elif a < 0.9:                                                         # approach
            best = None
            for ang in np.linspace(-math.pi, math.pi, 36, endpoint=False):
                for d in np.arange(0.3, 3.0, 0.1):
                    if P.clearance(x + d * math.cos(ang), y + d * math.sin(ang)) < CLEAR + 0.08:
                        if best is None or d < best[1]:
                            best = (ang, d)
                        break
            if best is None:
                continue
            turn_to(best[0])
            drive(max(0.0, best[1] - 0.1))
            drive(rng.uniform(0.3, 0.8), backwards=True)
        else:                                                                 # sidestep
            h0 = th; s = rng.choice([-1, 1])
            turn_to(h0 + s * math.pi / 2); drive(rng.uniform(0.3, 0.8)); turn_to(h0)
    return np.asarray(poses[:n_pulses], float)


def head_motion(n: int, seed: int, variant: str = "full") -> np.ndarray:
    """-> [n, 2] true (pan, tilt) in rad."""
    if variant == "fixed":
        return np.zeros((n, 2))
    if variant == "roll":                                                      # the "full" head + an independent roll
        rr = np.random.default_rng(seed + 4242)
        t = np.arange(n); lo, hi = ROLL_LIMIT; half = (hi - lo) / 2
        roll = np.zeros(n)
        for amp, (p0, p1) in ((0.8, (25, 60)), (0.4, (10, 20))):
            roll += amp * half * np.sin(2 * math.pi * t / rr.uniform(p0, p1) + rr.uniform(0, 2 * math.pi))
        roll = np.clip(roll + (lo + hi) / 2, lo, hi)
        roll = np.clip(roll, np.r_[roll[0], roll[:-1]] - 6.0, np.r_[roll[0], roll[:-1]] + 6.0)
        return np.concatenate([head_motion(n, seed, "full"), np.radians(roll)[:, None]], 1)
    rng = np.random.default_rng(seed + 7777)
    lims = [PAN_LIMIT, TILT_LIMIT]
    t = np.arange(n)
    base = np.zeros((n, 2))
    # (amplitude as a fraction of the half range, period range in pulses) per component
    comps = [((0.65, (60, 120)), (0.3, (25, 50))),                          # pan (as in simulator v1)
             ((0.9, (12, 30)), (0.5, (6, 12)))]                             # tilt, v2: faster, overdriven (clipped)
    for ax, (lo, hi) in enumerate(lims):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        for amp, (p0, p1) in comps[ax]:
            base[:, ax] += amp * half * np.sin(2 * math.pi * t / rng.uniform(p0, p1) + rng.uniform(0, 2 * math.pi))
        base[:, ax] += mid
    out = np.zeros((n, 2)); cur = base[0].copy(); hold = 0; target = None; speed = 5.0
    for i in range(n):
        if target is None and rng.random() < 0.05:                             # occasional fast move (v1: 0.03)
            target = np.array([rng.uniform(*lims[0]), rng.uniform(*lims[1])]); speed = rng.uniform(6, 10)
            hold = rng.integers(3, 15) if rng.random() < 0.3 else rng.integers(1, 4)   # v2: mostly short holds
        goal = target if target is not None else base[i]
        d = goal - cur
        lim = speed if target is not None else 10.0
        nd = np.linalg.norm(d)
        if nd > lim:
            d = d * lim / nd
        cur = cur + d
        if target is not None and nd <= lim:
            hold -= 1
            if hold <= 0:
                target = None
        cur = np.array([np.clip(cur[0], *lims[0]), np.clip(cur[1], *lims[1])])
        out[i] = cur
    if variant == "pan":
        out[:, 1] = 0.0
    return np.radians(out)


def sensors(gt_pose: np.ndarray, head: np.ndarray, seed: int) -> dict:
    """The robot's own measurements (the ONLY motion information training may use)."""
    rng = np.random.default_rng(seed + 31337)
    n = len(gt_pose)
    d = np.zeros((n, 3))                                                       # increment i -> i+1 in body frame i
    dx, dy = gt_pose[1:, 0] - gt_pose[:-1, 0], gt_pose[1:, 1] - gt_pose[:-1, 1]
    c, s = np.cos(gt_pose[:-1, 2]), np.sin(gt_pose[:-1, 2])
    d[:-1, 0] = c * dx + s * dy
    d[:-1, 1] = -s * dx + c * dy
    d[:-1, 2] = np.angle(np.exp(1j * (gt_pose[1:, 2] - gt_pose[:-1, 2])))
    odo = d[:, :2] * (1 + 0.10 * rng.standard_normal((n, 2))) + 0.001 * rng.standard_normal((n, 2))
    gyro = d[:, 2] + math.radians(0.15) * rng.standard_normal(n)
    servo = head + math.radians(0.2) * rng.standard_normal(head.shape)
    return {"gt_delta": d, "odo_xy": odo, "gyro_yaw": gyro, "servo": servo}
