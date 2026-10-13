"""Closed-loop simulator: a differential-drive robot with an eRTIS in a floor plan, stepped at the control rate.

    sim = Simulator("office", device="cuda")
    obs = sim.step(v, w)          # integrates dt; every 1/pulse_rate s also fires the sonar
    obs["odom"], obs["gyro"], obs.get("maps")       # what the robot knows
    obs["gt"]                                         # ground truth (EVALUATION ONLY)

The same interface is mirrored by the ROS 2 node rtis_sim (cmd_vel in; odom, TF, energyscapes out).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from .robot import DiffDrive, OdomNoise
from .sensor import METHODS, RtisSensor
from .worlds import PLANS, build_world, driving_line


class Simulator:
    def __init__(self, plan: str, device: str = "cuda", methods=METHODS, dt: float = 0.1, pulse_rate: float = 10.0,
                 noise: OdomNoise | None = None, seed: int = 0, start=None):
        self.plan = PLANS[plan]()
        self.world = build_world(self.plan, seed)
        self.arrays = self.world.arrays()
        self.sensor = RtisSensor(self.arrays, device=device, methods=methods, seed=seed)
        self.dt, self.pulse_every = dt, max(1, int(round(1.0 / (pulse_rate * dt))))
        self._line = None
        if start is None:
            L = self.line
            start = (L[0, 0], L[0, 1], math.atan2(L[1, 1] - L[0, 1], L[1, 0] - L[0, 0]))
        self.robot = DiffDrive(pose=np.asarray(start, float), noise=noise or OdomNoise(), seed=seed + 1)
        self.k = 0
        self.t = 0.0
        occ, self._org, self._res = self.world.occupancy(0.02)
        self._clear = distance_transform_edt(~occ) * self._res          # ground-truth clearance (collisions)
        self.r_robot = 0.25
        self.collisions = 0

    def clearance(self, x, y) -> float:
        i, j = int((x - self._org[0]) / self._res), int((y - self._org[1]) / self._res)
        if not (0 <= i < self._clear.shape[0] and 0 <= j < self._clear.shape[1]):
            return 0.0
        return float(self._clear[i, j])

    @property
    def line(self) -> np.ndarray:
        if self._line is None:
            self._line = driving_line(self.plan, self.world)
        return self._line

    def occupancy(self, res: float = 0.05):
        return self.world.occupancy(res)

    def step(self, v: float, w: float, fire: bool | None = None) -> dict:
        prev = (self.robot.pose.copy(), self.robot.odom.copy())
        gyro = self.robot.step(v, w, self.dt)
        bump = self.clearance(*self.robot.pose[:2]) < self.r_robot and             self.clearance(*self.robot.pose[:2]) < self.clearance(*prev[0][:2])
        if bump:                                   # the robot touches an obstacle: it stalls (pose and odometry)
            self.robot.pose, self.robot.odom = prev
            self.collisions += 1
        self.k += 1
        self.t += self.dt
        obs = {"t": self.t, "k": self.k, "odom": self.robot.odom.copy(), "gyro": gyro, "gt": self.robot.pose.copy(),
               "cmd": (v, w), "bump": bump}
        if fire if fire is not None else self.k % self.pulse_every == 0:
            maps, contrib = self.sensor.render(self.robot.pose)
            obs["maps"], obs["contrib"] = maps, contrib
        return obs
