"""Navigator: Nav2-style global planning on the RTISSLAM occupancy grid + the layered acoustic-flow controller.

    nav = Navigator(slam)                    # slam: RTISSLAM loaded from a map, in localization mode
    nav.set_goal((x, y))
    v, w = nav.step(odom, maps)              # every control step; maps = {method: energyscape} on pulse steps
    nav.pose, nav.path, nav.state            # estimate (map frame), current global path, 'driving' | 'reached' | 'failed'

Global costmap: the RTISSLAM grid rendered once from the map's node poses; OCCUPIED and UNKNOWN cells are blocked
(Nav2 allow_unknown = false): specular walls leave holes in the occupied evidence, so only space that was observed free
is planned through, plus the space the robot swept while mapping (its footprint is free, unknown cells within
SWEPT_UNKNOWN of the mapping path are traversable). Replanning when the robot is more than REPLAN_DEV m off the path or every REPLAN_S seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from batslam.backend.pose_graph import pose2
from rtissim.planner import GridPlanner
from rtissim.robot import PurePursuit

from .afc import AcousticFlowController, AFCConfig
from .costmap import CostmapConfig, planning_grid


@dataclass
class NavConfig:
    v_max: float = 0.3
    w_max: float = 0.8
    lookahead: float = 0.6
    goal_tol: float = 0.3
    replan_dev: float = 0.8
    replan_s: float = 15.0
    use_afc: bool = True
    stuck_s: float = 20.0          # no progress (STUCK_M closer to the goal) for this long -> recovery
    stuck_m: float = 0.3
    recovery_s: float = 3.0        # back up at RECOVERY_V while turning slightly, then replan (Nav2-style recovery)
    recovery_v: float = -0.1
    costmap: CostmapConfig = field(default_factory=CostmapConfig)
    afc: AFCConfig = field(default_factory=AFCConfig)


class Navigator:
    def __init__(self, slam, cfg: NavConfig | None = None):
        self.cfg = cfg or NavConfig()
        self.slam = slam
        self.tern, L, org, res = planning_grid(slam, self.cfg.costmap)
        self.L, self.org, self.res = L, org, res
        blocked = self.tern != 0
        self.blocked = blocked
        self.planner = GridPlanner(blocked, org, res)
        self.afc = AcousticFlowController(self.cfg.afc, slam.ranges, slam.azimuths)
        self.goal = None
        self.path = None
        self.tracker = None
        self.state = "idle"
        self.pose = None
        self.layer = "PATH"
        self._t_plan = -1e9
        self._last_es = None
        self.t = 0.0
        self._best = np.inf
        self._t_best = 0.0
        self._recover_until = -1.0
        self.n_recoveries = 0
        self._recovery_cmd = (0.0, 0.0)

    def set_goal(self, goal):
        self.goal = np.asarray(goal, float)
        self.state = "driving"
        self.path = None
        self._best, self._t_best = np.inf, self.t

    def _plan(self):
        P = self.planner.plan(self.pose[:2], self.goal)
        if P is None:
            self.state = "failed"
            self.path, self.tracker = None, None
            return
        self.path = P
        self.tracker = PurePursuit(P, lookahead=self.cfg.lookahead, v_max=self.cfg.v_max, w_max=self.cfg.w_max, closed=False)
        self._t_plan = self.t

    def step(self, odom, maps=None, dt: float = 0.1, oracle=None) -> tuple[float, float]:
        """oracle: a pose to use instead of the RTISSLAM estimate (diagnostics: ground-truth localization)."""
        self.t += dt
        if maps is not None:
            if oracle is None:
                self.slam.process(maps, odom)
            self._last_es = maps[self.cfg.afc.method]
        if oracle is not None:
            self.pose = np.asarray(oracle, float).copy()
        else:
            T = pose2(self.slam.map_to_odom()).compose(pose2(odom))
            self.pose = np.array([T.x(), T.y(), T.theta()])
        if self.state != "driving":
            return 0.0, 0.0
        if np.hypot(*(self.pose[:2] - self.goal)) < self.cfg.goal_tol:
            self.state = "reached"
            return 0.0, 0.0
        dg = float(np.hypot(*(self.pose[:2] - self.goal)))
        if dg < self._best - self.cfg.stuck_m:
            self._best, self._t_best = dg, self.t
        if self.t < self._recover_until:
            self.layer = "RECOVERY"
            return self._recovery_cmd
        if self.t - self._t_best > self.cfg.stuck_s:
            # recovery: rotate in place (a round robot cannot hit anything while turning; the sonar does not see
            # behind the robot, so reversing is blind), then replan; only every third attempt backs up briefly
            self.n_recoveries += 1
            if self.n_recoveries % 3 == 0:
                self._recovery_cmd = (self.cfg.recovery_v, 0.0)
                dur = 1.5
            else:
                self._recovery_cmd = (0.0, 0.6 if self.n_recoveries % 2 else -0.6)
                dur = self.cfg.recovery_s
            self._recover_until = self.t + dur
            self._t_best = self.t + dur
            self.path = None
            self.layer = "RECOVERY"
            return self._recovery_cmd
        if self.path is None or self.t - self._t_plan > self.cfg.replan_s or \
                np.min(np.hypot(*(self.path - self.pose[:2]).T)) > self.cfg.replan_dev:
            self._plan()
            if self.path is None:
                return 0.0, 0.0
        v, w = self.tracker.command(self.pose)
        self.layer = "PATH"
        if self.cfg.use_afc and self._last_es is not None:
            v, w, self.layer = self.afc.command(self._last_es, v, w)
        return v, w
