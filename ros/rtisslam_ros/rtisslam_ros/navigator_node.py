"""rtis_navigator_node: goal -> Nav2 global path (planner_server on the RTISSLAM /map) -> path following with the
layered acoustic-flow controller (Jansen, Laurijssen, Steckel, Sensors 2022) -> cmd_vel.

A standalone node (decision 2026-10-12: no Nav2 controller plugin): Nav2 provides the global planner and costmap
(planner_server with a static layer on /map, allow_unknown false); this node replaces the controller_server.

Subscribes  /goal_pose (PoseStamped, e.g. RViz "2D Goal Pose"), /rtis/energyscape, TF map -> base_link
Publishes   cmd_vel, /rtis_nav/plan (the path being followed), /rtis_nav/layer (std_msgs/String: active layer)
Uses        action compute_path_to_pose (nav2_msgs/ComputePathToPose) of planner_server
Parameters  v_max, w_max, lookahead, goal_tol, replan_s, use_afc, afc_method (default dmas3cf:ams2s)
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from . import ROOT  # noqa: F401
from rtis_msgs.msg import Energyscape

from .common import msg_to_es, quat_to_yaw


class NavigatorNode(Node):
    def __init__(self):
        super().__init__("rtis_navigator")
        from rtisnav.afc import AcousticFlowController, AFCConfig
        from rtissim.robot import PurePursuit
        self.PP = PurePursuit
        self.v_max = float(self.declare_parameter("v_max", 0.3).value)
        self.w_max = float(self.declare_parameter("w_max", 0.8).value)
        self.look = float(self.declare_parameter("lookahead", 0.6).value)
        self.goal_tol = float(self.declare_parameter("goal_tol", 0.3).value)
        self.replan_s = float(self.declare_parameter("replan_s", 15.0).value)
        self.use_afc = bool(self.declare_parameter("use_afc", True).value)
        # energyscape the acoustic-flow controller works on ("imaging:post" key, as published by the simulator /
        # imaging node); DMAS3-CF by default, so that nothing learned (deconvolution) is in the loop
        self.afc_cfg = AFCConfig(method=str(self.declare_parameter("afc_method", "dmas3cf:ams2s").value))
        self.AFC = AcousticFlowController
        self.afc = None
        self.es = None
        self.goal = None
        self.tracker = None
        self.t_plan = None
        self.pending = False
        self.tfb = Buffer()
        self.tfl = TransformListener(self.tfb, self)
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.plan_pub = self.create_publisher(Path, "/rtis_nav/plan", 10)
        self.layer_pub = self.create_publisher(String, "/rtis_nav/layer", 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._goal, 10)
        self.create_subscription(Energyscape, "/rtis/energyscape", self._es, 10)
        self.ac = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.create_timer(0.1, self._tick)

    def _es(self, m: Energyscape):
        if m.method != self.afc_cfg.method:
            return
        r, az, es = msg_to_es(m)
        if self.afc is None:
            self.afc = self.AFC(self.afc_cfg, r, az)
        self.es = es

    def _goal(self, m: PoseStamped):
        self.goal = m
        self.tracker = None
        self._request_plan()

    def _request_plan(self):
        if not self.ac.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn("planner_server not available")
            return
        g = ComputePathToPose.Goal()
        g.goal, g.use_start = self.goal, False
        self.pending = True
        self.ac.send_goal_async(g).add_done_callback(self._accepted)

    def _accepted(self, fut):
        h = fut.result()
        if not h.accepted:
            self.pending = False
            return
        h.get_result_async().add_done_callback(self._planned)

    def _planned(self, fut):
        self.pending = False
        path = fut.result().result.path
        if not path.poses:
            self.get_logger().warn("no path to the goal")
            return
        P = np.array([[p.pose.position.x, p.pose.position.y] for p in path.poses])
        from rtissim.planner import resample
        P = resample(P, 0.05)
        self.tracker = self.PP(P, lookahead=self.look, v_max=self.v_max, w_max=self.w_max, closed=False)
        self.t_plan = self.get_clock().now()
        self.plan_pub.publish(path)

    def _pose(self):
        try:
            t = self.tfb.lookup_transform("map", "base_link", Time())
        except Exception:                                      # noqa: BLE001
            return None
        tr = t.transform
        return np.array([tr.translation.x, tr.translation.y, quat_to_yaw(tr.rotation)])

    def _tick(self):
        cmd = Twist()
        pose = self._pose()
        if self.goal is None or pose is None or self.tracker is None:
            self.cmd_pub.publish(cmd)
            return
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        if math.hypot(pose[0] - gx, pose[1] - gy) < self.goal_tol:
            self.get_logger().info("goal reached")
            self.goal, self.tracker = None, None
            self.cmd_pub.publish(cmd)
            return
        if not self.pending and (self.get_clock().now() - self.t_plan).nanoseconds * 1e-9 > self.replan_s:
            self._request_plan()
        v, w = self.tracker.command(pose)
        layer = "PATH"
        if self.use_afc and self.afc is not None and self.es is not None:
            v, w, layer = self.afc.command(self.es, v, w)
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.cmd_pub.publish(cmd)
        self.layer_pub.publish(String(data=layer))


def main():
    rclpy.init()
    n = NavigatorNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()
