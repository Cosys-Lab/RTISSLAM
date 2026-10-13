"""rtisslam_node: RTISSLAM (BatSLAM 2.0 pose graph on eRTIS energyscapes + acoustic odometry + occupancy grid).

Subscribes  /rtis/energyscape (rtis_msgs/Energyscape; one per "imaging:post" key the map uses, same stamp), odom
            (nav_msgs/Odometry: the wheel odometry pose in the odom frame)
Publishes   TF map -> odom (replaces AMCL), /rtisslam/pose (PoseWithCovarianceStamped), /rtisslam/path,
            /rtisslam/hypotheses (geometry_msgs/PoseArray: the mixture's hypotheses or a sample of the particles),
            /rtisslam/candidates (visualization_msgs/MarkerArray: implied poses of this pulse's recognitions, colour =
            P(correct), green = near the estimate), /rtisslam/status (std_msgs/String, JSON per processed pulse: what
            the localizer is doing: candidates, best similarity, hypotheses / mode mass, sigma, ms per pulse),
            /map (nav_msgs/OccupancyGrid, transient local: the planning grid for Nav2's static layer),
            /rtisslam/grid (the raw RTISSLAM log-odds grid as OccupancyGrid 0..100), /rtisslam/graph (links, markers)
Services    ~/save_map (std_srvs/Trigger): writes the map to `save_file`
Parameters  mode (mapping | localization), map_file (localization), save_file, start ([x, y, yaw]), start_sigma,
            localizer (localization: graph | mix | pf, see rtisslam/localize.py),
            grid_period (s, mapping: how often the grid is re-rendered and published)
"""
from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from . import ROOT  # noqa: F401
from rtis_msgs.msg import Energyscape

from .common import msg_to_es, parse_pose, quat_to_yaw, transform, yaw_to_quat


def grid_msg(stamp, values: np.ndarray, org, res) -> OccupancyGrid:
    g = OccupancyGrid()
    g.header.stamp, g.header.frame_id = stamp, "map"
    g.info.resolution = float(res)
    g.info.width, g.info.height = int(values.shape[0]), int(values.shape[1])
    g.info.origin.position.x, g.info.origin.position.y = float(org[0]), float(org[1])
    g.info.origin.orientation.w = 1.0
    g.data = np.asarray(values, np.int8).T.ravel().tolist()            # row-major in y
    return g


class SlamNode(Node):
    def __init__(self):
        super().__init__("rtisslam")
        from rtisslam.system import RTISSLAM, RTISSLAMConfig
        self.mode = self.declare_parameter("mode", "mapping").value
        self.save_file = self.declare_parameter("save_file", "rtisslam_map.npz").value
        start = parse_pose(self.declare_parameter("start", "0,0,0").value)
        sigma = list(self.declare_parameter("start_sigma", [0.3, 0.3, 0.1]).value)
        self.grid_period = float(self.declare_parameter("grid_period", 5.0).value)
        self.localizer = self.declare_parameter("localizer", "graph").value
        self.slam = None
        self._ctor = (RTISSLAM, RTISSLAMConfig)
        if self.mode == "localization":
            self.slam = RTISSLAM.load(self.declare_parameter("map_file", "").value)
            self.slam.start_localization(start, tuple(sigma), localizer=self.localizer)
            self.get_logger().info(f"localization in {self.slam.n_map_nodes} map nodes, localizer {self.localizer}")
        self.start = start
        self.odom = None
        self.latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", self.latched)
        self.raw_pub = self.create_publisher(OccupancyGrid, "/rtisslam/grid", self.latched)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/rtisslam/pose", 10)
        self.path_pub = self.create_publisher(Path, "/rtisslam/path", 10)
        self.graph_pub = self.create_publisher(Marker, "/rtisslam/graph", 10)
        self.hyp_pub = self.create_publisher(PoseArray, "/rtisslam/hypotheses", 10)
        self.cand_pub = self.create_publisher(MarkerArray, "/rtisslam/candidates", 10)
        self.status_pub = self.create_publisher(String, "/rtisslam/status", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, "odom", self._odom, 50)
        self.create_subscription(Energyscape, "/rtis/energyscape", self._es, 10)
        self.create_service(Trigger, "~/save_map", self._save)
        self.create_timer(0.05, self._tf_tick)
        self.create_timer(self.grid_period, self._grid_tick)
        self._T = np.zeros(3)
        self._last_pm = None
        if self.slam is not None:
            self._publish_grid()

    def _odom(self, m: Odometry):
        p = m.pose.pose
        self.odom = np.array([p.position.x, p.position.y, quat_to_yaw(p.orientation)])

    def _es(self, m: Energyscape):
        RTISSLAM, RTISSLAMConfig = self._ctor
        r, az, es = msg_to_es(m)
        if self.slam is None:
            self.slam = RTISSLAM(RTISSLAMConfig(), r, az, initial_pose=tuple(self.start))
        # one message per energyscape key ("imaging:post"); process a pulse once all keys the map uses are in
        need = self.slam.cfg.energyscape_keys()
        stamp = (m.header.stamp.sec, m.header.stamp.nanosec)
        if getattr(self, "_stamp", None) != stamp:
            self._stamp, self._maps = stamp, {}
        self._maps[m.method] = es
        if not need <= self._maps.keys() or self.odom is None:
            return
        t0 = time.perf_counter()
        log = self.slam.process({k: self._maps[k] for k in need}, self.odom)
        ms = 1e3 * (time.perf_counter() - t0)
        self._maps = {}
        st = m.header.stamp
        if log is None:
            # gated (the robot moved less than gate_m / gate_rad since the last update): no update, but the estimate
            # (map -> odom composed with the current odometry) and the status are still published every pulse, so a
            # standing robot does not look like a dead localizer
            if self._last_pm is not None:
                T = self._T
                c, s_ = math.cos(T[2]), math.sin(T[2])
                ox, oy, oth = self.odom
                self._last_pm.header.stamp = st
                self._last_pm.pose.pose.position.x = float(T[0] + c * ox - s_ * oy)
                self._last_pm.pose.pose.position.y = float(T[1] + s_ * ox + c * oy)
                self._last_pm.pose.pose.orientation = yaw_to_quat(T[2] + oth)
                self.pose_pub.publish(self._last_pm)
                self._publish_status(None, ms, self._last_pm)
            return
        self._T = self.slam.map_to_odom()
        x, y, th = self.slam.pose()
        pm = PoseWithCovarianceStamped()
        pm.header.stamp, pm.header.frame_id = st, "map"
        pm.pose.pose.position.x, pm.pose.pose.position.y = float(x), float(y)
        pm.pose.pose.orientation = yaw_to_quat(th)
        try:
            C = self.slam.covariance()
            cov = np.zeros((6, 6)); cov[:2, :2] = C[:2, :2]; cov[5, 5] = C[2, 2]; cov[:2, 5] = C[:2, 2]; cov[5, :2] = C[2, :2]
            pm.pose.covariance = cov.ravel().tolist()
        except Exception:                                       # noqa: BLE001 (marginals can fail right after a reset)
            pass
        self.pose_pub.publish(pm)
        self._last_pm = pm
        if self.slam.localizer is not None:
            self._publish_hypotheses(st)
            self._publish_candidates(st)
        self._publish_status(log, ms, pm)

    def _tf_tick(self):
        if self.slam is None:
            return
        self.tf.sendTransform(transform(self.get_clock().now().to_msg(), "map", "odom", *self._T))

    def _grid_tick(self):
        if self.slam is None or self.slam.n < 2:
            return
        if self.mode == "mapping":
            self._publish_grid()
        st = self.get_clock().now().to_msg()
        P = self.slam.poses()
        path = Path(); path.header.stamp, path.header.frame_id = st, "map"
        if self.slam.localizer is not None:
            trail = np.asarray(self.slam.track).reshape(-1, 3)
        elif self.mode == "localization":
            trail = P[self.slam.n_map_nodes:]
        else:
            trail = P
        for x, y, th in trail:
            ps = PoseStamped(); ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y); ps.pose.orientation = yaw_to_quat(th)
            path.poses.append(ps)
        self.path_pub.publish(path)
        mk = Marker(); mk.header = path.header; mk.ns, mk.type = "links", Marker.LINE_LIST
        mk.scale.x = 0.03; mk.color.r, mk.color.g, mk.color.a = 1.0, 0.55, 0.8
        from geometry_msgs.msg import Point
        for q, a in self.slam.links():
            for k in (q, a):
                p = Point(); p.x, p.y = float(P[k, 0]), float(P[k, 1]); mk.points.append(p)
        self.graph_pub.publish(mk)

    def _publish_hypotheses(self, stamp):
        from rtisslam.localize import ParticleLocalizer
        loc = self.slam.localizer
        if isinstance(loc, ParticleLocalizer):
            k = np.random.default_rng(0).choice(len(loc.X), min(300, len(loc.X)), replace=False, p=loc.w)
            X = loc.X[k]
        else:
            X = np.array([h.mu for h in loc.H]).reshape(-1, 3)
        pa = PoseArray()
        pa.header.stamp, pa.header.frame_id = stamp, "map"
        for x, y, th in X:
            p = Pose(); p.position.x, p.position.y = float(x), float(y); p.orientation = yaw_to_quat(th)
            pa.poses.append(p)
        self.hyp_pub.publish(pa)

    def _publish_candidates(self, stamp):
        cands = getattr(self.slam._rec, "last", [])
        x0 = np.asarray(self.slam.pose())
        arr = MarkerArray()
        clear = Marker(); clear.header.frame_id = "map"; clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, c in enumerate(cands):
            mk = Marker()
            mk.header.stamp, mk.header.frame_id = stamp, "map"
            mk.ns, mk.id, mk.type, mk.action = "candidates", i, Marker.ARROW, Marker.ADD
            mk.pose.position.x, mk.pose.position.y = float(c.z[0]), float(c.z[1])
            mk.pose.orientation = yaw_to_quat(float(c.z[2]))
            mk.scale.x, mk.scale.y, mk.scale.z = 0.35, 0.06, 0.06
            near = np.hypot(*(c.z[:2] - x0[:2])) < 1.0
            mk.color.r, mk.color.g, mk.color.b = (0.1, 0.9, 0.2) if near else (1.0, 0.2, 0.2)
            mk.color.a = float(0.25 + 0.75 * c.p)
            arr.markers.append(mk)
        self.cand_pub.publish(arr)

    def _publish_status(self, log, ms, pm):
        import json
        C = np.array(pm.pose.covariance).reshape(6, 6)
        d = {"t": pm.header.stamp.sec + 1e-9 * pm.header.stamp.nanosec, "localizer": self.localizer if self.mode == "localization" else "mapping",
             "ms": round(ms, 1), "gated": log is None, "x": round(pm.pose.pose.position.x, 3), "y": round(pm.pose.pose.position.y, 3),
             "sigma_xy": round(float(np.sqrt(max(C[0, 0] + C[1, 1], 0))), 3),
             "sigma_th_deg": round(float(np.degrees(np.sqrt(max(C[5, 5], 0)))), 2),
             "n_candidates": int(getattr(log, "n_candidates", 0)),
             "best_similarity": None if not np.isfinite(getattr(log, "best_similarity", np.nan)) else round(float(log.best_similarity), 3)}
        loc = self.slam.localizer
        if log is None:
            pass
        elif loc is not None:
            cands = getattr(self.slam._rec, "last", [])
            x0 = np.asarray(self.slam.pose())
            d["n_near"] = int(sum(np.hypot(*(c.z[:2] - x0[:2])) < 1.0 for c in cands))
            d["best_p"] = round(max((c.p for c in cands), default=0.0), 3)
            d["mass"] = round(float(log.mass), 3)
            if hasattr(loc, "H"):
                d["hypotheses"] = [[round(float(h.w), 3), round(float(h.mu[0]), 2), round(float(h.mu[1]), 2)] for h in
                                   sorted(loc.H, key=lambda h: -h.w)[:5]]
                d["switches"], d["spawned"] = loc.n_switches, loc.n_spawned
            else:
                d["injected"], d["switches"] = loc.n_injected, loc.n_switches
            d["state"] = "tracking" if d["mass"] > 0.5 and d["sigma_xy"] < 0.5 else "uncertain"
        else:
            d["n_alive"], d["locked"], d["links"] = int(getattr(log, "n_alive", 0)), bool(getattr(log, "locked", False)), len(self.slam.slam.graph.loops)
        self.status_pub.publish(String(data=json.dumps(d)))

    def _publish_grid(self):
        from rtisnav.costmap import planning_grid
        st = self.get_clock().now().to_msg()
        tern, L, org, res = planning_grid(self.slam)
        self.map_pub.publish(grid_msg(st, tern, org, res))
        raw = np.round(100 / (1 + np.exp(-L))).astype(np.int8)
        self.raw_pub.publish(grid_msg(st, raw, org, res))

    def _save(self, req, resp):
        if self.slam is None:
            resp.success, resp.message = False, "no map yet"
            return resp
        self.slam.save(self.save_file)
        resp.success, resp.message = True, f"saved {self.slam.n} nodes to {self.save_file}"
        return resp


def main():
    rclpy.init()
    n = SlamNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()
