"""rtis_sim_node: the closed-loop simulator (rtissim.Simulator) as a ROS 2 node; it owns the simulation clock.

Subscribes  cmd_vel (geometry_msgs/Twist)
Publishes   /clock, odom (nav_msgs/Odometry, odom -> base_link) + TF, /rtis/energyscape (rtis_msgs/Energyscape, one
            message per method and pulse, frame rtis_link), static TF base_link -> rtis_link,
            /ground_truth/odom (EVALUATION ONLY), /world_markers (walls and objects, latched)
Parameters  world (office), device (cpu), methods ([deconv]; "imaging:post" keys apply RW-CFAR, e.g. dmas3cf:ams2s), dt (0.1 s), real_time_factor (0 = as fast as possible),
            start ([] = start of the plan's driving line), seed,
            output (energyscape | pulse): pulse publishes the raw microphone signals as rtis_msgs/RtisPulse on /rtis/pulse
            instead of energyscapes, so that rtis_imaging_node does the imaging exactly as on the real robot
All other nodes run with use_sim_time = true: on a CPU (WSL) a pulse takes longer than 0.1 s, so the closed loop runs
slower than real time but correctly timed.
"""
from __future__ import annotations

import time

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from . import ROOT  # noqa: F401  (puts the project on sys.path)
from rtis_msgs.msg import Energyscape, RtisPulse

from .common import es_to_msg, parse_pose, transform, yaw_to_quat


class SimNode(Node):
    def __init__(self):
        super().__init__("rtis_sim")
        from rtissim.sensor import MOUNT
        from rtissim.sim import Simulator
        world = self.declare_parameter("world", "office").value
        device = self.declare_parameter("device", "cpu").value
        from rtisslam.imaging_post import POST
        keys = list(self.declare_parameter("methods", ["deconv"]).value)
        self.split = {k: (k.split(":") + ["none"])[:2] for k in keys}
        self.post = POST
        methods = sorted({im for im, _ in self.split.values()})
        dt = float(self.declare_parameter("dt", 0.1).value)
        self.rtf = float(self.declare_parameter("real_time_factor", 0.0).value)
        start = parse_pose(self.declare_parameter("start", "0,0,0").value)
        use_start = bool(self.declare_parameter("use_start", False).value)
        seed = int(self.declare_parameter("seed", 0).value)
        self.output = self.declare_parameter("output", "energyscape").value
        if self.output == "pulse":
            methods = ["das"]                          # the sensor only synthesizes; imaging is rtis_imaging_node's
        self.sim = Simulator(world, device=device, methods=methods, dt=dt, seed=seed, start=start if use_start else None)
        self.methods = methods
        self.cmd = (0.0, 0.0)
        self.t_cmd = 0.0
        self.create_subscription(Twist, "cmd_vel", self._cmd, 10)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.gt_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self.es_pub = self.create_publisher(Energyscape, "/rtis/energyscape", 10)
        self.pulse_pub = self.create_publisher(RtisPulse, "/rtis/pulse", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mk_pub = self.create_publisher(MarkerArray, "/world_markers", latched)
        self.tf = TransformBroadcaster(self)
        self.stf = StaticTransformBroadcaster(self)
        self.stf.sendTransform(transform(self._stamp(), "base_link", "rtis_link", MOUNT[0], MOUNT[1], 0.0))
        self._markers()
        self.timer = self.create_timer(0.001, self._tick)
        self.get_logger().info(f"rtis_sim: world {world}, methods {methods}, device {device}")

    def _stamp(self) -> Time:
        t = Time()
        ns = int(round(self.sim.t * 1e9))  # round once: t = 95.9999999 must give 96 s, not 95 s + 0 ns
        t.sec, t.nanosec = divmod(ns, 1_000_000_000)
        return t

    def _cmd(self, m: Twist):
        self.cmd = (float(m.linear.x), float(m.angular.z))
        self.t_cmd = self.sim.t

    def _markers(self):
        arr = MarkerArray()
        mk = Marker()
        mk.header.frame_id = "map"
        mk.ns, mk.id, mk.type, mk.action = "walls", 0, Marker.LINE_LIST, Marker.ADD
        mk.scale.x = 0.1
        mk.color.r = mk.color.g = mk.color.b = 0.6
        mk.color.a = 1.0
        from geometry_msgs.msg import Point
        for w in self.sim.plan.walls:
            for x, y in ((w[0], w[1]), (w[2], w[3])):
                p = Point(); p.x, p.y = float(x), float(y); mk.points.append(p)
        arr.markers.append(mk)
        self.mk_pub.publish(arr)

    def _tick(self):
        t0 = time.perf_counter()
        v, w = self.cmd if self.sim.t - self.t_cmd < 0.5 else (0.0, 0.0)
        if self.output == "pulse":
            obs = self.sim.step(v, w, fire=False)
            fire = self.sim.k % self.sim.pulse_every == 0
        else:
            obs = self.sim.step(v, w)
            fire = False
        st = self._stamp()
        c = Clock(); c.clock = st
        self.clock_pub.publish(c)
        x, y, th = obs["odom"]
        self.tf.sendTransform(transform(st, "odom", "base_link", x, y, th))
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = st, "odom", "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y = float(x), float(y)
        o.pose.pose.orientation = yaw_to_quat(th)
        o.twist.twist.linear.x, o.twist.twist.angular.z = v, obs["gyro"]
        self.odom_pub.publish(o)
        g = Odometry()
        g.header.stamp, g.header.frame_id, g.child_frame_id = st, "map", "ground_truth"
        g.pose.pose.position.x, g.pose.pose.position.y = float(obs["gt"][0]), float(obs["gt"][1])
        g.pose.pose.orientation = yaw_to_quat(obs["gt"][2])
        self.gt_pub.publish(g)
        if fire:
            from sondec.sensor import FS
            x, _ = self.sim.sensor.pulse(self.sim.robot.pose)
            x = x.float().cpu().numpy()
            p = RtisPulse()
            p.header.stamp, p.header.frame_id = st, "rtis_link"
            p.fs, p.n_mics, p.n_samples, p.array_id = float(FS), int(x.shape[0]), int(x.shape[1]), "ertis_sim"
            p.data = x.ravel().tolist()
            self.pulse_pub.publish(p)
        if "maps" in obs:
            s = self.sim.sensor
            for k, (im, post) in self.split.items():
                self.es_pub.publish(es_to_msg(st, "rtis_link", k, s.ranges, s.azimuths, self.post[post](obs["maps"][im])))
        if self.rtf > 0:
            time.sleep(max(0.0, self.sim.dt / self.rtf - (time.perf_counter() - t0)))


def main():
    rclpy.init()
    n = SimNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()
