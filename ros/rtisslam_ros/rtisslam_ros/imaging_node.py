"""rtis_imaging_node: raw eRTIS pulses -> 2D energyscapes (for the real sensor; the simulator publishes energyscapes
directly). Same imaging code as the simulator (sondec DAS / DMAS, the SPL deconvolution network on DAS).

Subscribes  /rtis/pulse (rtis_msgs/RtisPulse)
Publishes   /rtis/energyscape (rtis_msgs/Energyscape), one message per method
Parameters  methods ([deconv]; "imaging:post" applies RW-CFAR post-processing, e.g. dmas3cf:ams2s), device (cpu), ckpt ("" = RTISSLAM_DATA/models/spl_main_s0.pt)

The real eRTIS ROS 2 driver publishes its own message type: a small adapter (to be written against that driver)
converts it to RtisPulse (demodulated signals, 450 kHz, 32 microphones, array_id of the calibrated geometry).
"""
from __future__ import annotations

import numpy as np
import rclpy
import torch
from rclpy.node import Node

from . import ROOT  # noqa: F401
from rtis_msgs.msg import Energyscape, RtisPulse

from .common import es_to_msg


class ImagingNode(Node):
    def __init__(self):
        super().__init__("rtis_imaging")
        from rtissim.sensor import RtisSensor
        from rtisslam.imaging_post import POST
        self.methods = list(self.declare_parameter("methods", ["deconv"]).value)
        self.split = {k: (k.split(":") + ["none"])[:2] for k in self.methods}
        self.post = POST
        imagings = sorted({im for im, _ in self.split.values()})
        device = self.declare_parameter("device", "cpu").value
        ckpt = self.declare_parameter("ckpt", "").value or None
        empty = {"pos": np.zeros((0, 3), np.float32), "normal": np.zeros((0, 3), np.float32), "area": np.zeros(0, np.float32),
                 "mat": np.zeros((0, 3), np.float32), "obj": np.zeros(0, np.int32), "two": np.zeros(0, bool),
                 "obj_size": np.zeros(0, np.float32), "obj_tilt": np.zeros(0, np.float32), "obj_name": np.array([]),
                 "boxes": np.zeros((0, 7), np.float32), "capsules": np.zeros((0, 7), np.float32)}
        self.sensor = RtisSensor(empty, device=device, methods=imagings, ckpt=ckpt)   # imaging only
        self.dev = device
        self.pub = self.create_publisher(Energyscape, "/rtis/energyscape", 10)
        self.create_subscription(RtisPulse, "/rtis/pulse", self._pulse, 10)

    def _pulse(self, m: RtisPulse):
        x = torch.as_tensor(np.asarray(m.data, np.float32).reshape(m.n_mics, m.n_samples), device=self.dev)
        maps = self.sensor.image(x)
        for k, (im, post) in self.split.items():
            es = self.post[post](maps[im])
            self.pub.publish(es_to_msg(m.header.stamp, m.header.frame_id, k, self.sensor.ranges, self.sensor.azimuths, es))


def main():
    rclpy.init()
    n = ImagingNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()
