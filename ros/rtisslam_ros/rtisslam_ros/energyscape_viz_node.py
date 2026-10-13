"""rtis_energyscape_viz_node: eRTIS energyscapes for RViz2.

Subscribes  /rtis/energyscape (rtis_msgs/Energyscape; only messages of parameter `method`)
Publishes   /rtis/energyscape_cloud (sensor_msgs/PointCloud2, in the sensor frame: one point per range-azimuth cell within
            `dynamic_range_db` of the strongest range-compensated cell, field `intensity` = level above the floor in dB;
            shows the echoes on top of the map) and /rtis/energyscape_image (sensor_msgs/Image, rgb8: the energyscape as
            a fan, forward up, left = left, for an RViz Image panel)
Parameters  method (deconv), dynamic_range_db (30), max_range (5.0 m), image_px (360)
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField

from . import ROOT  # noqa: F401
from rtis_msgs.msg import Energyscape

from .common import msg_to_es


def inferno(x: np.ndarray) -> np.ndarray:
    """x in [0, 1] -> uint8 RGB (matplotlib's inferno, sampled)."""
    from matplotlib import colormaps
    return (colormaps["inferno"](np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


class EnergyscapeViz(Node):
    def __init__(self):
        super().__init__("rtis_energyscape_viz")
        self.method = self.declare_parameter("method", "deconv").value
        self.dr = float(self.declare_parameter("dynamic_range_db", 30.0).value)
        self.rmax = float(self.declare_parameter("max_range", 5.0).value)
        self.px = int(self.declare_parameter("image_px", 360).value)
        self.cloud_pub = self.create_publisher(PointCloud2, "/rtis/energyscape_cloud", 5)
        self.img_pub = self.create_publisher(Image, "/rtis/energyscape_image", 5)
        self.create_subscription(Energyscape, "/rtis/energyscape", self._es, 10)
        self._lut = None

    def _fan_lut(self, r, az):
        """Pixel -> (range index, azimuth index) of a fan image, forward up."""
        n = self.px
        h = n // 2 + 10
        u, v = np.meshgrid(np.arange(n), np.arange(h))
        y = (n / 2 - u) / (n / 2) * self.rmax          # left positive
        x = (h - 1 - v) / (n / 2) * self.rmax           # forward
        rr, aa = np.hypot(x, y), np.arctan2(y, x)
        ir = np.round((rr - r[0]) / (r[1] - r[0])).astype(int)
        ia = np.round((aa - az[0]) / (az[1] - az[0])).astype(int)
        ok = (ir >= 0) & (ir < len(r)) & (ia >= 0) & (ia < len(az)) & (rr <= self.rmax)
        return np.clip(ir, 0, len(r) - 1), np.clip(ia, 0, len(az) - 1), ok

    def _es(self, m: Energyscape):
        if m.method != self.method:
            return
        r, az, es = msg_to_es(m)
        keep = r <= self.rmax
        L = es + 40 * np.log10(np.maximum(r, 0.05))[:, None]      # range-compensated level
        top = float(L[keep].max())
        rel = np.clip((L - (top - self.dr)) / self.dr, 0, 1)
        # point cloud: cells above the floor
        R, A = np.meshgrid(r, az, indexing="ij")
        sel = (rel > 0) & keep[:, None]
        pts = np.zeros(int(sel.sum()), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
        pts["x"], pts["y"] = R[sel] * np.cos(A[sel]), R[sel] * np.sin(A[sel])
        pts["intensity"] = rel[sel] * self.dr
        c = PointCloud2()
        c.header = m.header
        c.height, c.width = 1, len(pts)
        c.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(("x", "y", "z", "intensity"))]
        c.is_bigendian, c.point_step, c.row_step, c.is_dense = False, 16, 16 * len(pts), True
        c.data = pts.tobytes()
        self.cloud_pub.publish(c)
        # fan image
        if self._lut is None:
            self._lut = self._fan_lut(r, az)
        ir, ia, ok = self._lut
        img = inferno(rel[ir, ia])
        img[~ok] = 25
        im = Image()
        im.header = m.header
        im.height, im.width = img.shape[:2]
        im.encoding, im.is_bigendian, im.step = "rgb8", 0, 3 * img.shape[1]
        im.data = img.tobytes()
        self.img_pub.publish(im)


def main():
    rclpy.init()
    n = EnergyscapeViz()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()
