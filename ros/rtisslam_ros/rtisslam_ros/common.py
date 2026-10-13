"""Message conversion helpers shared by the nodes."""
from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import Quaternion, TransformStamped

from rtis_msgs.msg import Energyscape


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z, q.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return q


def quat_to_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def transform(stamp, parent: str, child: str, x: float, y: float, yaw: float) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, parent, child
    t.transform.translation.x, t.transform.translation.y = float(x), float(y)
    t.transform.rotation = yaw_to_quat(yaw)
    return t


def es_to_msg(stamp, frame: str, method: str, ranges, azimuths, es_db) -> Energyscape:
    m = Energyscape()
    m.header.stamp, m.header.frame_id, m.method = stamp, frame, method
    m.ranges = np.asarray(ranges, np.float32).tolist()
    m.azimuths = np.asarray(azimuths, np.float32).tolist()
    m.data = np.asarray(es_db, np.float32).ravel().tolist()
    return m


def msg_to_es(m: Energyscape):
    r, az = np.asarray(m.ranges, np.float32), np.asarray(m.azimuths, np.float32)
    return r, az, np.asarray(m.data, np.float32).reshape(len(r), len(az))


def parse_pose(s) -> list[float]:
    """'x,y,yaw' (launch argument) -> [x, y, yaw]."""
    v = [float(t) for t in str(s).replace("[", "").replace("]", "").split(",") if t.strip()]
    return (v + [0.0, 0.0, 0.0])[:3]
