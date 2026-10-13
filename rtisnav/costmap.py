"""Planning grid from an RTISSLAM map (shared by the offline Navigator and the ROS 2 node that publishes /map for Nav2).

    tern, L, origin, res = planning_grid(slam, cfg)     # tern: int8 (100 occupied, 0 free, -1 unknown)

Steps: the occupancy grid rendered from the map's node poses; free = log-odds <= FREE_THRESH (one free-ray hit),
pinholes of unknown inside observed free space closed (free-space rays fan out with range); the space swept by the
robot while mapping is free within SWEPT_FREE of its path (it drove there) and unknown within SWEPT_UNKNOWN is free.
Nav2 then runs with allow_unknown = false: specular walls leave holes in the occupied evidence, so only observed or
driven space is planned through.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt


@dataclass
class CostmapConfig:
    free_thresh: float = -0.2
    close_iter: int = 2
    swept_free: float = 0.4
    swept_unknown: float = 0.5


def planning_grid(slam, cfg: CostmapConfig | None = None):
    cfg = cfg or CostmapConfig()
    n = slam.n_map_nodes or slam.n
    P = slam.poses()[:n]
    L, org, res = slam.grid.render(P)
    tern = slam.grid.ternary(L)
    free = (L <= cfg.free_thresh) & (tern != 100)
    free = binary_closing(free, iterations=cfg.close_iter) & (tern != 100)
    tern = np.where(free, 0, tern).astype(np.int8)
    sw = np.zeros(L.shape, bool)
    sw[np.clip(((P[:, 0] - org[0]) / res).astype(int), 0, L.shape[0] - 1),
       np.clip(((P[:, 1] - org[1]) / res).astype(int), 0, L.shape[1] - 1)] = True
    d = distance_transform_edt(~sw) * res
    tern[(tern == 100) & (d <= cfg.swept_free)] = 0
    tern[(tern == -1) & (d <= cfg.swept_unknown)] = 0
    return tern, L, org, res
