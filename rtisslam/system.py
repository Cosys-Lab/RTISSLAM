"""RTISSLAM: BatSLAM 2.0 on eRTIS energyscapes + an occupancy grid anchored to the pose graph.

    slam = RTISSLAM(RTISSLAMConfig(), ranges, azimuths, initial_pose)
    for each sonar pulse:  slam.process(maps, odom_pose)      # maps: {method: [R, D] dB}, odom_pose (x, y, yaw)
    slam.pose()            current pose estimate (map frame)
    slam.map_to_odom()     the TF map -> odom
    slam.grid_map()        (log-odds grid, origin, res) rendered from the current graph
    slam.save(path); RTISSLAM.load(path, ...)                  map persistence
    slam.start_localization(pose, sigma)                       localization-only session in a loaded map
    slam.start_localization(pose, sigma, localizer="mix")      ... with a Gaussian-mixture (multi-hypothesis EKF) or
                                                               particle filter ("pf") instead of the pose graph
                                                               (rtisslam/localize.py); pose=None: unknown start

Pulse gating (as on the robot): a pulse becomes a pose-graph node only after GATE_M of travel or GATE_RAD of turning
(odometry); `process` returns None for skipped pulses.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gtsam
import numpy as np

from batslam.backend.pose_graph import X, pose2
from batslam.config import BatSLAMConfig, _update
from batslam.slam import BatSLAM

from .acoustic_odometry import AcousticOdometry, AcousticOdometryConfig
from .descriptor import EnergyscapeDescriptor, EnergyscapeDescriptorConfig
from .grid import GridConfig, KeyframeGrid
from .register import RegisterConfig, compose_submap, register


def default_batslam() -> BatSLAMConfig:
    c = BatSLAMConfig()
    c.templates.spacing_m = 0.3
    # evidence curve of the descriptor on DMAS3 + CF + RW-CFAR 2 s (scripts/fit_evidence.py, office development drive,
    # top-5 candidates): P(correct | s) = 1/2 at 0.888, logit +2 at 0.929; the candidate floor is where l(s) reaches -4
    c.recognition.llr_mid = 0.888
    c.recognition.llr_slope = 49.3
    c.templates.min_similarity = 0.807
    c.templates.exclude_recent_m = 5.0
    c.graph.use_shift_offset = True
    return c


# energyscapes ("imaging:post", rtisslam/imaging_post.py): one energyscape for every role, third-order DMAS with
# coherence factor and a smoothed range-wise CFAR of gain 2 (no learned imaging); the descriptor was tuned for it on
# the office development drive (scripts/tune_descriptor.py: 20 sectors, compression gamma 0.2)
DESCRIPTOR_KEY = AO_KEY = GRID_KEY = "dmas3cf:ams2s"
DESCRIPTOR_SECTORS, DESCRIPTOR_GAMMA = 20, 0.2


@dataclass
class RTISSLAMConfig:
    descriptor: EnergyscapeDescriptorConfig = field(default_factory=lambda: EnergyscapeDescriptorConfig(
        method=DESCRIPTOR_KEY, n_sectors=DESCRIPTOR_SECTORS, gamma=DESCRIPTOR_GAMMA))
    grid: GridConfig = field(default_factory=lambda: GridConfig(method=GRID_KEY))
    batslam: BatSLAMConfig = field(default_factory=default_batslam)
    register: RegisterConfig = field(default_factory=RegisterConfig)
    # reflectors closer than 1.5 m bias the rotation of the flow estimate on DMAS3 + CF (-1.5 % -> -0.6 %, office drive)
    acoustic_odometry: AcousticOdometryConfig = field(default_factory=lambda: AcousticOdometryConfig(
        method=AO_KEY, model="flow", r_min=1.5))
    use_acoustic_odometry: bool = True
    ao_factors: bool = False                  # acoustic odometry as between factors (off: calibration wins, see report)
    ao_calibrate: bool = True                 # acoustic odometry calibrates the wheel odometry (scale of ds and dtheta)
    ao_calib_min: int = 40                    # accepted pairs before the calibration is applied
    ao_calib_sharp: float = 1.6               # only sharp correlation peaks feed the calibration
    ao_sigma: tuple = (0.01, 0.02, 0.006)      # m, m, rad per node step (acoustic odometry factor)
    register_method: str = DESCRIPTOR_KEY   # energyscape whose echo detections are registered ("" = the grid's)
    gate_m: float = 0.10
    gate_rad: float = math.radians(6)
    # recognition for localization sessions (llr_mid, llr_slope, min_similarity; empty = the mapping values): the map is
    # frozen, so a wrong link cannot bend it, and the robot is always in mapped space (a true revisit is far more
    # likely than while mapping); fitted across sessions by scripts/fit_evidence.py --query-tag
    localization: dict = field(default_factory=dict)

    def energyscape_keys(self) -> set:
        """All energyscapes ("imaging:post" keys) that process() needs."""
        return {self.descriptor.method, self.grid.method, self.acoustic_odometry.method} | ({self.register_method} - {""})

    def to_dict(self):
        return {"descriptor": asdict(self.descriptor), "grid": asdict(self.grid), "batslam": self.batslam.to_dict(),
                "register": asdict(self.register), "acoustic_odometry": asdict(self.acoustic_odometry),
                "use_acoustic_odometry": self.use_acoustic_odometry, "ao_sigma": list(self.ao_sigma),
                "ao_factors": self.ao_factors, "ao_calibrate": self.ao_calibrate, "ao_calib_min": self.ao_calib_min,
                "ao_calib_sharp": self.ao_calib_sharp,
                "register_method": self.register_method, "gate_m": self.gate_m, "gate_rad": self.gate_rad,
                "localization": dict(self.localization)}

    @classmethod
    def from_dict(cls, d):
        c = cls()
        c.register_method = ""                  # maps saved before register_method existed registered the grid's echoes
        for k in ("register_method", "gate_m", "gate_rad", "use_acoustic_odometry", "ao_factors", "ao_calibrate", "ao_calib_min", "ao_calib_sharp"):
            if k in d:
                setattr(c, k, d[k])
        if "descriptor" in d:
            _update(c.descriptor, d["descriptor"])
        if "grid" in d:
            gd = dict(d["grid"]); gd["mount"] = tuple(gd.get("mount", c.grid.mount))
            _update(c.grid, gd)
        if "ao_sigma" in d:
            c.ao_sigma = tuple(d["ao_sigma"])
        if "acoustic_odometry" in d:
            ad = dict(d["acoustic_odometry"]); ad["mount"] = tuple(ad.get("mount", c.acoustic_odometry.mount))
            _update(c.acoustic_odometry, ad)
        if "register" in d:
            rd = dict(d["register"]); rd["sigma"] = tuple(rd.get("sigma", c.register.sigma))
            _update(c.register, rd)
        if "batslam" in d:
            c.batslam = BatSLAMConfig.from_dict(d["batslam"])
        c.localization = dict(d.get("localization", {}))
        return c


@dataclass
class FilterLog:
    n_candidates: int
    best_similarity: float
    mass: float                     # weight of the reported mode / hypothesis


def _between(a, b):
    return pose2(a).between(pose2(b))


class RTISSLAM:
    def __init__(self, cfg: RTISSLAMConfig, ranges, azimuths, initial_pose=(0.0, 0.0, 0.0)):
        self.cfg = cfg
        self.ranges, self.azimuths = np.asarray(ranges), np.asarray(azimuths)
        self.desc = EnergyscapeDescriptor(cfg.descriptor, ranges, azimuths)
        self.slam = BatSLAM(cfg.batslam, BatSLAM.descriptor_shape(cfg.batslam, self.desc.shape),
                            cfg.descriptor.range_bin, initial_pose=tuple(initial_pose))
        self.grid = KeyframeGrid(cfg.grid, ranges, azimuths)
        self.mode = "mapping"
        self.node_odom: list[np.ndarray] = []         # odometry pose per node (this process)
        self._last_odom = None
        self._session_start = 0
        self.n_map_nodes = 0
        self.localizer, self.track = None, []          # mixture / particle localizer (start_localization)
        self.node_pts: list[np.ndarray] = []          # echo detections per node (base_link), for registration
        self.reg_log: list = []                       # (query, anchor, accepted, score, contrast, correction)
        self.ao = AcousticOdometry(cfg.acoustic_odometry, ranges, azimuths) if cfg.use_acoustic_odometry else None
        self._ao_prev = None                          # previous node's acoustic-odometry image
        self.ao_log: list = []                        # (node, ds, dth, ok, peak, sharp)
        # wheel calibration from acoustic odometry: least-squares scale of ds and dtheta (sums of products)
        self._cal = np.zeros(4)                       # sum(ao_ds wh_ds), sum(wh_ds^2), sum(ao_th wh_th), sum(wh_th^2)
        self._cal_n = 0
        self.k_s, self.k_th = 1.0, 1.0
        if cfg.register.enabled:
            self.slam.refine_link = self._refine

    # ------------------------------------------------------------------ per pulse
    def gate(self, odom) -> bool:
        if self._last_odom is None:
            return True
        d = _between(self._last_odom, odom)
        return math.hypot(d.x(), d.y()) >= self.cfg.gate_m or abs(d.theta()) >= self.cfg.gate_rad

    def process(self, maps: dict, odom, force: bool = False):
        odom = np.asarray(odom, float)
        if not (force or self.gate(odom)):
            return None
        if self.localizer is not None:
            return self._process_filter(maps, odom)
        if self._last_odom is None:
            delta = None
            raw = None
        else:
            d = _between(self._last_odom, odom)
            raw = (d.x(), d.y(), d.theta())
            delta = (raw[0] * self.k_s, raw[1] * self.k_s, raw[2] * self.k_th)        # calibrated wheel odometry
        occ, _ = self.grid.evidence(maps[self.cfg.register_method or self.cfg.grid.method])
        self.node_pts.append(occ[:, :2].copy())
        self.node_odom.append(odom)
        log = self.slam.process(self.desc.compute(maps[self.cfg.descriptor.method]), delta)
        self._last_odom = odom
        if self.ao is not None:
            img = self.ao.image(maps[self.cfg.acoustic_odometry.method])
            if delta is not None and self._ao_prev is not None:
                # search window centred on the RAW wheel increment: centring it on the calibrated one closes a
                # feedback loop (broad correlation peaks lean towards the window centre)
                ds = math.hypot(raw[0], raw[1]) * (1 if raw[0] >= 0 else -1)
                e = self.ao.estimate(self._ao_prev, img, ds, raw[2])
                self.ao_log.append((log.key, e.ds, e.dth, e.ok, e.peak, e.sharp))
                if e.ok:
                    if self.cfg.ao_factors:
                        z = (e.ds * math.cos(e.dth / 2), e.ds * math.sin(e.dth / 2), e.dth)
                        self.slam.graph.add_between(log.key - 1, log.key, z, self.cfg.ao_sigma)
                    if self.cfg.ao_calibrate and e.sharp >= self.cfg.ao_calib_sharp:
                        wds = math.hypot(raw[0], raw[1]) * (1 if raw[0] >= 0 else -1)
                        self._cal += (e.ds * wds, wds * wds, e.dth * raw[2], raw[2] * raw[2])
                        self._cal_n += 1
                        if self._cal_n >= self.cfg.ao_calib_min and self._cal[3] > 0.5:
                            self.k_s = float(np.clip(self._cal[0] / self._cal[1], 0.9, 1.1))
                            self.k_th = float(np.clip(self._cal[2] / self._cal[3], 0.9, 1.1))
            self._ao_prev = img
        if self.mode == "mapping" and self.grid.want_keyframe(odom):
            self.grid.add(log.key, odom, maps[self.cfg.grid.method])
        return log

    def _refine(self, c, z):
        """BatSLAM hook: register the query submap against the anchor submap (see register.py)."""
        rc = self.cfg.register
        s = np.asarray(self.slam.odom_s)
        q, a = c.query_key, c.anchor_key
        Q = compose_submap(q, self.node_pts, self.node_odom, s, rc.submap_m, causal=True)
        A = compose_submap(a, self.node_pts, self.node_odom, s, rc.submap_m, causal=False)
        zi = z.inverse()
        p, sc, con = register(Q, A, (zi.x(), zi.y(), zi.theta()), rc)
        if p is None:
            self.reg_log.append((q, a, False, sc, con, 0.0))
            return z, None
        zr = gtsam.Pose2(*p).inverse()
        self.reg_log.append((q, a, True, sc, con, float(np.hypot(p[0] - zi.x(), p[1] - zi.y()))))
        return zr, rc.sigma

    # ------------------------------------------------------------------ state
    @property
    def n(self) -> int:
        return self.slam.graph.n

    def pose(self, key: int | None = None) -> np.ndarray:
        if key is None and self.localizer is not None:
            return self._est[0].copy()
        p = self.slam.graph.pose(self.n - 1 if key is None else key)
        return np.array([p.x(), p.y(), p.theta()])

    def poses(self) -> np.ndarray:
        return self.slam.graph.poses()

    def map_to_odom(self) -> np.ndarray:
        """T_map_odom such that T_map_odom * odom_pose(last node) = pose(last node)."""
        if self._last_odom is None:
            return np.zeros(3)
        if self.localizer is not None:
            t = pose2(self._est[0]).compose(pose2(self._last_odom).inverse())
            return np.array([t.x(), t.y(), t.theta()])
        t = self.slam.graph.pose(self.n - 1).compose(pose2(self._last_odom).inverse())
        return np.array([t.x(), t.y(), t.theta()])

    def covariance(self) -> np.ndarray:
        if self.localizer is not None:
            return self._est[1].copy()
        return self.slam.graph.marginal_covariance(self.n - 1)

    def grid_map(self, poses=None, bounds=None):
        return self.grid.render(self.poses() if poses is None else poses, bounds)

    def links(self) -> list:
        return [(l.query_key, l.anchor_key) for l in self.slam.graph.loops]

    def template_nodes(self) -> np.ndarray:
        return np.array([t.anchor_key for t in self.slam.templates.templates], int)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        """Map = pose graph (node poses, odometry chain, active view links), templates, grid keyframes, config."""
        g = self.slam.graph
        P = self.poses()
        links = [(l.query_key, l.anchor_key, *l.measured, l.hypothesis_id) for l in g.loops]
        odo = np.array([[p.x(), p.y(), p.theta()] for p in self.slam.odom])
        T = self.slam.templates
        np.savez_compressed(
            path, poses=P, odom=odo, odom_s=np.asarray(self.slam.odom_s), links=np.asarray(links, float).reshape(-1, 6),
            tpl_anchor=np.array([t.anchor_key for t in T.templates], np.int64),
            tpl_s=np.array([t.odom_s for t in T.templates]),
            tpl_desc=np.stack([t.descriptor for t in T.templates]) if T.templates else np.zeros((0,) + T.shape, np.float32),
            ranges=self.ranges, azimuths=self.azimuths, config=json.dumps(self.cfg.to_dict()),
            node_odom=np.asarray(self.node_odom), wheel_cal=np.r_[self._cal, self._cal_n, self.k_s, self.k_th], node_pts_n=np.array([len(x) for x in self.node_pts], np.int64),
            node_pts=np.vstack(self.node_pts).astype(np.float32) if self.node_pts else np.zeros((0, 2), np.float32),
            **self.grid.state())

    @classmethod
    def load(cls, path, cfg: RTISSLAMConfig | None = None) -> "RTISSLAM":
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}          # read once: NpzFile decompresses on every access
        cfg = cfg or RTISSLAMConfig.from_dict(json.loads(str(d["config"])))
        P = d["poses"]
        self = cls(cfg, d["ranges"], d["azimuths"], initial_pose=tuple(P[0]))
        g, s = self.slam.graph, self.slam
        # pose graph: prior on node 0, odometry chain from the stored odometry, links; initial values = map poses
        odo = d["odom"]
        g.add_first(tuple(P[0]))
        for i in range(1, len(P)):
            dd = pose2(odo[i - 1]).between(pose2(odo[i]))
            g.add_odometry((dd.x(), dd.y(), dd.theta()))
        for q, a, x, y, th, hid in d["links"]:
            g.add_view_link(int(q), int(a), (x, y, th), hypothesis_id=int(hid))
        for i in range(len(P)):
            g._pending_values.update(X(i), pose2(P[i]))
        g.update(extra_iterations=3)
        s.odom = [pose2(o) for o in odo]
        s.odom_s = list(d["odom_s"])
        for k, desc, ts in zip(d["tpl_anchor"], d["tpl_desc"], d["tpl_s"]):
            s.templates.add(int(k), desc, float(ts))
        self.grid.load_state(d)
        self.node_odom = [np.asarray(o) for o in d["node_odom"]]
        oo = np.r_[0, np.cumsum(d["node_pts_n"])]
        self.node_pts = [d["node_pts"][oo[i]:oo[i + 1]] for i in range(len(P))]
        self.n_map_nodes = len(P)
        if "wheel_cal" in d:                          # the wheel calibration learned while mapping carries over
            wc = d["wheel_cal"]
            self._cal, self._cal_n, self.k_s, self.k_th = wc[:4].copy(), int(wc[4]), float(wc[5]), float(wc[6])
        return self

    def start_localization(self, pose, sigma=(0.5, 0.5, 0.3), localizer: str = "graph", loc_cfg=None) -> None:
        """Localization-only session in the loaded map: the next pulse starts a detached node with a prior at `pose`;
        no new templates, no new grid keyframes (the map stays as it is).
        localizer: "graph" (BatSLAM pose graph with a detached session), "mix" (Gaussian mixture of EKFs) or "pf"
        (particle filter), see rtisslam/localize.py; loc_cfg: MixtureConfig / PFConfig; pose None = unknown start
        (filters only)."""
        self.mode = "localization"
        if pose is None and localizer == "graph":
            raise ValueError("the graph localizer needs a start pose")
        loc = self.cfg.localization
        rc, tc = self.cfg.batslam.recognition, self.cfg.batslam.templates
        # the mapping curve stays available: an UNCONDITIONED recognition (one that would start a new track) is
        # judged with it, a recognition near the believed pose with the localization curve (rtisslam/localize.py)
        self.map_curve = (rc.llr_mid, rc.llr_slope)
        rc.llr_mid = loc.get("llr_mid", rc.llr_mid)
        rc.llr_slope = loc.get("llr_slope", rc.llr_slope)
        tc.min_similarity = loc.get("min_similarity", tc.min_similarity)
        # freeze the map: tight priors on all map nodes, so that the localization links move the robot, not the map
        # (the planning grid is rendered from these node poses)
        g = self.slam.graph
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-4]))
        for i, p in enumerate(self.poses()[:self.n]):
            g._add_base(gtsam.PriorFactorPose2(X(i), pose2(p), noise))
        g.update()
        self.slam._initial = tuple(pose) if pose is not None else (0.0, 0.0, 0.0)
        self.slam.initial_sigma = tuple(sigma)
        self.slam.lay_templates = False
        self._last_odom = None
        self._ao_prev = None
        self._session_start = self.n
        self.localizer, self.track = None, []
        if localizer == "graph":
            return
        from .localize import (LikelihoodField, MixtureConfig, MixtureLocalizer, ParticleLocalizer, PFConfig,
                               Recognition)
        self._rec = Recognition(self)
        self._global = pose is None
        if localizer == "pf":
            fld = LikelihoodField(self, self._rec.P_map)
            self.localizer = ParticleLocalizer(loc_cfg or PFConfig(), fld)
            if self._global:
                self.localizer.start_global(fld.free_xy)
            else:
                self.localizer.start(pose, sigma)
        elif localizer == "mix":
            self.localizer = MixtureLocalizer(loc_cfg or MixtureConfig(), LikelihoodField(self, self._rec.P_map))
            if not self._global:
                self.localizer.start(pose, sigma)
        else:
            raise ValueError(f"unknown localizer {localizer!r}")
        p0 = np.zeros(3) if pose is None else np.asarray(pose, float)
        self._est = (p0, np.diag(np.square(sigma if pose is not None else (10.0, 10.0, math.pi))), 0.0)

    def _process_filter(self, maps: dict, odom):
        """One pulse of the mixture / particle localizer (localization mode)."""
        from .localize import ParticleLocalizer
        delta = None
        if self._last_odom is not None:
            d = _between(self._last_odom, odom)
            delta = (d.x() * self.k_s, d.y() * self.k_s, d.theta() * self.k_th)     # calibrated wheel odometry
        self._last_odom = odom
        view = self.desc.compute(maps[self.cfg.descriptor.method])
        cands = self._rec.candidates(view, 0.0 if delta is None else math.hypot(delta[0], delta[1]))
        loc = self.localizer
        if delta is not None:
            loc.predict(delta)
        occ = self.grid.evidence(maps[self.cfg.grid.method])[0] if loc.cfg.use_grid else None
        if isinstance(loc, ParticleLocalizer):
            loc.update(cands, occ, delta)
        elif self._global and not loc.H:
            if cands:
                loc.start_global(cands)
        else:
            loc.update(cands, delta, occ)
        if isinstance(loc, ParticleLocalizer) or loc.H:
            self._est = loc.estimate()
        self.track.append(self._est[0].copy())
        return FilterLog(len(cands), max((c.sim for c in cands), default=float("nan")), self._est[2])
