"""Localization in a frozen RTISSLAM map: a particle filter and a multi-hypothesis EKF in (x, y, theta), driven by the
place recognition of BatSLAM 2.0 against the map's templates.

Jan, 2026-10-12: "If we recognize a pose in the pose graph from BatSLAM, we know some info about the position, and can
use that to resample the particle cloud. When we have no recognition, we propagate the particles with the odometry."

Both filters share the recognition front end (`Recognition`):
  - every pulse, the local view (BatSLAM's stack of recent views) is compared with every template of the map;
  - each of the top-k templates is a candidate: its anchor node has a frozen map pose A, the best range shift s gives
    the implied robot pose Z = A * (-s, 0, 0) (as BatSLAM's view links), and two evidence curves give P(correct |
    similarity) = sigmoid(l(s)):
      p      the localization curve (scripts/fit_evidence.py --query-tag; the map stores it): for a candidate inside
             the gate of a hypothesis / near particles, i.e. CONDITIONED on being near the believed pose, where a true
             template is likely (the analogue of BatSLAM's plausibility gate);
      p_new  the mapping curve (fitted where most of the drive has no template): for an UNCONDITIONED candidate that
             would start a new track (spawn / anchor resampling). Off the mapping route (other directions, turns in
             place) most top candidates are wrong, e.g. the corridor seen backwards; in closed-loop navigation only ~2 %
             of the pulses had a correct one among the top 5, so new tracks must be judged strictly;
  - the echo detections of the grid energyscape, for the likelihood field of the occupancy grid (particle filter).

Measurement model of a recognition (both filters): with probability P_j the robot is at Z_j (Gaussian, `view_sigma`),
otherwise the candidate is clutter. Consecutive local views overlap (each stacks ~1 m of travel), so counting every
pulse as independent evidence lets a few aliased pulses outweigh a good track: the evidence of a pulse is weighted by
the motion since the previous one relative to `corr_len` (`evidence_weight`), i.e. evidence per metre, not per pulse.
A place mapped on several passes has several templates, which all match at once: summing their evidence lets a place
mapped three times outvote the true place mapped once (seen between the two parallel corridors of the office). Both
filters therefore use the BEST supporting candidate per hypothesis / particle, not the sum.

ParticleLocalizer   n particles; odometry propagation with noise; weights from the recognitions (and optionally the
                    likelihood field of the grid); systematic resampling; ANCHOR RESAMPLING: a confident recognition
                    the cloud does not support replaces part of the cloud by samples around its implied pose (this is
                    what re-localizes after a gap, or from a wrong / unknown start). Estimate: the dominant mode.
GRID EVIDENCE (both filters): the echo detections of the pulse, placed at a hypothesised pose, must fall on the
walls of the map's occupancy grid (likelihood field). Unlike the views, this does not depend on the driving
direction: in closed-loop navigation, driving one corridor westwards matched the templates of the parallel corridor
driven eastwards (a self-consistent wrong track); the grid tells the two apart.

MixtureLocalizer    up to `max_hyp` Gaussian hypotheses (each an EKF; "Rao-Blackwellized PF with 10 particles"):
                    odometry = EKF prediction; every pulse each hypothesis branches into "no candidate is right" and
                    "candidate j is right" (EKF update, weight x the measurement likelihood); branches are merged and
                    pruned; a confident recognition far from every hypothesis spawns a new one. Estimate: the heaviest
                    hypothesis (mean and covariance).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def compose(P: np.ndarray, d: np.ndarray) -> np.ndarray:
    """P [n, 3] (+) d [n, 3] or [3] (SE(2), d in the frame of P)."""
    d = np.broadcast_to(d, P.shape)
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2])
    return np.c_[P[:, 0] + c * d[:, 0] - s * d[:, 1], P[:, 1] + s * d[:, 0] + c * d[:, 1], wrap(P[:, 2] + d[:, 2])]


# ---------------------------------------------------------------------------------------------- recognition front end
@dataclass
class Candidate:
    z: np.ndarray            # implied robot pose in the map (x, y, theta)
    p: float                 # P(correct | similarity), localization curve (near the believed pose)
    p_new: float             # P(correct | similarity), mapping curve (unconditioned: new tracks)
    sim: float
    anchor: int              # anchor node of the template


class Recognition:
    """Per pulse: candidates (implied poses + P(correct)) and echo detections, from a loaded RTISSLAM map."""

    def __init__(self, slam, top_k: int = 5):
        self.slam = slam
        self.bs = slam.slam                                        # BatSLAM
        self.top_k = top_k
        n = slam.n_map_nodes or slam.n
        self.P_map = slam.poses()[:n].copy()
        self.anchor = slam.template_nodes()
        self.A = self.P_map[self.anchor]
        self.s = 0.0                                               # odometric travel (local-view stacking)
        rc, tc = slam.cfg.batslam.recognition, slam.cfg.batslam.templates
        self.rc, self.tc = rc, tc
        self.range_bin = slam.cfg.descriptor.range_bin
        self.map_curve = getattr(slam, "map_curve", (rc.llr_mid, rc.llr_slope))

    def p_correct(self, sim, curve=None) -> np.ndarray:
        mid, slope = curve or (self.rc.llr_mid, self.rc.llr_slope)
        l = np.clip(slope * (np.asarray(sim) - mid), -self.rc.llr_clip, self.rc.llr_clip)
        return 1.0 / (1.0 + np.exp(-l))

    def candidates(self, view: np.ndarray, ds: float) -> list[Candidate]:
        self.s += abs(ds)
        d = self.bs._local_view(view, self.s)
        self.bs.templates.observe(d)
        sim, shift = self.bs.templates.similarities(d)
        if sim.size == 0:
            return []
        out = []
        for j in np.argsort(-sim)[:self.top_k]:
            if sim[j] < self.tc.min_similarity:
                break
            z = compose(self.A[j][None], np.array([-shift[j] * self.range_bin, 0.0, 0.0]))[0]
            out.append(Candidate(z, float(self.p_correct(sim[j])), float(self.p_correct(sim[j], self.map_curve)),
                                 float(sim[j]), int(self.anchor[j])))
        self.last = out
        return out


class LikelihoodField:
    """Distance to the nearest occupied cell of the map's occupancy grid (log-odds >= occ_thresh)."""

    def __init__(self, slam, P_map: np.ndarray):
        L, org, res = slam.grid.render(P_map)
        occ = L >= slam.grid.cfg.occ_thresh
        self.d = (distance_transform_edt(~occ) * res).astype(np.float32)
        self.org, self.res = np.asarray(org, float), float(res)
        self.gx, self.gy = (g.astype(np.float32) / res for g in np.gradient(self.d))   # d(distance)/dx, /dy
        fi, fj = np.nonzero((L <= slam.grid.cfg.free_thresh) & (self.d > 0.2))
        self.free_xy = np.c_[self.org[0] + (fi + 0.5) * res, self.org[1] + (fj + 0.5) * res]   # unknown start

    def dist_grad(self, xy: np.ndarray):
        """xy [n, 2] -> distance [n], gradient [n, 2] (outside the grid: 1 m, zero gradient)."""
        i = np.floor((xy[:, 0] - self.org[0]) / self.res).astype(np.int64)
        j = np.floor((xy[:, 1] - self.org[1]) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.d.shape[0]) & (j >= 0) & (j < self.d.shape[1])
        d = np.full(len(xy), 1.0, np.float32)
        g = np.zeros((len(xy), 2), np.float32)
        d[ok] = self.d[i[ok], j[ok]]
        g[ok, 0], g[ok, 1] = self.gx[i[ok], j[ok]], self.gy[i[ok], j[ok]]
        return d, g

    def register(self, pose: np.ndarray, pts: np.ndarray, trunc: float = 0.3, sigma: float = 0.1, iters: int = 6):
        """Align the echo detections pts [m, 2] (base_link) with the map's walls, starting at pose: Gauss-Newton on
        the truncated distance field. -> (pose, information matrix [3, 3] of the alignment, inlier count) or None.
        The information is small along directions the walls do not constrain (along a corridor)."""
        x = np.asarray(pose, float).copy()
        H = None
        for _ in range(iters):
            c, s = math.cos(x[2]), math.sin(x[2])
            P = np.c_[x[0] + c * pts[:, 0] - s * pts[:, 1], x[1] + s * pts[:, 0] + c * pts[:, 1]]
            d, g = self.dist_grad(P)
            inl = d < trunc
            if inl.sum() < 15:
                return None
            dth = np.c_[-s * pts[:, 0] - c * pts[:, 1], c * pts[:, 0] - s * pts[:, 1]]
            J = np.c_[g[:, 0], g[:, 1], (g * dth).sum(1)][inl]
            r = d[inl]
            H = J.T @ J + 1e-6 * np.eye(3)
            step = -np.linalg.solve(H, J.T @ r)
            step = np.clip(step, [-0.1, -0.1, -0.05], [0.1, 0.1, 0.05])
            x += step
            x[2] = wrap(x[2])
            if np.abs(step[:2]).max() < 0.005 and abs(step[2]) < 0.002:
                break
        return x, H / sigma ** 2, int(inl.sum())

    def dist(self, xy: np.ndarray) -> np.ndarray:
        """xy [..., 2] map coordinates -> distance [m] (cells outside the grid: 1 m)."""
        i = np.floor((xy[..., 0] - self.org[0]) / self.res).astype(np.int64)
        j = np.floor((xy[..., 1] - self.org[1]) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.d.shape[0]) & (j >= 0) & (j < self.d.shape[1])
        out = np.full(i.shape, 1.0, np.float32)
        out[ok] = self.d[i[ok], j[ok]]
        return out


def evidence_weight(delta, corr_len: float, rad_m: float = 0.5) -> float:
    """Share of an independent observation a pulse carries: motion since the previous pulse (turning counts rad_m per
    rad) over the correlation length of the local view, at most 1."""
    if delta is None:
        return 1.0
    return float(min(1.0, (math.hypot(delta[0], delta[1]) + rad_m * abs(delta[2])) / corr_len))


# ------------------------------------------------------------------------------------------------- odometry noise
@dataclass
class OdomNoise:
    trans_per_m: float = 0.05              # sigma of the travelled distance, per m
    lat_per_m: float = 0.02                # sideways slip, per m
    rot_per_rad: float = 0.05              # sigma of the heading change, per rad turned
    rot_per_m: float = 0.03                # heading drift, rad per m
    floor: tuple = (0.003, 0.002, 0.003)   # per step (m, m, rad)

    def sigmas(self, delta) -> np.ndarray:
        d = math.hypot(delta[0], delta[1])
        return np.array([self.trans_per_m * d + self.floor[0], self.lat_per_m * d + self.floor[1],
                         self.rot_per_rad * abs(delta[2]) + self.rot_per_m * d + self.floor[2]])


# ------------------------------------------------------------------------------------------------- particle filter
@dataclass
class PFConfig:
    n: int = 2000
    odom: OdomNoise = field(default_factory=OdomNoise)
    view_sigma: tuple = (0.45, 0.20, 0.25)     # implied pose: along, across (m), heading (rad); templates 0.45 m apart
    view_gain: float = 1.0                      # tempering of the recognition evidence, per corr_len of motion
    corr_len: float = 0.6                       # m: motion over which the local view decorrelates
    view_odds_max: float = 50.0
    use_grid: bool = True
    grid_sigma: float = 0.10                    # m
    grid_z_rand: float = 0.3
    grid_points: int = 120
    grid_gain: float = 4.0                      # effective independent detections per pulse
    wall_d: float = 0.12                        # particle closer than this to an occupied cell: penalised
    wall_penalty: float = 0.2
    resample_neff: float = 0.5
    inject_p: float = 0.6                       # anchor resampling: candidates at least this likely (p_new)
    inject_support: float = 0.2                 # ... that hold less of the weight than this (within 3 sigma)
    inject_frac: float = 0.3                    # ... get up to this fraction of the cloud, x P(correct)
    inject_sigma: tuple = (0.25, 0.25, 0.15)
    mode_radius: float = 0.75                   # estimate: weighted mean within this of the densest particle
    switch_ratio: float = 10.0                  # another mode is reported only once it holds this x the mass of the
                                                # mode at the previous estimate (as the mixture)


class ParticleLocalizer:
    def __init__(self, cfg: PFConfig, field_: LikelihoodField | None, seed: int = 0):
        self.cfg = cfg
        self.field = field_
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((cfg.n, 3))
        self.w = np.full(cfg.n, 1.0 / cfg.n)
        self.n_injected = 0
        self.n_resampled = 0
        self.n_switches = 0
        self._prev = None                           # previous estimate (mode hysteresis)
        self._mode_mass = 0.0                       # mass of the reported mode (scales the anchor resampling)

    def start(self, pose, sigma) -> None:
        self.X = np.asarray(pose, float) + self.rng.normal(0, sigma, (self.cfg.n, 3))
        self.X[:, 2] = wrap(self.X[:, 2])
        self.w[:] = 1.0 / self.cfg.n

    def start_global(self, free_xy: np.ndarray) -> None:
        """Unknown start: uniform over the given free positions, uniform heading."""
        k = self.rng.integers(0, len(free_xy), self.cfg.n)
        self.X = np.c_[free_xy[k] + self.rng.uniform(-0.05, 0.05, (self.cfg.n, 2)),
                       self.rng.uniform(-np.pi, np.pi, self.cfg.n)]
        self.w[:] = 1.0 / self.cfg.n

    def predict(self, delta) -> None:
        sg = self.cfg.odom.sigmas(delta)
        self.X = compose(self.X, np.asarray(delta) + self.rng.normal(0, 1, (self.cfg.n, 3)) * sg)

    def _view_loglik(self, cands: list[Candidate], g: float) -> np.ndarray:
        c = self.cfg
        if not cands:
            return np.zeros(len(self.X))
        best = np.zeros(len(self.X))
        sa, sc, st = c.view_sigma
        for k in cands:
            dx, dy = self.X[:, 0] - k.z[0], self.X[:, 1] - k.z[1]
            ca, sn = math.cos(k.z[2]), math.sin(k.z[2])
            al, ac = ca * dx + sn * dy, -sn * dx + ca * dy
            dth = wrap(self.X[:, 2] - k.z[2])
            d2 = (al / sa) ** 2 + (ac / sc) ** 2 + (dth / st) ** 2
            odds = min(k.p / max(1 - k.p, 1e-6), c.view_odds_max)
            best = np.maximum(best, odds * np.exp(-0.5 * d2) * 10.0)   # 10: inverse background density per unit odds
        return g * c.view_gain * np.log1p(best)

    def _grid_loglik(self, occ: np.ndarray | None, g: float) -> np.ndarray:
        c = self.cfg
        if not c.use_grid or self.field is None or occ is None or not len(occ):
            return np.zeros(len(self.X))
        if len(occ) > c.grid_points:
            occ = occ[self.rng.choice(len(occ), c.grid_points, replace=False, p=occ[:, 2] / occ[:, 2].sum())]
        cth, sth = np.cos(self.X[:, 2]), np.sin(self.X[:, 2])
        gx = self.X[:, 0, None] + cth[:, None] * occ[None, :, 0] - sth[:, None] * occ[None, :, 1]
        gy = self.X[:, 1, None] + sth[:, None] * occ[None, :, 0] + cth[:, None] * occ[None, :, 1]
        d = self.field.dist(np.stack([gx, gy], -1))
        ll = np.log((1 - c.grid_z_rand) * np.exp(-0.5 * (d / c.grid_sigma) ** 2) + c.grid_z_rand)
        return g * c.grid_gain * ll.mean(axis=1)

    def update(self, cands: list[Candidate], occ: np.ndarray | None = None, delta=None) -> None:
        c = self.cfg
        g = evidence_weight(delta, c.corr_len)
        lw = self._view_loglik(cands, g) + self._grid_loglik(occ, g)
        if self.field is not None:
            lw += np.where(self.field.dist(self.X[:, :2]) < c.wall_d, math.log(c.wall_penalty), 0.0)
        lw += np.log(np.maximum(self.w, 1e-300))
        lw -= lw.max()
        self.w = np.exp(lw)
        self.w /= self.w.sum()
        if 1.0 / np.sum(self.w ** 2) < c.resample_neff * c.n:
            self._resample()
        self._inject(cands)

    def _resample(self) -> None:
        n = self.cfg.n
        u = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), u), n - 1)
        self.X = self.X[idx]
        self.w[:] = 1.0 / n
        self.n_resampled += 1

    def _inject(self, cands: list[Candidate]) -> None:
        """Anchor resampling: confident recognitions the cloud does not support get particles."""
        c = self.cfg
        sa, sc, st = c.view_sigma
        for k in cands:
            if k.p_new < c.inject_p:
                continue
            dx, dy = self.X[:, 0] - k.z[0], self.X[:, 1] - k.z[1]
            ca, sn = math.cos(k.z[2]), math.sin(k.z[2])
            d2 = ((ca * dx + sn * dy) / sa) ** 2 + ((-sn * dx + ca * dy) / sc) ** 2 + (wrap(self.X[:, 2] - k.z[2]) / st) ** 2
            support = self.w[d2 < 9.0].sum()
            if support >= c.inject_support:
                continue
            # a confident filter (heavy mode) barely injects: aliased places must not grow a cluster by injection
            # alone; a lost one injects up to inject_frac x P(correct) of the cloud
            m = int(c.n * c.inject_frac * k.p_new * (1 - support / c.inject_support) * (1 - self._mode_mass))
            if m <= 0:
                continue
            idx = np.argsort(self.w)[:m]                       # replace the least likely particles
            self.X[idx] = k.z + self.rng.normal(0, 1, (m, 3)) * np.asarray(c.inject_sigma)
            self.X[idx, 2] = wrap(self.X[idx, 2])
            self.w[idx] = np.median(self.w)
            self.w /= self.w.sum()
            self.n_injected += m

    def estimate(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(pose, covariance, mass) of the dominant mode."""
        c = self.cfg
        # density at each particle from a coarse histogram, the densest cell picks the mode
        cell = c.mode_radius
        ij = np.floor(self.X[:, :2] / cell).astype(np.int64)
        keys, inv = np.unique(ij, axis=0, return_inverse=True)
        inv = inv.ravel()
        mass = np.bincount(inv, weights=self.w, minlength=len(keys))
        # 3x3 neighbourhood sum
        lut = {tuple(k): i for i, k in enumerate(keys)}
        nb = np.array([sum(mass[lut[(a + da, b + db)]] for da in (-1, 0, 1) for db in (-1, 0, 1)
                           if (a + da, b + db) in lut) for a, b in keys])
        b = int(np.argmax(nb))
        if self._prev is not None:
            # hysteresis: stay with the mode at the previous estimate unless another one is switch_ratio x heavier
            pc = np.floor(self._prev[:2] / cell).astype(np.int64)
            near = np.nonzero(np.abs(keys - pc).max(axis=1) <= 1)[0]
            if len(near):
                p = near[np.argmax(nb[near])]
                if nb[b] < c.switch_ratio * nb[p]:
                    b = p
            if np.hypot(*(keys[b] - pc)) > 2:
                self.n_switches += 1
        best = keys[b]
        sel = (np.abs(ij - best).max(axis=1) <= 1)
        w = self.w[sel] / self.w[sel].sum()
        Xs = self.X[sel]
        th = math.atan2(np.sum(w * np.sin(Xs[:, 2])), np.sum(w * np.cos(Xs[:, 2])))
        mu = np.array([np.sum(w * Xs[:, 0]), np.sum(w * Xs[:, 1]), th])
        D = Xs - mu
        D[:, 2] = wrap(D[:, 2])
        C = (w[:, None] * D).T @ D + np.diag([1e-6, 1e-6, 1e-8])
        self._prev, self._mode_mass = mu, float(self.w[sel].sum())
        return mu, C, self._mode_mass


# --------------------------------------------------------------------------------------- multi-hypothesis EKF
@dataclass
class MixtureConfig:
    max_hyp: int = 10
    odom: OdomNoise = field(default_factory=OdomNoise)
    view_sigma: tuple = (0.45, 0.20, 0.25)     # measurement noise of an implied pose (along, across, heading)
    view_gain: float = 1.0                      # tempering, per corr_len of motion (as the particle filter)
    corr_len: float = 0.6
    view_odds_max: float = 50.0
    switch_ratio: float = 10.0                  # report another hypothesis only once it is this much heavier
    background: float = 0.002                   # density of a wrong candidate's implied pose (1 / (m^2 rad))
    gate_d2: float = 11.3                       # chi^2(3) 99 %: a candidate that can update a hypothesis
    merge_d2: float = 1.0                       # hypotheses this close (Mahalanobis) are merged
    prune_w: float = 1e-4
    spawn_p: float = 0.6                        # a candidate this likely (p_new), outside every gate, spawns one
    spawn_w: float = 1e-3                       # ... with this prior weight x p_new: a new track must earn its weight
    spawn_sigma: tuple = (0.30, 0.30, 0.20)
    # new tracks only while the reported hypothesis is in doubt (as the particle filter's injection, which scales with
    # 1 - mode mass): otherwise a self-consistent aliased track (the parallel corridor, driven the other way) gets a
    # foothold and wins on view evidence the grid cannot overrule (both corridors look alike in the grid too)
    doubt_w: float = 0.5                        # ... its weight is below this
    doubt_sigma: float = 0.75                   # ... or its sigma_xy (m) above this (long off-route stretch)
    doubt_fit: float = 0.3                      # ... or the share of echoes within fit_d of a wall (EMA) below this
    fit_d: float = 0.25                         # (kidnapped: the echoes stop fitting the grid around it)
    fit_ema: float = 0.2
    use_grid: bool = True                       # weight hypotheses by the grid likelihood of the echo detections
    grid_sigma: float = 0.10
    grid_z_rand: float = 0.3
    grid_points: int = 120
    grid_gain: float = 4.0                      # effective independent detections per corr_len of motion
    # grid REGISTRATION as an EKF measurement: the echoes are aligned with the walls around each hypothesis, and the
    # aligned pose updates its mean (the grid weighting alone only reweights hypotheses, so their means drifted with
    # the odometry off the mapping route, where recognitions are rare)
    use_registration: bool = True
    reg_inflate: float = 10.0                   # information / reg_inflate x evidence weight (correlated echoes)
    reg_max_sigma: float = 1.0                  # only hypotheses at most this uncertain (m) are registered
    reg_gate_d2: float = 11.3


@dataclass
class Hyp:
    mu: np.ndarray
    S: np.ndarray
    w: float
    id: int = 0


class MixtureLocalizer:
    def __init__(self, cfg: MixtureConfig, field_: LikelihoodField | None = None, seed: int = 0):
        self.cfg = cfg
        self.field = field_
        self.rng = np.random.default_rng(seed)
        self.H: list[Hyp] = []
        self.n_spawned = 0
        self.n_switches = 0
        self._next_id = 0
        self.current = -1                       # id of the reported hypothesis
        self.fit = 1.0                          # EMA of the share of echoes on the grid's walls, reported hypothesis
        self.in_doubt = True

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def start(self, pose, sigma) -> None:
        self.H = [Hyp(np.asarray(pose, float).copy(), np.diag(np.square(sigma)), 1.0, self._new_id())]
        self.current = self.H[0].id

    def start_global(self, cands: list[Candidate]) -> None:
        """Unknown start: one hypothesis per candidate of the first pulse (weights ~ P(correct))."""
        self.H = [Hyp(k.z.copy(), np.diag(np.square(self.cfg.spawn_sigma)), max(k.p_new, 1e-3), self._new_id())
                  for k in cands]
        self._normalise()
        self._merge()
        self._prune()

    def predict(self, delta) -> None:
        dx, dy, dth = delta
        Q = np.diag(np.square(self.cfg.odom.sigmas(delta)))
        for h in self.H:
            c, s = math.cos(h.mu[2]), math.sin(h.mu[2])
            F = np.array([[1, 0, -s * dx - c * dy], [0, 1, c * dx - s * dy], [0, 0, 1]])
            G = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            h.mu = compose(h.mu[None], np.asarray(delta, float))[0]
            h.S = F @ h.S @ F.T + G @ Q @ G.T

    def _R(self, z) -> np.ndarray:
        sa, sc, st = self.cfg.view_sigma
        c, s = math.cos(z[2]), math.sin(z[2])
        Rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return Rot @ np.diag([sa * sa, sc * sc, st * st]) @ Rot.T

    def _grid_loglik(self, X: np.ndarray, occ: np.ndarray) -> np.ndarray:
        """Mean log-likelihood of the echo detections occ [m, 3] (base_link) at poses X [n, 3]."""
        c = self.cfg
        cth, sth = np.cos(X[:, 2]), np.sin(X[:, 2])
        gx = X[:, 0, None] + cth[:, None] * occ[None, :, 0] - sth[:, None] * occ[None, :, 1]
        gy = X[:, 1, None] + sth[:, None] * occ[None, :, 0] + cth[:, None] * occ[None, :, 1]
        d = self.field.dist(np.stack([gx, gy], -1))
        return np.log((1 - c.grid_z_rand) * np.exp(-0.5 * (d / c.grid_sigma) ** 2) + c.grid_z_rand).mean(axis=1)

    def _grid_weight(self, occ, delta) -> None:
        """Multiply every hypothesis by the grid likelihood of the detections, averaged over its 7 sigma points."""
        c = self.cfg
        if not c.use_grid or self.field is None or occ is None or not len(occ) or not self.H:
            return
        if len(occ) > c.grid_points:
            occ = occ[self.rng.choice(len(occ), c.grid_points, replace=False, p=occ[:, 2] / occ[:, 2].sum())]
        g = c.grid_gain * evidence_weight(delta, c.corr_len)
        cur = [h for h in self.H if h.id == self.current]
        if cur:
            h = max(cur, key=lambda h: h.w)
            X = h.mu[None]
            cth, sth = np.cos(X[:, 2]), np.sin(X[:, 2])
            gx = X[:, 0, None] + cth[:, None] * occ[None, :, 0] - sth[:, None] * occ[None, :, 1]
            gy = X[:, 1, None] + sth[:, None] * occ[None, :, 0] + cth[:, None] * occ[None, :, 1]
            share = float((self.field.dist(np.stack([gx, gy], -1)) < c.fit_d).mean())
            self.fit += c.fit_ema * (share - self.fit)
        lls = []
        for h in self.H:
            L = np.linalg.cholesky(h.S + 1e-9 * np.eye(3)) * math.sqrt(3.0)
            X = np.vstack([h.mu, h.mu + L.T, h.mu - L.T])
            ll = self._grid_loglik(X, occ)
            lls.append(np.log(np.mean(np.exp(ll - ll.max()))) + ll.max())
        lls = np.array(lls) * g
        lls -= lls.max()
        for h, l in zip(self.H, lls):
            h.w *= math.exp(l)
        self._normalise()

    def _register(self, occ, delta) -> None:
        """EKF update of every (not too uncertain) hypothesis with its grid registration."""
        c = self.cfg
        if not c.use_registration or self.field is None or occ is None or len(occ) < 15:
            return
        pts = occ[:, :2]
        if len(pts) > c.grid_points:
            pts = pts[self.rng.choice(len(pts), c.grid_points, replace=False, p=occ[:, 2] / occ[:, 2].sum())]
        g = evidence_weight(delta, c.corr_len)
        for h in self.H:
            if math.sqrt(h.S[0, 0] + h.S[1, 1]) > c.reg_max_sigma:
                continue
            r = self.field.register(h.mu, pts)
            if r is None:
                continue
            z, info, _ = r
            info = info * g / c.reg_inflate
            # measurement z with information `info` (may be rank-deficient): information-form EKF update
            nu = z - h.mu
            nu[2] = wrap(nu[2])
            Si = np.linalg.inv(h.S)
            Sn = np.linalg.inv(Si + info)
            # gate: the correction must be plausible under the hypothesis' own uncertainty
            if float(nu @ info @ Sn @ Si @ nu) > c.reg_gate_d2:
                continue
            h.mu = h.mu + Sn @ info @ nu
            h.mu[2] = wrap(h.mu[2])
            h.S = 0.5 * (Sn + Sn.T)

    def update(self, cands: list[Candidate], delta=None, occ: np.ndarray | None = None) -> None:
        cfg = self.cfg
        g = cfg.view_gain * evidence_weight(delta, cfg.corr_len)
        self._register(occ, delta)
        if not cands or not self.H:
            if cands and not self.H:
                self.start_global(cands)
            self._grid_weight(occ, delta)
            return
        out: list[Hyp] = []
        gated_any = np.zeros(len(cands), bool)
        for h in self.H:
            # branch "no candidate is right" (clutter) and one branch for the best-supporting candidate: several
            # templates of one place (several passes in the map) must not add up their evidence
            out.append(Hyp(h.mu, h.S, h.w, h.id))
            best = None
            for j, k in enumerate(cands):
                R = self._R(k.z)
                Sinn = h.S + R
                nu = k.z - h.mu
                nu[2] = wrap(nu[2])
                d2 = float(nu @ np.linalg.solve(Sinn, nu))
                if d2 > cfg.gate_d2:
                    continue
                gated_any[j] = True
                lik = math.exp(-0.5 * d2) / math.sqrt((2 * math.pi) ** 3 * np.linalg.det(Sinn))
                odds = min(k.p / max(1 - k.p, 1e-6), cfg.view_odds_max)
                ratio = (odds * lik / cfg.background) ** g                    # vs. "candidate j is clutter"
                if best is None or ratio > best[0]:
                    best = (ratio, nu, Sinn)
            if best is not None:
                ratio, nu, Sinn = best
                K = h.S @ np.linalg.inv(Sinn)
                mu = h.mu + K @ nu
                mu[2] = wrap(mu[2])
                S = (np.eye(3) - K) @ h.S
                out.append(Hyp(mu, 0.5 * (S + S.T), h.w * ratio, h.id))
        # confident candidates outside every gate: new hypotheses (re-localization after a gap / wrong start), but
        # only while the reported hypothesis is in doubt
        cur = [h for h in self.H if h.id == self.current]
        hc = max(cur, key=lambda h: h.w) if cur else None
        self.in_doubt = (hc is None or hc.w < cfg.doubt_w or math.sqrt(hc.S[0, 0] + hc.S[1, 1]) > cfg.doubt_sigma
                         or self.fit < cfg.doubt_fit)
        for j, k in enumerate(cands):
            if self.in_doubt and not gated_any[j] and k.p_new >= cfg.spawn_p:
                out.append(Hyp(k.z.copy(), np.diag(np.square(cfg.spawn_sigma)), cfg.spawn_w * k.p_new, self._new_id()))
                self.n_spawned += 1
        self.H = out
        self._normalise()
        self._merge()
        self._grid_weight(occ, delta)
        self._prune()

    def _normalise(self) -> None:
        tot = sum(h.w for h in self.H)
        for h in self.H:
            h.w /= tot

    def _merge(self) -> None:
        H = sorted(self.H, key=lambda h: -h.w)
        merged: list[Hyp] = []
        used = np.zeros(len(H), bool)
        for i, h in enumerate(H):
            if used[i]:
                continue
            group = [h]
            used[i] = True
            for j in range(i + 1, len(H)):
                if used[j]:
                    continue
                nu = H[j].mu - h.mu
                nu[2] = wrap(nu[2])
                if float(nu @ np.linalg.solve(h.S, nu)) < self.cfg.merge_d2:
                    group.append(H[j])
                    used[j] = True
            if len(group) == 1:
                merged.append(h)
                continue
            w = np.array([g.w for g in group])
            W = w.sum()
            ref = group[0].mu
            D = np.array([np.r_[g.mu[:2] - ref[:2], wrap(g.mu[2] - ref[2])] for g in group])
            m = (w[:, None] * D).sum(0) / W
            S = sum(wi * (g.S + np.outer(d - m, d - m)) for wi, g, d in zip(w, group, D)) / W
            # the merged hypothesis keeps the identity of the reported one if it is in the group, else of the heaviest
            hid = self.current if any(g.id == self.current for g in group) else group[0].id
            merged.append(Hyp(np.r_[ref[:2] + m[:2], wrap(ref[2] + m[2])], S, W, hid))
        self.H = merged

    def _prune(self) -> None:
        H = sorted([h for h in self.H if h.w >= self.cfg.prune_w], key=lambda h: -h.w)[:self.cfg.max_hyp]
        self.H = H if H else sorted(self.H, key=lambda h: -h.w)[:1]
        self._normalise()

    def estimate(self) -> tuple[np.ndarray, np.ndarray, float]:
        """The reported hypothesis, with hysteresis: a branch of the same hypothesis (same id) may take over freely,
        another hypothesis only once it is switch_ratio x heavier (aliased places must keep winning to take over)."""
        best = max(self.H, key=lambda h: h.w)
        cur = [h for h in self.H if h.id == self.current]
        if cur:
            c = max(cur, key=lambda h: h.w)
            if best.id != c.id and best.w < self.cfg.switch_ratio * c.w:
                best = c
        if best.id != self.current:
            self.n_switches += self.current >= 0
            self.current = best.id
        return best.mu.copy(), best.S.copy(), float(best.w)
