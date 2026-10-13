"""Layered acoustic-flow navigation controller (Jansen, Laurijssen, Steckel, "Real-Time Sonar Fusion for Layered
Navigation Controller", Sensors 22(9):3109, 2022), single eRTIS, as the local controller under a global path.

Subsumption, highest priority first; each layer is a masked sum over the energyscape E(r, theta) (sensor frame),
the masks are ternary (+1 left of the platform, -1 right, 0 outside the zone), and only the closest reflection per
azimuth counts (multipath affects range, not the angular decision):
    CA   collision avoidance: half disc in front (R_CA). Trigger sum E M_CA > T_CA -> turn on the spot at W_CA
         away from the side with more energy; V = 0, reversing at V_REV after 4 consecutive triggers.
    OA   obstacle avoidance: a lane ahead (|y| < Y_OA, x < X_OA). Trigger > T_OA ->
         w = w_i + lambda_OA sum(E M / r^2) / sum(E |M|) (steering away), V = V_i (1 - mu_OA sum E |M| / norm).
    AFF  acoustic flow following: for linear motion the flow lines are lines parallel to the heading at lateral
         distance d (C = r sin theta); Gamma(d) = mean energy along each line. One strong wall: keep its distance
         (w = w_i + lambda_AFF (d_s - d_p)); two walls left and right: centre (w = w_i + lambda_AFF (d_l - d_r)).
    RCF  reactive corridor following: peripheral zones left and right; w = w_i + lambda_RCF sum(E M / r^2) / sum(E |M|).
Layers that do not trigger pass (V_i, w_i) through. Gains as in the paper's real-robot experiments (0.1), signs
chosen so that every correction steers AWAY from energy (the paper's sign conventions are not given explicitly).

Energy: the energyscape (dB) range-compensated, above the view's noise floor + DET_DB, as linear excess (a
detection map): the thresholds are then in "detection energy" units, independent of the absolute sensor gain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AFCConfig:
    method: str = "dmas3cf:ams2s"   # energyscape ("imaging:post") the controller reacts to
    det_db: float = 22.0
    peak_db: float = 30.0
    mount: tuple = (0.10, 0.0)
    r_robot: float = 0.25
    # collision avoidance
    r_ca: float = 0.42          # m from the sensor
    t_ca: float = 0.5
    w_ca: float = 0.5
    v_rev: float = -0.1
    # obstacle avoidance
    x_oa: float = 1.1
    y_oa: float = 0.36
    t_oa: float = 2.0          # 0.5 fired on isolated speckle of the deconvolved energyscape (2026-10-12)
    lambda_oa: float = 0.1
    mu_oa: float = 0.5
    # acoustic flow following
    use_aff: bool = True
    aff_x: tuple = (0.3, 2.5)
    aff_d: tuple = (0.4, 1.6)   # |lateral distance| range of the flow lines
    aff_dd: float = 0.05
    t_aff_single: float = 0.15
    t_aff_corr: float = 0.08
    lambda_aff: float = 0.1
    # reactive corridor following
    use_rcf: bool = True
    rcf_x: tuple = (0.0, 2.0)
    rcf_y: tuple = (0.45, 1.6)
    t_rcf: float = 0.5
    lambda_rcf: float = 0.1
    w_max: float = 0.8


class AcousticFlowController:
    LAYERS = ("CA", "OA", "AFF", "RCF", "PATH")

    def __init__(self, cfg: AFCConfig, ranges, azimuths):
        self.cfg = cfg
        r, az = np.asarray(ranges, float), np.asarray(azimuths, float)
        self.r, self.az = r, az
        self.tvg = 40 * np.log10(r)[:, None]
        R, A = np.meshgrid(r, az, indexing="ij")
        self.R = R
        self.x = R * np.cos(A)
        self.y = R * np.sin(A)
        self.side = np.sign(self.y)                                    # +1 left, -1 right (ternary mask sign)
        c = cfg
        self.m_ca = (R <= c.r_ca) & (self.x >= 0)
        self.m_oa = (self.x > 0) & (self.x <= c.x_oa) & (np.abs(self.y) <= c.y_oa) & ~self.m_ca
        self.m_rcf = (self.x >= c.rcf_x[0]) & (self.x <= c.rcf_x[1]) & (np.abs(self.y) >= c.rcf_y[0]) & (np.abs(self.y) <= c.rcf_y[1])
        self.d_lines = np.arange(c.aff_d[0], c.aff_d[1] + 1e-9, c.aff_dd)
        self.m_aff_x = (self.x >= c.aff_x[0]) & (self.x <= c.aff_x[1])
        self._ca_count = 0
        self._d_prev = None
        self.last_layer = "PATH"
        self.debug = {}

    def energy(self, es_db: np.ndarray) -> np.ndarray:
        """Detection energy [R, D]: linear excess above the detection threshold, closest reflection per azimuth
        only (plus 0.3 m behind it, the extent of one reflector's echo)."""
        c = self.cfg
        v = np.asarray(es_db, np.float32) + self.tvg
        thr = max(float(np.median(v)) + c.det_db, float(v.max()) - c.peak_db)
        E = np.clip((v - thr) / 10.0, 0, None)
        hit = E > 0
        first = np.where(hit.any(0), np.argmax(hit, axis=0), len(self.r))
        keep = self.r[:, None] <= self.r[np.minimum(first, len(self.r) - 1)][None, :] + 0.3
        return E * keep

    def command(self, es_db: np.ndarray, v_i: float, w_i: float) -> tuple[float, float, str]:
        c = self.cfg
        E = self.energy(es_db)
        # CA
        s_ca = float(E[self.m_ca].sum())
        if s_ca > c.t_ca:
            self._ca_count += 1
            lr = float((E * self.side)[self.m_ca].sum())
            w = -math.copysign(c.w_ca, lr if abs(lr) > 1e-6 else -w_i if abs(w_i) > 1e-6 else 1.0)
            v = c.v_rev if self._ca_count >= 4 else 0.0
            return self._out(v, w, "CA", s_ca=s_ca)
        self._ca_count = 0
        # OA
        s_oa = float(E[self.m_oa].sum())
        if s_oa > c.t_oa and v_i > 0.02:                 # turning on the spot cannot hit anything: no OA then
            M = self.side[self.m_oa]
            num = float((E[self.m_oa] * M / self.R[self.m_oa] ** 2).sum())
            w = w_i - c.lambda_oa * num / max(s_oa, 1e-9) * 3.0
            v = v_i * max(0.25, 1 - c.mu_oa * s_oa / (s_oa + 5.0))
            return self._out(v, w, "OA", s_oa=s_oa)
        # AFF
        if c.use_aff:
            G = []
            for side in (1, -1):
                g = []
                for d in self.d_lines:
                    m = self.m_aff_x & (np.abs(self.y - side * d) <= c.aff_dd / 2)
                    g.append(float(E[m].sum() / max(m.sum(), 1)))
                G.append(np.array(g))
            GL, GR = G
            iL, iR = int(np.argmax(GL)), int(np.argmax(GR))
            dl, dr = self.d_lines[iL], self.d_lines[iR]
            if GL[iL] > c.t_aff_corr and GR[iR] > c.t_aff_corr:
                self._d_prev = None
                return self._out(v_i, w_i + c.lambda_aff * (dl - dr) * 2.0, "AFF", d_l=dl, d_r=dr)
            if max(GL[iL], GR[iR]) > c.t_aff_single:
                d = dl if GL[iL] >= GR[iR] else -dr
                w = w_i
                if self._d_prev is not None and np.sign(d) == np.sign(self._d_prev):
                    w = w_i + c.lambda_aff * (d - self._d_prev) * 10.0
                self._d_prev = d
                return self._out(v_i, w, "AFF", d=d)
            self._d_prev = None
        # RCF
        if c.use_rcf:
            s = float(E[self.m_rcf].sum())
            if s > c.t_rcf:
                M = self.side[self.m_rcf]
                num = float((E[self.m_rcf] * M / self.R[self.m_rcf] ** 2).sum())
                return self._out(v_i, w_i - c.lambda_rcf * num / max(s, 1e-9), "RCF", s_rcf=s)
        return self._out(v_i, w_i, "PATH")

    def _out(self, v, w, layer, **dbg):
        self.last_layer = layer
        self.debug = dbg
        return float(v), float(np.clip(w, -self.cfg.w_max, self.cfg.w_max)), layer
