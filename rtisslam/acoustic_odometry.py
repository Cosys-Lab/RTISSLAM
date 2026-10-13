"""Acoustic odometry from consecutive energyscapes (Steckel, Boen, Peremans, IEEE T-RO 29(1), 2013, Sec. IV-B).

For reflectors in the horizontal plane, a robot moving with v_lin, v_rot sees them move (their eq. 14)
    dr/dt = -cos(theta) v_lin,     dtheta/dt = sin(theta) / r v_lin - v_rot
and for small bearings and small dr/r (eqs. 16-18) two consecutive energyscapes in (r, theta) are related by a
translation
    dr ~= -v_lin dt,               dtheta ~= -v_rot dt.
The shift is found as the maximum of the 2D correlation of the two energyscapes, searched in a window around the
displacement predicted by the motor commands (here: the wheel odometry), and refined by a parabolic fit around the
maximum. Our energyscapes are already horizontal-plane (el = 0) slices of the eRTIS image.

Image per pulse: range-compensated level (dB, +40 log r) over R_MIN..R_MAX and |azimuth| <= AZ_MAX, clipped to the
top DR_DB below the view peak and scaled to 0..1, zero-mean. Extension (model = "flow"): instead of the translation approximation, each motion hypothesis (ds, dtheta) of the
robot (non-holonomic: along an arc; the sensor sits MOUNT ahead of the rotation centre) warps the previous image
exactly: every current cell (r, theta) is mapped to the previous sensor frame and the previous image is sampled
there. This removes the cos(theta) bias of the translation model for off-axis reflectors.

A sharpness measure (peak / median correlation in the
window) tells whether the estimate can be trusted (e.g. not in a featureless stretch).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class AcousticOdometryConfig:
    method: str = "deconv"
    model: str = "translation"  # "translation": the 2013 shift in (r, theta) | "flow": exact rigid-motion warp
    mount: tuple = (0.10, 0.0)  # sensor position in base_link (flow model)
    r_min: float = 0.5
    r_max: float = 4.5
    az_max_deg: float = 80.0
    dr_db: float = 25.0
    win_r: int = 4              # +- range cells around the prediction (2 cm cells)
    win_az: int = 5             # +- azimuth cells around the prediction (1 deg cells)
    min_sharp: float = 1.3      # peak / median correlation in the window
    min_peak: float = 0.15      # normalised correlation at the peak
    peak_fit: str = "parabola"  # sub-cell refinement: "parabola" (3-point) | "gauss" (parabola on log-correlation)


@dataclass
class AOEstimate:
    ds: float                   # forward displacement (m)
    dth: float                  # rotation (rad)
    peak: float
    sharp: float
    ok: bool


class AcousticOdometry:
    def __init__(self, cfg: AcousticOdometryConfig, ranges, azimuths):
        self.cfg = cfg
        r, az = np.asarray(ranges, float), np.asarray(azimuths, float)
        self.ri = np.nonzero((r >= cfg.r_min) & (r <= cfg.r_max))[0]
        self.ai = np.nonzero(np.abs(az) <= math.radians(cfg.az_max_deg))[0]
        self.dr = float(r[1] - r[0])
        self.daz = float(az[1] - az[0])
        self.tvg = 40 * np.log10(r[self.ri])[:, None]
        # pad so that shifted windows stay inside the stored image
        self.pr, self.pa = cfg.win_r + 12, cfg.win_az + 50
        self._r_all, self._az_all = r, az

    def image(self, es_db: np.ndarray) -> np.ndarray:
        """[R, D] dB -> padded, compressed image (range x azimuth) used for correlation."""
        c = self.cfg
        lo_r, hi_r = self.ri[0] - self.pr, self.ri[-1] + self.pr + 1
        lo_a, hi_a = self.ai[0] - self.pa, self.ai[-1] + self.pa + 1
        R, D = es_db.shape
        full = np.full((hi_r - lo_r, hi_a - lo_a), np.nan, np.float32)
        rs, re_ = max(lo_r, 0), min(hi_r, R)
        as_, ae = max(lo_a, 0), min(hi_a, D)
        v = es_db[rs:re_, as_:ae].astype(np.float32) + 40 * np.log10(self._r_all[rs:re_])[:, None]
        full[rs - lo_r:re_ - lo_r, as_ - lo_a:ae - lo_a] = v
        pk = np.nanmax(full[self.pr:-self.pr, self.pa:-self.pa])
        img = np.clip((full - (pk - c.dr_db)) / c.dr_db, 0, 1)
        return np.nan_to_num(img, nan=0.0)

    def _core(self, img, kr=0, ka=0):
        return img[self.pr + kr: img.shape[0] - self.pr + kr, self.pa + ka: img.shape[1] - self.pa + ka]

    def _warp(self, img_prev, ds, dth):
        """Previous image resampled on the current core grid for robot motion (ds, dth)."""
        if not hasattr(self, "_grid"):
            r = self._r_all[self.ri]; az = self._az_all[self.ai]
            Rg, Ag = np.meshgrid(r, az, indexing="ij")
            self._grid = (Rg * np.cos(Ag), Rg * np.sin(Ag))
        mx, my = self.cfg.mount
        # robot motion along an arc: chord at half the rotation
        tx, ty = ds * math.cos(dth / 2), ds * math.sin(dth / 2)
        c, s = math.cos(dth), math.sin(dth)
        X, Y = self._grid
        bx, by = mx + X, my + Y                                   # current sensor frame -> current base
        px, py = tx + c * bx - s * by, ty + s * bx + c * by       # -> previous base
        qx, qy = px - mx, py - my                                 # -> previous sensor frame
        rr, aa = np.hypot(qx, qy), np.arctan2(qy, qx)
        lo_r = self.ri[0] - self.pr
        lo_a = self.ai[0] - self.pa
        ir = (rr - self._r_all[0]) / self.dr - lo_r
        ia = (aa - self._az_all[0]) / self.daz - lo_a
        return map_coordinates(img_prev, [ir, ia], order=1, mode="constant", cval=0.0)

    def estimate_flow(self, img_prev, img_cur, pred_ds, pred_dth) -> AOEstimate:
        """Coarse-to-fine: every second hypothesis over the window, then the full-resolution 3 x 3 around the best
        (the correlation surface is smooth at the scale of one cell); same result as the full search at ~1/3 cost."""
        c = self.cfg
        B = self._core(img_cur)
        B = B - B.mean()
        nB = np.linalg.norm(B) + 1e-9
        cache = {}

        def corr(i, j):
            if (i, j) not in cache:
                A = self._warp(img_prev, pred_ds + i * self.dr, pred_dth + j * self.daz)
                A = A - A.mean()
                cache[(i, j)] = float((A * B).sum() / (nB * (np.linalg.norm(A) + 1e-9)))
            return cache[(i, j)]
        I = np.arange(-c.win_r, c.win_r + 1)
        J = np.arange(-c.win_az, c.win_az + 1)
        coarse = [(i, j) for i in I[::2] for j in J[::2]]
        bi, bj = max(coarse, key=lambda ij: corr(*ij))
        for _ in range(3):                                    # hill-climb on the full-resolution grid
            nb = [(i, j) for i in range(bi - 1, bi + 2) for j in range(bj - 1, bj + 2)
                  if -c.win_r <= i <= c.win_r and -c.win_az <= j <= c.win_az]
            ni, nj = max(nb, key=lambda ij: corr(*ij))
            if (ni, nj) == (bi, bj):
                break
            bi, bj = ni, nj
        C = np.full((len(I), len(J)), np.nan)
        for (i, j), v in cache.items():
            C[i + c.win_r, j + c.win_az] = v
        med = float(np.nanmedian(C))
        # local 3 x 3 for the parabolic fit (all evaluated by the hill-climb)
        sub = np.array([[corr(i, j) if (-c.win_r <= i <= c.win_r and -c.win_az <= j <= c.win_az) else -1.0
                         for j in (bj - 1, bj, bj + 1)] for i in (bi - 1, bi, bi + 1)])
        dss = pred_ds + self.dr * np.array([bi - 1, bi, bi + 1])
        dts = pred_dth + self.daz * np.array([bj - 1, bj, bj + 1])
        e = self._peak(sub, dss, dts)
        edge = abs(bi) == c.win_r or abs(bj) == c.win_az
        sharp = e.peak / max(med, 1e-3)
        return AOEstimate(e.ds, e.dth, e.peak, sharp, e.peak >= c.min_peak and sharp >= c.min_sharp and not edge)

    def _peak(self, C, dss, dts) -> AOEstimate:
        c = self.cfg
        i, j = np.unravel_index(np.argmax(C), C.shape)
        peak = float(C[i, j])
        sharp = peak / max(float(np.median(C)), 1e-3)

        def parab(y0, y1, y2):
            if c.peak_fit == "gauss":              # a broad, Gaussian-like peak: fit the parabola to its logarithm
                y0, y1, y2 = (np.log(max(y, 1e-6)) for y in (y0, y1, y2))
            den = y0 - 2 * y1 + y2
            return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        fr = parab(C[i - 1, j], C[i, j], C[i + 1, j]) if 0 < i < len(dss) - 1 else 0.0
        fa = parab(C[i, j - 1], C[i, j], C[i, j + 1]) if 0 < j < len(dts) - 1 else 0.0
        ds = dss[i] + fr * (dss[1] - dss[0])
        dth = dts[j] + fa * (dts[1] - dts[0])
        edge = len(dss) > 3 and (i in (0, len(dss) - 1) or j in (0, len(dts) - 1))
        return AOEstimate(float(ds), float(dth), peak, sharp, peak >= c.min_peak and sharp >= c.min_sharp and not edge)

    def estimate(self, img_prev: np.ndarray, img_cur: np.ndarray, pred_ds: float, pred_dth: float) -> AOEstimate:
        """Shift of img_cur relative to img_prev: reflectors at (r, th) in the previous image appear at
        (r - ds, th - dth) in the current one."""
        c = self.cfg
        if c.model == "flow":
            return self.estimate_flow(img_prev, img_cur, pred_ds, pred_dth)
        A = self._core(img_prev)
        A = A - A.mean()
        nA = np.linalg.norm(A) + 1e-9
        kr0 = int(round(-pred_ds / self.dr))
        ka0 = int(round(-pred_dth / self.daz))
        krs = np.arange(kr0 - c.win_r, kr0 + c.win_r + 1)
        kas = np.arange(ka0 - c.win_az, ka0 + c.win_az + 1)
        krs = krs[np.abs(krs) <= self.pr]
        kas = kas[np.abs(kas) <= self.pa]
        C = np.zeros((len(krs), len(kas)))
        for i, kr in enumerate(krs):
            for j, ka in enumerate(kas):
                B = self._core(img_cur, kr, ka)
                B = B - B.mean()
                C[i, j] = float((A * B).sum() / (nA * (np.linalg.norm(B) + 1e-9)))
        i, j = np.unravel_index(np.argmax(C), C.shape)
        peak = float(C[i, j])
        sharp = peak / max(float(np.median(C)), 1e-3)

        def parab(y0, y1, y2):
            if c.peak_fit == "gauss":              # a broad, Gaussian-like peak: fit the parabola to its logarithm
                y0, y1, y2 = (np.log(max(y, 1e-6)) for y in (y0, y1, y2))
            den = y0 - 2 * y1 + y2
            return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        fr = parab(C[i - 1, j], C[i, j], C[i + 1, j]) if 0 < i < len(krs) - 1 else 0.0
        fa = parab(C[i, j - 1], C[i, j], C[i, j + 1]) if 0 < j < len(kas) - 1 else 0.0
        kr, ka = krs[i] + fr, kas[j] + fa
        edge = i in (0, len(krs) - 1) or j in (0, len(kas) - 1)
        ok = peak >= c.min_peak and sharp >= c.min_sharp and not edge
        return AOEstimate(-kr * self.dr, -ka * self.daz, peak, sharp, ok)
