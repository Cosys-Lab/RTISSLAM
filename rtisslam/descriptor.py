"""Energyscape -> BatSLAM local-view descriptor (Jan: "as descriptor I have used just an energyscape").

Settings chosen on the office mapping drive (scripts/eval_descriptor.py, 2026-10-12): the deconvolved energyscape,
60 sectors, gamma 0.33: AUC 0.983, P(same place | s) = 0.5 at s = 0.915, 0.99 above s = 0.95 (DMAS3 + CF close).

The eRTIS energyscape [R x D] (dB, 2 cm range cells from 0.2 m, 181 azimuths) is turned into a small image like
BatSLAM's cochleogram, [1, A, B] (azimuth sectors x range bins; range LAST so BatSLAM's range-shift search works):
    linear power, range gain r^(2 k) (k = range_gain_exponent on amplitude), noise floor removed,
    pooled into A sectors (mean power) and 6 cm range bins, blurred (1 bin in range, 1 sector in azimuth),
    compressed (power^(gamma / 2): amplitude^gamma relative to the view peak), centred and unit-normalised.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class EnergyscapeDescriptorConfig:
    method: str = "deconv"            # which energyscape: das | dmas2 | dmas2cf | dmas3cf | deconv
    n_sectors: int = 60               # azimuth sectors over -90..90 deg (3 deg each)
    range_bin: float = 0.06           # m
    max_range: float = 4.5            # m
    range_gain_exponent: float = 1.0  # amplitude x r^k
    floor_quantile: float = 0.5       # noise floor = this quantile of the (range-compensated) view
    gamma: float = 0.33               # amplitude^gamma compression
    blur_range_bins: float = 0.7
    blur_sectors: float = 0.5


class EnergyscapeDescriptor:
    def __init__(self, cfg: EnergyscapeDescriptorConfig, ranges: np.ndarray, azimuths: np.ndarray):
        self.cfg = cfg
        self.r = np.asarray(ranges, float)
        keep = self.r <= cfg.max_range
        self.keep = keep
        r = self.r[keep]
        self.n_bins = int(np.ceil((r[-1] - r[0] + 1e-9) / cfg.range_bin))
        self.bin_idx = np.minimum(((r - r[0]) / cfg.range_bin).astype(int), self.n_bins - 1)
        self.sec_idx = np.minimum(((np.asarray(azimuths) - azimuths[0]) / (azimuths[-1] - azimuths[0] + 1e-9)
                                   * cfg.n_sectors).astype(int), cfg.n_sectors - 1)
        self.gain = r ** (2 * cfg.range_gain_exponent)
        self.range_centers = r[0] + (np.arange(self.n_bins) + 0.5) * cfg.range_bin
        # pooling matrices (mean)
        self.Pr = np.zeros((self.n_bins, len(r)))
        self.Pr[self.bin_idx, np.arange(len(r))] = 1
        self.Pr /= self.Pr.sum(1, keepdims=True)
        self.Pa = np.zeros((len(azimuths), cfg.n_sectors))
        self.Pa[np.arange(len(azimuths)), self.sec_idx] = 1
        self.Pa /= self.Pa.sum(0, keepdims=True)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, self.cfg.n_sectors, self.n_bins)

    def compute(self, es_db: np.ndarray) -> np.ndarray:
        P = np.power(10.0, np.asarray(es_db, np.float64)[self.keep] / 10) * self.gain[:, None]      # [R, D]
        P = np.maximum(P - np.quantile(P, self.cfg.floor_quantile), 0)
        V = (self.Pr @ P @ self.Pa).T                                                               # [A, B]
        V = gaussian_filter(V, (self.cfg.blur_sectors, self.cfg.blur_range_bins), mode="constant")
        V = (V / max(V.max(), 1e-30)) ** (self.cfg.gamma / 2)
        V = V - V.mean()
        return (V / max(np.linalg.norm(V), 1e-12))[None].astype(np.float32)
