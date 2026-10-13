"""Place-recognition quality of the energyscape descriptor on a drive (ground truth for SCORING only).

All pairs (i, j) with > EXCL m of odometric travel between them: similarity = max over range shifts (+-3 bins) of the
normalised correlation (BatSLAM's TemplateStore). Same place: GT distance < 0.3 m and heading difference < 0.3 rad.
Prints, per method: AUC, recall of true pairs at the similarity where precision 99 %, and the evidence curve
P(same | s), which sets BatSLAM's llr_mid / llr_slope.

    python scripts/eval_descriptor.py office [--methods das dmas3cf deconv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtisslam.descriptor import EnergyscapeDescriptor, EnergyscapeDescriptorConfig  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

EXCL = 5.0


def sims(D: np.ndarray, S: int = 3) -> np.ndarray:
    """D [N, A, B] unit descriptors -> [N, N] best shifted correlation (GPU)."""
    X = torch.as_tensor(D, device="cuda")
    N, A, B = X.shape
    best = torch.full((N, N), -1.0, device="cuda")
    for s in range(-S, S + 1):
        Q = torch.zeros_like(X)
        if s >= 0:
            Q[..., :B - s] = X[..., s:]
        else:
            Q[..., -s:] = X[..., :B + s]
        Q = Q - Q.mean((1, 2), keepdim=True)
        Q = Q / Q.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None]
        best = torch.maximum(best, Q.flatten(1) @ X.flatten(1).T)
    return best.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drive")
    ap.add_argument("--methods", nargs="*", default=["das", "dmas2cf", "dmas3cf", "deconv"])
    ap.add_argument("--set", nargs="*", default=[], help="descriptor overrides key=value")
    ap.add_argument("--post", default="none", help="none | ams")
    a = ap.parse_args()
    f = np.load(data_dir() / "drives" / f"{a.drive}_map.npz")
    gt, odo = f["gt"], f["odom"]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(odo[:, :2], axis=0).T))]
    dxy = np.hypot(gt[:, None, 0] - gt[None, :, 0], gt[:, None, 1] - gt[None, :, 1])
    dth = np.abs(np.angle(np.exp(1j * (gt[:, None, 2] - gt[None, :, 2]))))
    far = np.abs(s[:, None] - s[None, :]) > EXCL
    iu = np.triu_indices(len(gt), 1)
    m = far[iu]
    same = ((dxy < 0.3) & (dth < 0.3))[iu][m]
    print(f"{a.drive}: {len(gt)} pulses, {m.sum()} pairs, {same.sum()} same-place pairs")
    for meth in a.methods:
        cfg = EnergyscapeDescriptorConfig(method=meth)
        for kv in a.set:
            k, v = kv.split("=")
            setattr(cfg, k, type(getattr(cfg, k))(v))
        de = EnergyscapeDescriptor(cfg, f["ranges"], f["azimuths"])
        from rtisslam.imaging_post import POST
        D = np.stack([de.compute(POST[a.post](x))[0] for x in f[f"{meth}_db"]])
        S = sims(D)[iu][m]
        order = np.argsort(-S)
        tp = np.cumsum(same[order]); prec = tp / np.arange(1, len(order) + 1)
        auc = (np.argsort(np.argsort(S))[same].mean() - (same.sum() - 1) / 2) / (~same).sum()
        k99 = np.nonzero(prec >= 0.99)[0]
        r99 = tp[k99[-1]] / same.sum() if len(k99) else 0.0
        s99 = S[order][k99[-1]] if len(k99) else np.nan
        bins = np.arange(0.3, 1.0, 0.05)
        curve = []
        for b0 in bins:
            sel = (S >= b0) & (S < b0 + 0.05)
            curve.append(f"{b0:.2f}:{same[sel].mean() if sel.any() else np.nan:.2f}({sel.sum()})")
        # similarity at which P(same | s) = 0.5 (sliding window of 0.02)
        grid = np.arange(0.7, 0.995, 0.005)
        ps = np.array([same[(S >= g - 0.01) & (S < g + 0.01)].mean() if ((S >= g - 0.01) & (S < g + 0.01)).sum() > 20 else np.nan for g in grid])
        k5 = np.nonzero(ps >= 0.5)[0]
        s50 = float(grid[k5[0]]) if len(k5) else float("nan")
        print(f"  {meth:8s} s(P=0.5) {s50:.3f}  AUC {auc:.3f}  recall@P99 {r99:.3f} (s >= {s99:.3f})  best-same-s median "
              f"{np.median(S[same]):.3f}\n    P(same|s): " + " ".join(curve))


if __name__ == "__main__":
    main()
