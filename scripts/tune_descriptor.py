"""Tune the energyscape descriptor for one imaging method on the development drive (office, 1 lap).

For every setting (RW-CFAR post-processing x descriptor parameters), scores what the sequence verifier sees:
templates every 0.3 m, the best candidate per node among templates older than 5 m of travel; a candidate is correct
if its anchor is within 0.6 m / 26 deg (as scripts/fit_evidence.py). Reports
    R@P95   recall of the best candidates at 95 % precision (sorted by similarity), over nodes that have a correct
            template available (revisit nodes)
    R@P99   the same at 99 %
    AUC     ranking of correct vs wrong best candidates
and the fitted evidence curve (s0, slope). Ground truth is used for scoring only.

    python scripts/tune_descriptor.py --imaging dmas3cf --posts none ams15 ams2 ams2s --grid n_sectors=30,60,90 gamma=0.33,0.5
Writes RTISSLAM_DATA/tuning/descriptor_<imaging>.json (merged).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtisslam.descriptor import EnergyscapeDescriptor, EnergyscapeDescriptorConfig  # noqa: E402
from rtisslam.imaging_post import POST  # noqa: E402
from rtisslam.system import default_batslam  # noqa: E402
from scripts.fit_evidence import crossing, isotonic, shifted_sims  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

TOL_M, TOL_RAD = 0.6, np.radians(26)


def best_candidates(D, gt, s, spacing, excl, shift):
    """Per node: (similarity of the best older template, correct?, correct template available?)."""
    tpl = [0]
    for i in range(1, len(s)):
        if s[i] - s[tpl[-1]] >= spacing:
            tpl.append(i)
    tpl = np.asarray(tpl)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Dt = torch.as_tensor(D[tpl], device=dev)
    dxy = np.hypot(gt[:, None, 0] - gt[None, tpl, 0], gt[:, None, 1] - gt[None, tpl, 1])
    dth = np.abs(np.angle(np.exp(1j * (gt[:, None, 2] - gt[None, tpl, 2]))))
    good = (dxy < TOL_M) & (dth < TOL_RAD)
    old = (s[:, None] - s[None, tpl]) >= excl
    out = []
    for q0 in range(0, len(D), 512):
        sim = shifted_sims(torch.as_tensor(D[q0:q0 + 512], device=dev), Dt, shift).cpu().numpy()
        for k, row in enumerate(sim):
            q = q0 + k
            idx = np.nonzero(old[q])[0]
            if not len(idx):
                continue
            j = idx[np.argmax(row[idx])]
            out.append((row[j], good[q, j], good[q, idx].any()))
    return np.asarray(out, float)


def recall_at(S, y, avail, prec):
    o = np.argsort(-S)
    tp = np.cumsum(y[o]); p = tp / np.arange(1, len(o) + 1)
    k = np.nonzero(p >= prec)[0]
    return float(tp[k[-1]] / max(avail.sum(), 1)) if len(k) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="office")
    ap.add_argument("--tag", default="map")
    ap.add_argument("--imaging", default="dmas3cf")
    ap.add_argument("--posts", nargs="+", default=["none"])
    ap.add_argument("--grid", nargs="*", default=[], help="descriptor field=v1,v2,...")
    a = ap.parse_args()
    f = np.load(data_dir() / "drives" / f"{a.drive}_{a.tag}.npz")
    gt, odo, es = f["gt"], f["odom"], f[f"{a.imaging}_db"]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(odo[:, :2], axis=0).T))]
    bc = default_batslam()
    names = [g.split("=")[0] for g in a.grid]
    values = [[type(getattr(EnergyscapeDescriptorConfig(), n))(v) for v in g.split("=")[1].split(",")]
              for n, g in zip(names, a.grid)]
    path = data_dir() / "tuning" / f"descriptor_{a.imaging}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(path.read_text()) if path.exists() else {}
    for post in a.posts:
        E = np.stack([POST[post](x) for x in es])
        for combo in itertools.product(*values) if values else [()]:
            cfg = EnergyscapeDescriptorConfig(method=a.imaging)
            for n, v in zip(names, combo):
                setattr(cfg, n, v)
            de = EnergyscapeDescriptor(cfg, f["ranges"], f["azimuths"])
            D = np.stack([de.compute(x) for x in E]).astype(np.float32)
            C = best_candidates(D, gt, s, bc.templates.spacing_m, bc.templates.exclude_recent_m,
                                bc.templates.max_shift_bins)
            S, y, av = C[:, 0], C[:, 1], C[:, 2] > 0
            pos, neg = S[y > 0], S[y == 0]
            auc = float((np.argsort(np.argsort(S))[y > 0].mean() - (len(pos) - 1) / 2) / max(len(neg), 1))
            ss, P = isotonic(S, y)
            s0 = crossing(ss, P, 0.5)
            slope = 2.0 / (crossing(ss, P, 1 / (1 + np.exp(-2.0))) - s0)
            key = f"{post}|" + ",".join(f"{n}={v}" for n, v in zip(names, combo))
            r = {"post": post, **{n: v for n, v in zip(names, combo)}, "r_p95": recall_at(S, y, av, 0.95),
                 "r_p99": recall_at(S, y, av, 0.99), "auc": auc, "s0": s0, "slope": slope,
                 "revisit_nodes": int(av.sum())}
            res[key] = r
            print(f"{key:45s} R@P95 {r['r_p95']:.3f}  R@P99 {r['r_p99']:.3f}  AUC {auc:.3f}  s0 {s0:.3f}  "
                  f"a {slope:.0f}", flush=True)
            path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
