"""Fit the evidence curve of the place recognition per imaging / post-processing variant, the BatSLAM 2.0 way.

On the development drive, templates are laid every `spacing_m` of odometric travel; every node queries them exactly as
the TemplateStore does (best range-shifted correlation, top-k, templates within `exclude_recent_m` of recent travel
ignored), without a similarity floor. A candidate is correct if the ground-truth poses of query and anchor are within
0.6 m and 26 deg. The evidence curve of BatSLAM 2.0 is
    l(s) = clip(a (s - s_0), +-l_max),  P(correct | s_0) = 1/2,
where P(correct | s) is fitted by isotonic (monotone) regression, s_0 is where it reaches 1/2 and the slope a follows
from where its logit reaches +2 (the upper branch, as in BatSLAM 2.0); the candidate floor is where the evidence reaches its clip, s_min = s_0 - l_max / a (BatSLAM 2.0: 0.60 vs 0.70).

    python scripts/fit_evidence.py office --variants deconv:none dmas3cf:none dmas3cf:ams2s das:ams2
    python scripts/fit_evidence.py office --variants dmas3cf:ams2s --query-tag loc
Writes RTISSLAM_DATA/evidence/<drive>.json (one entry per variant, merged with what is there; drives other than
<drive>_map: <drive>_<tag>.json).

--query-tag fits the curve for LOCALIZATION sessions ("<variant>@loc"): templates from the mapping drive
<drive>_<--tag> (default map), queries from a second drive <drive>_<query-tag> (gen_drive.py --seed 1 --tag loc),
all templates allowed.
Localization runs in mapped space against a frozen map, so true revisits are common and a wrong link cannot bend the
map; the same recipe then gives a lower s_0 than the mapping curve.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtisslam.descriptor import EnergyscapeDescriptor, EnergyscapeDescriptorConfig  # noqa: E402
from rtisslam.imaging_post import POST  # noqa: E402
from rtisslam.system import default_batslam  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

TOL_M, TOL_RAD = 0.6, np.radians(26)


def shifted_sims(Dq: torch.Tensor, Dt: torch.Tensor, S: int) -> torch.Tensor:
    """[Nq, Nt] best range-shifted normalised correlation (as TemplateStore.similarities)."""
    def norm(X):
        X = X.flatten(1)
        return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    T = norm(Dt)
    B = Dq.shape[-1]
    best = torch.full((Dq.shape[0], Dt.shape[0]), -2.0, device=Dq.device)
    for s in range(-S, S + 1):
        Q = torch.zeros_like(Dq)
        if s >= 0:
            Q[..., :B - s] = Dq[..., s:]
        else:
            Q[..., -s:] = Dq[..., :B + s]
        best = torch.maximum(best, norm(Q) @ T.T)
    return best


def isotonic(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotone (non-decreasing) least-squares fit of y over s (pool adjacent violators); returns (s, P) sorted."""
    o = np.argsort(s)
    s, y = s[o], y[o]
    val, wt, cnt = [], [], []
    for v in y:
        val.append(float(v)); wt.append(1.0); cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            w = wt[-2] + wt[-1]
            val[-2] = (val[-2] * wt[-2] + val[-1] * wt[-1]) / w
            wt[-2] = w; cnt[-2] += cnt[-1]
            val.pop(); wt.pop(); cnt.pop()
    return s, np.repeat(val, cnt)


def crossing(s: np.ndarray, P: np.ndarray, level: float) -> float:
    """Similarity at which the (step-wise monotone) P first reaches level, at the centre of that block."""
    k = np.nonzero(P >= level)[0]
    if not len(k):
        return float("nan")
    blk = P == P[k[0]]
    lo = s[np.nonzero(blk)[0][0]]
    prev = s[np.nonzero(blk)[0][0] - 1] if np.nonzero(blk)[0][0] > 0 else lo
    return float(0.5 * (lo + prev))


def candidates(drive: str, meth: str, post: str, query_tag: str = "", tag: str = "map"):
    cfg = default_batslam()
    f = np.load(data_dir() / "drives" / f"{drive}_{tag}.npz")
    gt, odo, es = f["gt"], f["odom"], f[f"{meth}_db"]
    from dataclasses import replace
    from rtisslam.system import RTISSLAMConfig
    # the descriptor settings of the system (tuned per imaging method), with this energyscape
    de = EnergyscapeDescriptor(replace(RTISSLAMConfig().descriptor, method=meth), f["ranges"], f["azimuths"])
    D = np.stack([de.compute(POST[post](x)) for x in es]).astype(np.float32)
    if query_tag:                                                   # cross-session: queries from another drive
        fq = np.load(data_dir() / "drives" / f"{drive}_{query_tag}.npz")
        gq = fq["gt"]
        Dq = np.stack([de.compute(POST[post](x)) for x in fq[f"{meth}_db"]]).astype(np.float32)
    else:
        gq, Dq = gt, D
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(odo[:, :2], axis=0).T))]
    tpl = [0]
    for i in range(1, len(s)):
        if s[i] - s[tpl[-1]] >= cfg.templates.spacing_m:
            tpl.append(i)
    tpl = np.asarray(tpl)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Dt = torch.as_tensor(D[tpl], device=dev)
    out = []
    for q0 in range(0, len(Dq), 512):
        sim = shifted_sims(torch.as_tensor(Dq[q0:q0 + 512], device=dev), Dt, cfg.templates.max_shift_bins).cpu().numpy()
        for k, row in enumerate(sim):
            q = q0 + k
            # mapping: causal, only older templates; localization: the whole (frozen) map
            ok = np.ones(len(tpl), bool) if query_tag else s[q] - s[tpl] >= cfg.templates.exclude_recent_m
            idx = np.nonzero(ok)[0]
            if not len(idx):
                continue
            top = idx[np.argsort(-row[idx])[:cfg.templates.top_k]]
            for j in top:
                a = tpl[j]
                d = np.hypot(*(gq[q, :2] - gt[a, :2]))
                dth = abs(np.angle(np.exp(1j * (gq[q, 2] - gt[a, 2]))))
                out.append((row[j], float(d < TOL_M and dth < TOL_RAD), q))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drive")
    ap.add_argument("--variants", nargs="+", required=True, help="imaging:post, e.g. dmas3cf:ams2s")
    ap.add_argument("--clip", type=float, default=4.0)
    ap.add_argument("--query-tag", default="", help="localization curve: queries from drives/<drive>_<tag>.npz")
    ap.add_argument("--tag", default="map", help="mapping drive: drives/<drive>_<tag>.npz")
    a = ap.parse_args()
    path = data_dir() / "evidence" / (f"{a.drive}.json" if a.tag == "map" else f"{a.drive}_{a.tag}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(path.read_text()) if path.exists() else {}
    for v in a.variants:
        meth, post = v.split(":")
        C = candidates(a.drive, meth, post, a.query_tag, a.tag)
        S, y = C[:, 0], C[:, 1]
        ss, P = isotonic(S, y)
        # evidence curve of BatSLAM 2.0: l(s) = logit P(correct | s) on its upper branch, which is where true
        # revisits live: s_0 where P = 1/2, slope from where the logit reaches +2 (P = 0.88), as in
        # BatSLAM 2.0 ("+2 / +4 at s = 0.75 / 0.80"); the commit thresholds of the verifier are in these units
        mid = crossing(ss, P, 0.5)
        s20, s80 = crossing(ss, P, 1 / (1 + np.exp(2.0))), crossing(ss, P, 1 / (1 + np.exp(-2.0)))
        s98 = crossing(ss, P, 1 / (1 + np.exp(-4.0)))
        sl = 2.0 / (s80 - mid)        # (the +4 point is too coarse: the monotone fit often jumps to P = 1 there)
        floor = mid - a.clip / sl
        bins = np.arange(0.70, 1.0, 0.02)
        curve = {f"{b0:.2f}": [float(y[(S >= b0) & (S < b0 + 0.02)].mean()) if ((S >= b0) & (S < b0 + 0.02)).any() else None,
                               int(((S >= b0) & (S < b0 + 0.02)).sum())] for b0 in bins}
        # quality of the top-1 candidate: fraction of revisit nodes whose best candidate above s_0 is correct
        r = {"imaging": meth, "post": post, "s_logit_m2": s20, "s_logit_p2": s80, "s_logit_p4": s98, "llr_slope": sl, "llr_mid": mid, "min_similarity": floor,
             "n_candidates": int(len(y)), "prior_correct": float(y.mean()),
             "precision_above_mid": float(y[S >= mid].mean()) if (S >= mid).any() else None,
             "correct_above_mid": int(y[S >= mid].sum()), "wrong_above_mid": int((1 - y[S >= mid]).sum()),
             "binned": curve}
        v = f"{v}@loc" if a.query_tag else v
        res[v] = r
        print(f"{v:18s} s(-2) {s20:.3f} s(+2) {s80:.3f} s(+4) {s98:.3f}  slope {sl:6.1f}  mid {mid:.3f}  floor {floor:.3f}  prior {y.mean():.3f}  above mid: "
              f"{r['correct_above_mid']} correct / {r['wrong_above_mid']} wrong", flush=True)
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
