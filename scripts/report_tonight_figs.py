"""Figures and tables of the localization chapter of report/tonight (2026-10-12 night).

    python scripts/report_tonight_figs.py
Reads RTISSLAM_DATA/{evidence,loc_eval,nav} and report/tonight/data/*.json (Windows session: final maps); writes
report/tonight/figs/loc_*.pdf and report/tonight/generated/*.tex.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rtisnav.costmap import planning_grid  # noqa: E402
from rtisslam.system import RTISSLAM  # noqa: E402
from rtissim.worlds import PLANS  # noqa: E402
from sondec.paths import data_dir  # noqa: E402

OUT = ROOT / "report" / "tonight"
FIG, GEN = OUT / "figs", OUT / "generated"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})
COL = {"graph": "tab:blue", "mix": "#b10091", "pf": "tab:orange"}
NAME = {"graph": "pose graph", "mix": "Gaussian mixture", "pf": "particle filter"}


def fig_curves():
    ev = json.loads((data_dir() / "evidence" / "office.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    s = np.linspace(0.6, 1.0, 200)
    for key, lab, c in (("dmas3cf:ams2s", "mapping (same drive)", "tab:blue"),
                        ("dmas3cf:ams2s@loc", "localization (second drive)", "#b10091")):
        e = ev[key]
        b = [(float(k), v[0], v[1]) for k, v in e["binned"].items() if v[0] is not None and v[1] >= 20]
        ax[0].plot([x + 0.01 for x, _, _ in b], [p for _, p, _ in b], "o", ms=3, color=c)
        l = np.clip(e["llr_slope"] * (s - e["llr_mid"]), -4, 4)
        ax[0].plot(s, 1 / (1 + np.exp(-l)), color=c, label=f"{lab}: $s_0$ {e['llr_mid']:.3f}, slope {e['llr_slope']:.0f}")
        ax[1].plot(s, l, color=c)
        ax[1].axvline(e["min_similarity"], color=c, ls=":")
    ax[0].set_xlabel("similarity $s$"); ax[0].set_ylabel("P(correct | s), top-5 candidates")
    ax[0].legend(fontsize=7, loc="upper left"); ax[0].set_ylim(0, 1.02); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("similarity $s$"); ax[1].set_ylabel("evidence $\\ell(s)$"); ax[1].grid(alpha=0.3)
    ax[1].set_xlim(0.6, 1.0)
    fig.tight_layout()
    fig.savefig(FIG / "loc_curves.pdf"); plt.close(fig)


def fig_replay():
    d = data_dir() / "loc_eval"
    runs = [("graph", "office_loc_v1g_graph_known.npz"), ("mix", "office_loc_v6_mix_known.npz"),
            ("pf", "office_loc_v4_pf_known.npz")]
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
    for k, f in runs:
        z = np.load(d / f)
        ax[0].plot(z["at"], z["e_rel"], color=COL[k], lw=0.8, label=NAME[k])
        ax[1].plot(z["at"], z["e_gt"], color=COL[k], lw=0.8)
    ax[0].set_ylabel("error rel. to map (m)"); ax[1].set_ylabel("error to truth (m)")
    ax[1].set_xlabel("pulse of the second drive"); ax[0].legend(fontsize=7, ncol=3)
    for a in ax:
        a.set_ylim(0, 1.2); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "loc_replay.pdf"); plt.close(fig)


def fig_nav():
    slam = RTISSLAM.load(data_dir() / "runs" / "office" / "map.npz")
    tern, L, org, res = planning_grid(slam)
    img = np.where(tern == 100, 0.15, np.where(tern == 0, 1.0, 0.82)).T
    ext = (org[0], org[0] + tern.shape[0] * res, org[1], org[1] + tern.shape[1] * res)
    runs = [("graph", "office_nav_graph"), ("pf", "office_nav_pf3"), ("mix", "office_nav_mix4")]
    runs = [(k, r) for k, r in runs if (data_dir() / "nav" / f"{r}.npz").exists()]
    fig, ax = plt.subplots(1, len(runs), figsize=(7.0, 2.2))
    for a, (k, r) in zip(np.atleast_1d(ax), runs):
        z = np.load(data_dir() / "nav" / f"{r}.npz")
        m = json.loads((data_dir() / "nav" / f"{r}.json").read_text())
        a.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
        for w in PLANS["office"]().walls:
            a.plot([w[0], w[2]], [w[1], w[3]], color="tab:blue", lw=0.4, alpha=0.5)
        a.plot(z["gt"][:, 0], z["gt"][:, 1], color="0.3", lw=0.7)
        a.plot(z["est"][:, 0], z["est"][:, 1], color=COL[k], lw=0.7)
        a.plot(z["goals"][:, 0], z["goals"][:, 1], "*", color="tab:green", ms=7)
        a.set_title(f"{NAME[k]}\n{m['reached']}/{m['goals']} goals, {m['reached_within_0p5m_gt']} within 0.5 m\n"
                    f"loc. error {m['loc_err_mean_m']:.2f} m (max {m['loc_err_max_m']:.2f})", fontsize=7)
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    fig.tight_layout(); fig.savefig(FIG / "loc_nav.pdf"); plt.close(fig)


KID = [("office", "DMAS3-CF", "mix", 3), ("office", "DMAS3-CF", "mix", 8), ("office", "DMAS3-CF", "pf", 8),
       ("warehouse", "final", "mix", 8), ("warehouse", "final", "pf", 8), ("atrium", "final", "mix", 8),
       ("atrium", "final", "pf", 8)]


def fig_kidnap():
    """Kidnapped closed-loop runs: error to truth over the travelled distance."""
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.0))
    sty = {"office": "-", "warehouse": "--", "atrium": ":"}
    for w, mp, k, km in KID:
        p = data_dir() / "nav" / f"{w}_kid_{w}_{k}_{km}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        e = np.hypot(*(z["est"][:, :2] - z["gt"][:, :2]).T)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(z["gt"][:, :2], axis=0).T))]
        ax.plot(d, e, sty[w], color=COL[k], lw=0.9, label=f"{w} ({mp}), {NAME[k]}, {km:.0f} m off")
    ax.set_yscale("log"); ax.set_ylim(0.02, 30); ax.set_xlim(0, None)
    ax.axhline(0.5, color="0.5", lw=0.6)
    ax.set_xlabel("distance driven (m)"); ax.set_ylabel("error to truth (m)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(FIG / "loc_kidnap.pdf", bbox_inches="tight"); plt.close(fig)


def tables():
    GEN.mkdir(parents=True, exist_ok=True)
    d = data_dir() / "loc_eval"
    rows = []
    for k, f, tag in (("graph", "office_loc_v1g.json", "graph"), ("mix", "office_loc_v6.json", "mix"),
                      ("pf", "office_loc_v4.json", "pf")):
        r = json.loads((d / f).read_text())
        for st in ("known", "kidnapped", "global"):
            e = r.get(f"{tag}/{st}")
            if e is None:
                rows.append(f"{NAME[k]} & {st} & \\multicolumn{{6}}{{c}}{{not applicable}} \\\\")
                continue
            ff = "--" if e["first_fix_pulse"] < 0 else f"{e['first_fix_m']:.1f}"
            rows.append(f"{NAME[k]} & {st} & {e['err_rel_mean']:.3f} & {e['err_rel_max']:.2f} & "
                        f"{100 * e['share_rel_below_0p5']:.1f} & {ff} & {e['nees_xy']:.1f} & {e['ms_per_pulse']:.0f} \\\\")
    (GEN / "tab_replay.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for k, r in (("graph", "office_nav_graph"), ("pf", "office_nav_pf3"), ("mix", "office_nav_mix4"),
                 ("mix", "office_nav_mix2")):
        p = data_dir() / "nav" / f"{r}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        note = {"office_nav_mix2": " (before grid evidence)"}.get(r, "")
        rows.append(f"{NAME[k]}{note} & {m['reached']}/{m['goals']} & {m['reached_within_0p5m_gt']} & "
                    f"{m['loc_err_mean_m']:.2f} & {m['loc_err_max_m']:.2f} & {m['contact_events']} & {m['distance_m']:.0f} \\\\")
    (GEN / "tab_nav.tex").write_text("\n".join(rows) + "\n")
    # Windows: final maps
    rows = []
    for w in ("office", "warehouse", "atrium"):
        p = OUT / "data" / f"loc_{w}_final.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for k in ("graph", "mix", "pf"):
            for st in ("known", "kidnapped", "global"):
                e = r.get(f"{k}/{st}")
                if e is None:
                    continue
                ff = "--" if e["first_fix_pulse"] < 0 else f"{e['first_fix_m']:.1f}"
                rows.append(f"{w} & {NAME[k]} & {st} & {e['err_rel_mean']:.3f} & {e['err_rel_max']:.2f} & "
                            f"{100 * e['share_rel_below_0p5']:.1f} & {ff} & {e['ms_per_pulse']:.0f} \\\\")
    (GEN / "tab_final_maps.tex").write_text("\n".join(rows) + "\n" if rows else "% not available yet\n")
    rows = []
    for w in ("office", "warehouse", "atrium"):
        for k in ("graph", "mix", "pf"):
            p = OUT / "data" / f"nav_{w}_{k}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())
            rows.append(f"{w} & {NAME[k]} & {m['reached']}/{m['goals']} & {m['reached_within_0p5m_gt']} & "
                        f"{m['loc_err_mean_m']:.2f} & {m['loc_err_max_m']:.2f} & {m['contact_events']} \\\\")
    (GEN / "tab_final_nav.tex").write_text("\n".join(rows) + "\n" if rows else "% not available yet\n")
    rows = []
    for w, mp, k, km in KID:
        p = data_dir() / "nav" / f"{w}_kid_{w}_{k}_{km}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        ff = "never" if m.get("first_fix_m") is None else f"{m['first_fix_m']:.1f}"
        rows.append(f"{w} & {mp} & {NAME[k]} & {km} & {ff} & {100 * m.get('share_below_0p5_after_fix', 0):.0f} & "
                    f"{m['reached']}/{m['goals']} & {m['reached_within_0p5m_gt']} & {m['contact_events']} \\\\")
    rows2 = []
    for w, mp, k, r in (("office", "DMAS3-CF", "mix", "office_nav_mix4"), ("office", "DMAS3-CF", "mix", "office_reg_office_mix"),
                        ("office", "DMAS3-CF", "pf", "office_nav_pf3"),
                        ("warehouse", "final", "mix", "warehouse_reg_warehouse_mix"), ("warehouse", "final", "pf", "warehouse_reg_warehouse_pf"),
                        ("atrium", "final", "mix", "atrium_reg_atrium_mix")):
        p = data_dir() / "nav" / f"{r}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        lab = NAME[k] + (" + registration" if "reg_" in r and k == "mix" else "")
        rows2.append(f"{w} & {mp} & {lab} & {m['reached']}/{m['goals']} & {m['reached_within_0p5m_gt']} & "
                     f"{m['loc_err_mean_m']:.2f} & {m['loc_err_max_m']:.2f} \\\\")
    (GEN / "tab_reg.tex").write_text("\n".join(rows2) + "\n")
    (GEN / "tab_kidnap.tex").write_text("\n".join(rows) + "\n" if rows else "% not available yet\n")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_curves()
    fig_replay()
    fig_nav()
    fig_kidnap()
    tables()
    print("ok")
