"""Tables of the final runs and the ablation (scripts/run_final.py) for the paper.

    python scripts/table_final.py     -> prints Markdown, writes paper/generated/tab_slam.tex and tab_ablation.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sondec.paths import data_dir  # noqa: E402

WORLDS = ["office", "warehouse", "atrium"]
ABLATION = [("final_wheel", "Wheel odometry"),
            ("final_aofactors", r"\quad + AO factors"),
            ("final_calibdr", r"\quad + AO calibration"),
            (None, None),
            ("final_links_noreg_nocal", "BatSLAM 2.0 links"),
            ("final_links_nocal", r"\quad + registration"),
            ("final_links_noreg", r"\quad + AO calibration"),
            ("final_full_aofactors", r"\quad + reg., AO calib.\ and factors"),
            ("final", r"\quad + reg., AO calib.\ (RTISSLAM)")]


def load(w, v):
    p = data_dir() / "runs" / f"{w}_{v}" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out = Path(__file__).resolve().parents[1] / "paper" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    print("| World | Nodes | Distance | Odometry | Online | Final | Max | Links (false) | kappa_s, kappa_theta |")
    print("|---|---|---|---|---|---|---|---|---|")
    for w in WORLDS:
        m = load(w, "final")
        if m is None:
            continue
        k = m["wheel_calibration"]
        print(f"| {w} | {m['nodes']} | | {m['ate_odometry_m']:.2f} | {m['ate_online_m']:.2f} | {m['ate_final_m']:.2f} | "
              f"{m['max_err_final_m']:.2f} | {m['links']} ({m['links_wrong_gt_gt_1m']}) | {k[0]:.3f}, {k[1]:.3f} |")
        rows.append(f"{w.title()} & {m['nodes']} & {m['ate_odometry_m']:.2f} & {m['ate_online_m']:.2f} & "
                    rf"\textbf{{{m['ate_final_m']:.2f}}} & {m['max_err_final_m']:.2f} & {m['links']} ({m['links_wrong_gt_gt_1m']}) \\")
    (out / "tab_slam.tex").write_text("\n".join(rows) + "\n")
    rows = []
    print("\n| Variant | " + " | ".join(f"{w} online / final" for w in WORLDS) + " |")
    print("|---|" + "---|" * len(WORLDS))
    for v, lab in ABLATION:
        if v is None:
            rows.append(r"\midrule")
            continue
        cells, tex = [], []
        for w in WORLDS:
            m = load(w, v)
            if m is None:
                cells.append("--"); tex += ["--", "--"]
                continue
            cells.append(f"{m['ate_online_m']:.2f} / {m['ate_final_m']:.2f}")
            b = v == "final"
            f = (lambda x: rf"\textbf{{{x:.2f}}}") if b else (lambda x: f"{x:.2f}")
            tex += [f(m["ate_online_m"]), f(m["ate_final_m"])]
        print(f"| {lab.replace(chr(92) + 'quad ', '  ')} | " + " | ".join(cells) + " |")
        rows.append(f"{lab} & " + " & ".join(tex) + r" \\")
    (out / "tab_ablation.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
