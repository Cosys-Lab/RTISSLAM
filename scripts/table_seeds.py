"""Results over the seeded 2-lap drives (scripts/run_final.py) and the slam_toolbox baseline, for the paper.

    python scripts/table_seeds.py   -> prints Markdown, writes paper/generated/tab_seeds_*.tex and seeds_summary.json
Per world and variant: mean and standard deviation over the seeds of the final and online ATE, and the false links
(links between places more than 1 m apart) summed over the seeds. Baseline: data/baseline/<drive>_slamtoolbox*.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sondec.paths import data_dir  # noqa: E402

WORLDS = ["office", "warehouse", "atrium"]
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = [("final_wheel", "Wheel odometry"), ("final_calibdr", r"\quad + AO calibration"),
            ("final_links_noreg_nocal", "BatSLAM 2.0 links"), ("final_links_nocal", r"\quad + registration"),
            ("final", r"\quad + registration, AO calibration (RTISSLAM)")]


def metrics(w, v, s):
    p = data_dir() / "runs" / f"{w}_dm_{v}_s{s}" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def baseline(w, s, stock=False):
    tag = "map2" if s == 0 else f"map2s{s}"
    p = ROOT / "data" / "baseline" / f"{w}_{tag}_slamtoolbox{'_stock' if stock else ''}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m


def ms(x):
    x = np.asarray([v for v in x if v is not None], float)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, len(x)) if len(x) else (np.nan, np.nan, 0)


def main():
    out = ROOT / "paper" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    summ = {}
    rows = []
    print("| Variant | " + " | ".join(f"{w} online / final (false)" for w in WORLDS) + " |")
    print("|---|" + "---|" * len(WORLDS))
    for v, lab in VARIANTS:
        cells, tex = [], []
        for w in WORLDS:
            M = [metrics(w, v, s) for s in SEEDS]
            on = ms([m["ate_online_m"] if m else None for m in M])
            fi = ms([m["ate_final_m"] if m else None for m in M])
            fl = sum(m["links_wrong_gt_gt_1m"] for m in M if m)
            summ[f"{w}/{v}"] = {"online": on, "final": fi, "false_links": fl, "n": fi[2]}
            cells.append(f"{on[0]:.2f}±{on[1]:.2f} / {fi[0]:.2f}±{fi[1]:.2f} ({fl}, n={fi[2]})")
            tex += [f"{fi[0]:.2f} $\\pm$ {fi[1]:.2f}"]
        print(f"| {lab.replace(chr(92) + 'quad ', '  ')} | " + " | ".join(cells) + " |")
        rows.append(f"{lab} & " + " & ".join(tex) + r" \\")
    for stock, lab in ((False, "slam\\_toolbox (tuned on the office)"), (True, "slam\\_toolbox (stock parameters)")):
        tex, cells = [], []
        for w in WORLDS:
            B = [baseline(w, s, stock) for s in SEEDS]
            fi = ms([b.get("ate_final_m") if b else None for b in B])
            summ[f"{w}/slamtoolbox{'_stock' if stock else ''}"] = {"final": fi}
            cells.append(f"{fi[0]:.2f}±{fi[1]:.2f} (n={fi[2]})")
            tex.append(f"{fi[0]:.2f} $\\pm$ {fi[1]:.2f}")
        print(f"| {lab} | " + " | ".join(cells) + " |")
        rows.append(f"{lab} & " + " & ".join(tex) + r" \\")
    (out / "tab_seeds.tex").write_text("\n".join(rows) + "\n")
    (out / "seeds_summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
