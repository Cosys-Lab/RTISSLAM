"""Collect the SLAM results per imaging method and RW-CFAR post-processing (each with its own fitted evidence curve).

    python scripts/table_imaging.py [--prefix ev]   -> prints Markdown, writes paper/generated/tab_imaging.tex
Reads RTISSLAM_DATA/runs/<world>_<prefix>_<imaging>_<post>/metrics.json (scripts/sweep_imaging.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sondec.paths import data_dir  # noqa: E402

WORLDS = ["office", "warehouse", "atrium"]
IMAGING = [("das", "DAS"), ("dmas3cf", "DMAS3 + CF"), ("deconv", "Deconvolution")]
POSTS = [("none", "--"), ("ams", "1"), ("ams15", "1.5"), ("ams2", "2"), ("ams25", "2.5"), ("ams15s", "1.5 s"),
         ("ams2s", "2 s")]


def load(w, prefix, im, post):
    p = data_dir() / "runs" / f"{w}_{prefix}_{im}_{post}" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ev")
    a = ap.parse_args()
    rows = []
    print("| Imaging | RW-CFAR | s0 / slope | " + " | ".join(f"{w} final / online / max (links, false)" for w in WORLDS)
          + " | mean final |")
    print("|---|---|---|" + "---|" * (len(WORLDS) + 1))
    for im, iml in IMAGING:
        for post, g in POSTS:
            ms = [load(w, a.prefix, im, post) for w in WORLDS]
            if all(m is None for m in ms):
                continue
            cells, tex, fin = [], [], []
            for m in ms:
                if m is None:
                    cells.append("--"); tex += ["--", "--", "--"]
                    continue
                fin.append(m["ate_final_m"])
                cells.append(f"{m['ate_final_m']:.2f} / {m['ate_online_m']:.2f} / {m['max_err_final_m']:.2f} "
                             f"({m['links']}, {m['links_wrong_gt_gt_1m']})")
                tex += [f"{m['ate_final_m']:.2f}", f"{m['ate_online_m']:.2f}", f"{m['links_wrong_gt_gt_1m']}"]
            m0 = next(m for m in ms if m is not None)
            mean = sum(fin) / len(fin) if len(fin) == len(WORLDS) else float("nan")
            print(f"| {iml} | {g} | {m0['llr_mid']:.3f} / {m0['llr_slope']:.0f} | " + " | ".join(cells)
                  + f" | {mean:.2f} |")
            rows.append(f"{iml} & {g} & " + " & ".join(tex) + f" & {mean:.2f}" + r" \\")
    out = Path(__file__).resolve().parents[1] / "paper" / "generated" / f"tab_imaging_{a.prefix}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
