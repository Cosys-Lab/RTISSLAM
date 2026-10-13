"""Copy a saved map with the localization evidence curve of its descriptor (fit_evidence.py --query-tag) in its config.

    python scripts/with_loc_curve.py office_final --evidence office_map2 [--out office_final_loc]
Reads runs/<map>/map.npz and evidence/<evidence>.json["<descriptor key>@loc"]; writes runs/<out>/map.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sondec.paths import data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with np.load(data_dir() / "runs" / a.map / "map.npz") as z:
        d = {k: z[k] for k in z.files}
    cfg = json.loads(str(d["config"]))
    ev = json.loads((data_dir() / "evidence" / f"{a.evidence}.json").read_text())[cfg["descriptor"]["method"] + "@loc"]
    cfg["localization"] = {k: ev[k] for k in ("llr_mid", "llr_slope", "min_similarity")}
    d["config"] = np.array(json.dumps(cfg))
    out = data_dir() / "runs" / (a.out or f"{a.map}_loc")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "map.npz", **d)
    print(out / "map.npz", cfg["localization"])


if __name__ == "__main__":
    main()
