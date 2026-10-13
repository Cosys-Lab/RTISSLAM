"""Data, models and runs live outside the repository: %RTISSLAM_DATA% (default ~/RTISSLAM_data)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    d = Path(os.environ.get("RTISSLAM_DATA", Path.home() / "RTISSLAM_data"))
    d.mkdir(parents=True, exist_ok=True)
    return d
