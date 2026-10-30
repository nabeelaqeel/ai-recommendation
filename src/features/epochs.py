"""Sleep-epoch grid construction and signal/time alignment.

Epoch boundaries come from each night's own scored-events CSV (`Start Time` and
30-s duration), not from a reconstructed clock grid. This matters for the one
night with annotation clock drift (DS_41-N2, ~0.29 s/epoch): absolute start
times stay aligned with the EDF, while epoch indices do not.
"""

from __future__ import annotations

import pandas as pd

from .config import FeatureConfig

SLEEP_STAGES = ("N1", "N2", "N3", "REM")
ALL_STAGES = SLEEP_STAGES + ("Wake",)


def load_stage_table(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Start Time", "End Time"])
    if "Event.1" in df.columns:
        df = df.drop(columns=["Event.1"])
    stages = df[df["Event"].isin(ALL_STAGES)].copy()
    stages["Start Epoch"] = stages["Start Epoch"].astype(int)
    stages = stages.sort_values("Start Time").drop_duplicates(subset=["Start Epoch"], keep="first")
    return stages


def sleep_epochs(csv_path, edf_start: pd.Timestamp, cfg: FeatureConfig) -> tuple[pd.DataFrame, dict]:
    """Return one row per scored sleep epoch with its absolute offset in seconds."""
    stages = load_stage_table(csv_path)
    sleep = stages[stages["Event"].isin(SLEEP_STAGES)].copy()
    sleep["offset_s"] = (sleep["Start Time"] - edf_start).dt.total_seconds()
    sleep["epoch_id"] = sleep["Start Epoch"].astype(int)
    sleep["stage"] = sleep["Event"]
    sleep["duration_s"] = sleep["Duration"].astype(float)

    # A stage row shorter than the epoch length would break the fixed window;
    # PAAWS stages are 30.0 s, so flag rather than silently resize.
    irregular = int((sleep["duration_s"] - cfg.epoch_seconds).abs().gt(0.01).sum())

    meta = {
        "n_stage_rows": int(len(stages)),
        "n_sleep_epochs": int(len(sleep)),
        "n_wake_epochs": int((stages["Event"] == "Wake").sum()),
        "n_irregular_sleep_epochs": irregular,
        "first_sleep_offset_s": round(float(sleep["offset_s"].min()), 3) if len(sleep) else None,
        "last_sleep_offset_s": round(float(sleep["offset_s"].max()), 3) if len(sleep) else None,
    }
    cols = ["epoch_id", "stage", "offset_s", "duration_s", "Start Time"]
    return sleep[cols].reset_index(drop=True), meta


def slice_bounds(offset_s: float, fs: float, n_samples: int, epoch_seconds: int) -> tuple[int, int]:
    start = int(round(offset_s * fs))
    end = start + int(round(epoch_seconds * fs))
    start = max(0, min(start, n_samples))
    end = max(start, min(end, n_samples))
    return start, end
