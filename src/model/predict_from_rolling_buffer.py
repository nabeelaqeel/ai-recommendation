"""Live streaming bridge: predict from a rolling sensor buffer (FR #7, NFR #1).

The buffer is what a wrist device can actually supply in near-real-time: recent
HR samples, recent SpO2 samples, recent activity, optional per-epoch RMSSD
(Garmin computes HRV itself), and optionally a torso angle if an external
sensor provides one. There is no PSG sleep staging at run time, so every 30-s
window in the buffer contributes (documented limitation vs training, where
features are restricted to scored sleep).

Statistics are computed by the *same* `features.py` functions used in
training, so the aggregation logic cannot drift between offline and live use.

`python -m src.model.predict_from_rolling_buffer` runs the latency benchmark
against the 50 ms single-invocation budget.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.audit.common import OUTPUTS, PROJECT_ROOT
from src.features.config import FeatureConfig, Phase2Config, load_config
from src.features.features import (
    EPOCH_FEATURES,
    aggregate_nightly,
    circular_delta_deg,
    circular_mean_deg,
    circular_resultant,
    epochwise,
)
from src.model.classifier import Phase2Classifier

LATENCY_BUDGET_MS = 50.0
log = logging.getLogger("rolling")


def rolling_buffer_to_epoch_rows(buffer: dict, cfg: FeatureConfig) -> pd.DataFrame:
    """Chop a rolling buffer into 30-s windows and compute epoch statistics."""
    n_seconds = None
    for key in ("hr", "spo2", "activity", "posangle"):
        arr = buffer.get(key)
        if arr is not None and len(arr):
            fs = float(buffer.get(f"{key}_fs", 0) or 0)
            if fs <= 0:
                raise ValueError(f"buffer['{key}_fs'] must be > 0 when '{key}' is provided")
            n_seconds = max(n_seconds or 0, len(arr) / fs)

    if n_seconds is None and buffer.get("rmssd_ms") is None:
        raise ValueError("rolling buffer is empty: provide at least one signal series")

    n_seconds = n_seconds or len(buffer["rmssd_ms"]) * cfg.epoch_seconds
    n_windows = int(np.floor(n_seconds / cfg.epoch_seconds))
    offsets = np.arange(n_windows) * cfg.epoch_seconds

    rows = pd.DataFrame({"offset_s": offsets})

    def _attach(stats: pd.DataFrame, prefix: str, mapping: dict[str, str]) -> None:
        for src, dst in mapping.items():
            rows[dst] = stats[src].to_numpy() if len(stats) == len(rows) else np.nan

    if buffer.get("hr") is not None:
        hr = np.asarray(buffer["hr"], dtype=float)
        fs = float(buffer["hr_fs"])
        s = epochwise(hr, fs, offsets, cfg, lambda x, c: np.isfinite(x) & (x >= c.hr_valid_bpm[0]) & (x <= c.hr_valid_bpm[1]))
        _attach(
            s,
            "hr",
            {"mean": "hr_mean", "min": "hr_min", "max": "hr_max", "std": "hr_std", "median": "hr_median", "coverage": "hr_coverage"},
        )
    else:
        for c in ("hr_mean", "hr_min", "hr_max", "hr_std", "hr_median", "hr_coverage"):
            rows[c] = np.nan

    if buffer.get("spo2") is not None:
        sp = np.asarray(buffer["spo2"], dtype=float)
        fs = float(buffer["spo2_fs"])
        valid = lambda x, c: np.isfinite(x) & (x >= c.spo2_valid_pct[0]) & (x <= c.spo2_valid_pct[1])  # noqa: E731
        s = epochwise(sp, fs, offsets, cfg, valid)
        _attach(s, "spo2", {"mean": "spo2_mean", "min": "spo2_min", "coverage": "spo2_coverage"})
        below = []
        for off in offsets:
            a = int(round(off * fs))
            b = a + int(round(cfg.epoch_seconds * fs))
            seg = sp[a:b]
            m = valid(seg, cfg) if len(seg) else np.zeros(0, dtype=bool)
            below.append(float(np.mean(seg[m] < cfg.desat_threshold_pct)) if m.any() else np.nan)
        rows["spo2_below90_frac"] = below
    else:
        for c in ("spo2_mean", "spo2_min", "spo2_coverage", "spo2_below90_frac"):
            rows[c] = np.nan

    if buffer.get("activity") is not None:
        act = np.asarray(buffer["activity"], dtype=float)
        fs = float(buffer["activity_fs"])
        s = epochwise(act, fs, offsets, cfg, lambda x, c: np.isfinite(x))
        _attach(s, "act", {"mean": "activity_mean", "std": "activity_std", "p95": "activity_p95"})
    else:
        for c in ("activity_mean", "activity_std", "activity_p95"):
            rows[c] = np.nan

    if buffer.get("posangle") is not None:
        pos = np.asarray(buffer["posangle"], dtype=float)
        fs = float(buffer["posangle_fs"])
        deg, r = [], []
        for off in offsets:
            a = int(round(off * fs))
            b = a + int(round(cfg.epoch_seconds * fs))
            seg = pos[a:b]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 5:
                deg.append(circular_mean_deg(seg))
                r.append(circular_resultant(seg))
            else:
                deg.append(np.nan)
                r.append(np.nan)
        rows["posangle_deg"] = deg
        rows["posangle_r"] = r
        if len(deg) > 1:
            deg_arr = np.asarray(deg, dtype=float)
            deltas = circular_delta_deg(deg_arr[1:], deg_arr[:-1])
            rows["posangle_delta_deg"] = [np.nan, *deltas.tolist()]
        else:
            rows["posangle_delta_deg"] = np.nan
    else:
        for c in ("posangle_deg", "posangle_r", "posangle_delta_deg"):
            rows[c] = np.nan

    if buffer.get("rmssd_ms") is not None:
        rr = np.asarray(buffer["rmssd_ms"], dtype=float)
        lo, hi = cfg.rmssd_bounds_ms
        take = np.full(len(rows), np.nan)
        k = min(len(rr), len(rows))
        take[:k] = rr[:k]
        ok = np.isfinite(take) & (take >= lo) & (take <= hi)
        rows["rmssd_ms"] = np.where(ok, take, np.nan)
        rows["hrv_valid"] = ok
        rows["hrv_beats"] = np.nan
    else:
        rows["rmssd_ms"] = np.nan
        rows["hrv_valid"] = False
        rows["hrv_beats"] = 0

    rows["duration_s"] = cfg.epoch_seconds
    return rows


def predict_from_rolling_buffer(
    buffer: dict,
    classifier: Phase2Classifier | None = None,
    cfg: Phase2Config | None = None,
) -> dict:
    """Compute nightly-style features from a live buffer and classify once.

    Returns the 3-class guidance band with probabilities, the binary elevated
    view at the cost-sensitive threshold, and timing. One call = one
    classification; latency budget is 50 ms.
    """
    t0 = time.perf_counter()
    cfg = cfg or load_config()
    if classifier is None:
        classifier = Phase2Classifier.load()

    rows = rolling_buffer_to_epoch_rows(buffer, cfg.features)
    t_feat = time.perf_counter() - t0

    for c in EPOCH_FEATURES:
        if c not in rows.columns:
            rows[c] = np.nan
    nightly = aggregate_nightly(rows, cfg.features)
    t_agg = time.perf_counter() - t0 - t_feat

    # ONE model inference serves the band, the probabilities and the threshold
    # flag — the real-time budget does not allow paying inference twice.
    full = classifier.predict_full(nightly)
    t_total = (time.perf_counter() - t0) * 1000.0

    return {
        "band": full["band"],
        "probabilities": full["probabilities"],
        "p_positive": full["p_positive"],
        "positive_class": full["positive_class"],
        "flagged_positive": full["flagged_positive"],
        "threshold": full["threshold"],
        "n_windows": int(len(rows)),
        "hrv_valid_frac": nightly.get("hrv_valid_frac"),
        "disclaimer": full["disclaimer"],
        "latency_ms": {
            "total": round(t_total, 2),
            "features": round(t_feat * 1000.0, 2),
            "aggregate": round(t_agg * 1000.0, 2),
            "classify": round(t_total - t_feat * 1000.0 - t_agg * 1000.0, 2),
        },
    }


# ---------------------------------------------------------------------------
# latency benchmark
# ---------------------------------------------------------------------------
def synthetic_buffer(minutes: int = 10, cfg: FeatureConfig | None = None, seed: int = 0) -> dict:
    """A plausible 10-minute wrist-device buffer (HR 1 Hz, SpO2 ~1/3 Hz, etc.)."""
    rng = np.random.default_rng(seed)
    n = minutes * 60
    hr_fs, spo2_fs, act_fs, pos_fs = 1.0, 1.0, 1.0, 1.0
    beats = np.cumsum(rng.normal(1.0, 0.08, minutes * 2))  # ~60 bpm
    rmssd = rng.normal(40, 8, minutes * 2)
    return {
        "hr": np.clip(rng.normal(58, 6, int(n * hr_fs)), 40, 100),
        "hr_fs": hr_fs,
        "spo2": np.clip(rng.normal(95, 1.5, int(n * spo2_fs)), 85, 100),
        "spo2_fs": spo2_fs,
        "activity": np.abs(rng.normal(0.03, 0.05, int(n * act_fs))),
        "activity_fs": act_fs,
        "posangle": rng.normal(-30, 15, int(n * pos_fs)),
        "posangle_fs": pos_fs,
        "rmssd_ms": rmssd,
    }


def run_benchmark(n_iter: int = 200, classifier: Phase2Classifier | None = None) -> dict:
    cfg = load_config()
    classifier = classifier or Phase2Classifier.load()
    buf = synthetic_buffer(cfg=cfg.features)
    # warm-up
    predict_from_rolling_buffer(buf, classifier, cfg)
    times = []
    for i in range(n_iter):
        buf2 = synthetic_buffer(seed=i % 7)
        t0 = time.perf_counter()
        predict_from_rolling_buffer(buf2, classifier, cfg)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {
        "n_iter": n_iter,
        "p50_ms": round(float(np.percentile(t, 50)), 2),
        "p95_ms": round(float(np.percentile(t, 95)), 2),
        "max_ms": round(float(t.max()), 2),
        "budget_ms": LATENCY_BUDGET_MS,
        "within_budget": bool(np.percentile(t, 95) < LATENCY_BUDGET_MS),
    }


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    res = run_benchmark()
    out = PROJECT_ROOT / "outputs" / "phase2_latency_benchmark.csv"
    pd.DataFrame([res]).to_csv(out, index=False)
    # one worked example for the report
    example = predict_from_rolling_buffer(synthetic_buffer(), Phase2Classifier.load())
    (PROJECT_ROOT / "outputs" / "phase2_latency_example.json").write_text(json.dumps(example, indent=2, default=str))
    print(res)
    print(f"wrote {out}")
