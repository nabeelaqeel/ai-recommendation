"""Signal -> epoch -> night feature computation.

The same functions are used by the offline extractor and by the Stage 3
rolling-buffer inference helper, so training and serving cannot drift apart.

Design rules
------------
* Epoch windows come from absolute offsets, so annotation clock drift is harmless.
* Missing-value codes are masked before any statistic (SpO2/Pulse -0.05, HR 0.01,
  plus out-of-range junk such as SpO2 = -22 or HR = -5).
* HRV is computed only from ECG intervals inside the epoch that pass a
  rate-agreement gate and a physiological RMSSD bound; otherwise NaN, never imputed.
  It is deliberately causal (no look-ahead) so live inference is identical.
* `PosAngle` is a signed angle in degrees, not a linear quantity: it is reduced
  with circular statistics (resultant length, circular delta).
* `Activity` saturates at 10.0, so only distributional statistics are used.
* Only Garmin-plausible signals are read: HR, ECG (HRV), SpO2, Activity, PosAngle.
  Flow/effort/EEG/scored events are never touched here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal as sps

from .channels import (
    ChannelChoice,
    choose_hr_channel,
    resolve_channels,
    spo2_agreement,
    valid_hr,
    valid_spo2,
)
from .config import FeatureConfig
from .epochs import sleep_epochs, slice_bounds

META_FEATURES = ["night_id", "participant_id", "epoch_id", "start_time", "offset_s", "stage", "duration_s"]

HR_EPOCH_FEATURES = ["hr_mean", "hr_min", "hr_max", "hr_std", "hr_median", "hr_coverage"]
HRV_EPOCH_FEATURES = ["rmssd_ms", "hrv_valid", "hrv_beats"]
SPO2_EPOCH_FEATURES = ["spo2_mean", "spo2_min", "spo2_below90_frac", "spo2_coverage"]
ACTIVITY_EPOCH_FEATURES = ["activity_mean", "activity_std", "activity_p95"]
POSANGLE_EPOCH_FEATURES = ["posangle_deg", "posangle_r", "posangle_delta_deg"]

EPOCH_FEATURES = HR_EPOCH_FEATURES + HRV_EPOCH_FEATURES + SPO2_EPOCH_FEATURES + ACTIVITY_EPOCH_FEATURES + POSANGLE_EPOCH_FEATURES

NIGHTLY_FEATURES = [
    "hr_median",
    "hr_iqr",
    "hr_p5",
    "hr_p95",
    "hr_std",
    "hr_coverage",
    "rmssd_median",
    "rmssd_iqr",
    "hrv_valid_frac",
    "spo2_median",
    "spo2_iqr",
    "spo2_p5",
    "spo2_min",
    "spo2_below90_frac",
    "spo2_t90_h",
    "spo2_valid_h",
    "spo2_t90_per_valid_h",
    "spo2_coverage",
    "activity_mean",
    "activity_std",
    "activity_p95",
    "posangle_median",
    "posangle_iqr",
    "posangle_change_count",
    "posangle_delta_median",
    "posangle_r_mean",
    "hr_artefact",
    "spo2_artefact",
    "hrv_sufficient",
]

# Never allowed into a feature matrix: PSG-derived labels and their ingredients.
FORBIDDEN_FEATURE_PATTERNS = (
    "ahi",
    "resp_event",
    "desat_event",
    "tst",
    "sleep_efficiency",
    "n1_h",
    "n2_h",
    "n3_h",
    "rem_h",
    "stage",
)

ACTIVITY_SATURATION = 10.0


@dataclass
class NightResult:
    night_id: str
    participant_id: str
    epochs: pd.DataFrame
    nightly: dict
    audit: dict = field(default_factory=dict)


def empty_epoch_frame() -> pd.DataFrame:
    """Schema-complete empty frame, so zero-sleep nights never KeyError later."""
    cols = META_FEATURES + EPOCH_FEATURES
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})


def _stats(seg: np.ndarray, mask: np.ndarray, min_frac: float = 0.1) -> dict:
    """Distribution statistics over the valid part of one epoch."""
    n = len(seg)
    n_valid = int(mask.sum()) if n else 0
    if n == 0 or n_valid < max(5, int(min_frac * n)):
        return {
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "coverage": float(n_valid / n) if n else 0.0,
        }
    v = seg[mask]
    return {
        "mean": float(np.mean(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "coverage": float(n_valid / n),
    }


def epochwise(
    x: np.ndarray | None,
    fs: float,
    offsets: np.ndarray,
    cfg: FeatureConfig,
    validator,
    min_frac: float = 0.1,
) -> pd.DataFrame:
    """Apply `validator` inside each 30-s epoch and return distribution stats."""
    cols = ["mean", "min", "max", "std", "median", "p95", "coverage"]
    if x is None or not fs:
        return pd.DataFrame(np.nan, index=range(len(offsets)), columns=cols).assign(coverage=0.0)
    rows = []
    n = len(x)
    for off in offsets:
        a, b = slice_bounds(float(off), fs, n, cfg.epoch_seconds)
        seg = x[a:b]
        mask = validator(seg, cfg) if len(seg) else np.zeros(0, dtype=bool)
        rows.append(_stats(seg, mask, min_frac))
    return pd.DataFrame(rows, columns=cols)


def detect_r_peaks(ecg: np.ndarray, fs: int, cfg: FeatureConfig) -> np.ndarray:
    """Detect R peaks once per night on a 5-15 Hz band-passed ECG.

    Detection runs in epoch-sized blocks with a local amplitude threshold so a
    single high-amplitude artefact cannot hide beats elsewhere in the night.
    Flat blocks are skipped (SD == 0), which suppresses the long dropout runs.
    """
    x = np.nan_to_num(np.asarray(ecg, dtype=float))
    if len(x) < fs * 10 or not np.any(x):
        return np.zeros(0, dtype=int)
    sos = sps.butter(3, [5.0, 15.0], btype="bandpass", fs=fs, output="sos")
    try:
        y = sps.sosfiltfilt(sos, x)
    except ValueError:
        return np.zeros(0, dtype=int)
    block = int(cfg.epoch_seconds * fs)
    peaks: list[np.ndarray] = []
    for a in range(0, len(y), block):
        seg = y[a : a + block]
        if len(seg) < fs * 2:
            continue
        sd = float(np.std(seg))
        if sd <= 0:
            continue
        p, _ = sps.find_peaks(np.abs(seg), height=4.0 * sd, distance=int(0.3 * fs))
        if len(p):
            peaks.append(p + a)
    if not peaks:
        return np.zeros(0, dtype=int)
    return np.unique(np.concatenate(peaks))


def epoch_hrv(
    peaks: np.ndarray,
    fs: int,
    offsets: np.ndarray,
    hr: np.ndarray | None,
    hr_fs: int,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-epoch RMSSD from RR intervals inside that epoch (causal, no look-ahead).

    A window is usable only when the QRS-derived rate agrees with the recorded
    heart rate within 0.7-1.3x (or sits in 30-220 bpm when no HR channel exists)
    and at least `hrv_min_beats` RR intervals survive physiologic filtering.
    """
    rmssd = np.full(len(offsets), np.nan)
    usable = np.zeros(len(offsets), dtype=bool)
    beats = np.zeros(len(offsets), dtype=int)
    detected = np.full(len(offsets), np.nan)
    if len(peaks) == 0:
        return rmssd, usable, beats, detected

    lo_ms, hi_ms = cfg.rmssd_bounds_ms
    t = peaks / float(fs)
    for i, off in enumerate(offsets):
        a = float(off)
        b = a + cfg.epoch_seconds
        i0, i1 = np.searchsorted(t, [a, b], side="left")
        seg = t[i0:i1]
        if len(seg) < cfg.hrv_min_beats + 1:
            continue
        rr = np.diff(seg)
        rr = rr[(rr >= 0.3) & (rr <= 2.0)]
        beats[i] = len(rr)
        if len(rr) < cfg.hrv_min_beats:
            continue

        rate = 60.0 / float(np.median(rr))
        detected[i] = rate
        plausible = 30.0 <= rate <= 220.0
        if hr is not None and hr_fs and len(hr):
            ha = int(round(a * hr_fs))
            hb = min(len(hr), int(round(b * hr_fs)))
            hv = hr[ha:hb]
            hv = hv[valid_hr(hv, cfg)]
            if len(hv) >= 5:
                expected = float(np.median(hv))
                plausible = 0.7 * expected <= rate <= 1.3 * expected
        if not plausible:
            continue

        value = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0)
        if np.isfinite(value) and lo_ms <= value <= hi_ms:
            rmssd[i] = value
            usable[i] = True
    return rmssd, usable, beats, detected


def circular_mean_deg(angles: np.ndarray) -> float:
    z = np.mean(np.exp(1j * np.deg2rad(angles)))
    return float(np.rad2deg(np.angle(z)))


def circular_resultant(angles: np.ndarray) -> float:
    return float(np.abs(np.mean(np.exp(1j * np.deg2rad(angles)))))


def circular_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed shortest angular distance a - b, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def posangle_epochs(pos: np.ndarray | None, fs: float, offsets: np.ndarray, cfg: FeatureConfig) -> pd.DataFrame:
    cols = ["posangle_deg", "posangle_r"]
    if pos is None or not fs:
        return pd.DataFrame(np.nan, index=range(len(offsets)), columns=cols)
    deg, r = [], []
    for off in offsets:
        a, b = slice_bounds(float(off), fs, len(pos), cfg.epoch_seconds)
        seg = pos[a:b]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 5:
            deg.append(circular_mean_deg(seg))
            r.append(circular_resultant(seg))
        else:
            deg.append(np.nan)
            r.append(np.nan)
    out = pd.DataFrame({"posangle_deg": deg, "posangle_r": r})
    prev = out["posangle_deg"].shift(1)
    out["posangle_delta_deg"] = circular_delta_deg(out["posangle_deg"], prev)
    return out


def aggregate_nightly(epochs: pd.DataFrame, cfg: FeatureConfig) -> dict:
    """Collapse epoch rows into one row per night. Safe on an empty frame."""
    out: dict[str, float | int] = {}
    n = len(epochs)
    out["n_sleep_epochs"] = int(n)
    out["sleep_minutes"] = round(n * cfg.epoch_seconds / 60.0, 2)

    def col(name: str, default=np.nan) -> pd.Series:
        if name in epochs.columns:
            return epochs[name]
        return pd.Series(default, index=epochs.index, dtype="float64")

    hr = col("hr_median").dropna()
    out["hr_artefact"] = False
    if len(hr):
        out["hr_median"] = round(float(hr.median()), 2)
        out["hr_iqr"] = round(float(hr.quantile(0.75) - hr.quantile(0.25)), 2)
        out["hr_p5"] = round(float(hr.quantile(0.05)), 2)
        out["hr_p95"] = round(float(hr.quantile(0.95)), 2)
        out["hr_std"] = round(float(hr.std()), 2)
        hr_lo, hr_hi = cfg.hr_nightly_plausible_bpm
        if not (hr_lo <= out["hr_median"] <= hr_hi):
            # Whole-night artefact pattern (e.g. DS_97: sleep median ~145 bpm).
            # Values are withdrawn; coverage stays so missingness stays visible.
            out["hr_artefact"] = True
            for k in ("hr_median", "hr_iqr", "hr_p5", "hr_p95", "hr_std"):
                out[k] = np.nan
    else:
        for k in ("hr_median", "hr_iqr", "hr_p5", "hr_p95", "hr_std"):
            out[k] = np.nan
    out["hr_coverage"] = round(float(col("hr_coverage").mean()), 4) if n else np.nan

    rr = col("rmssd_ms").dropna()
    n_hrv = int(col("hrv_valid").fillna(False).astype(float).sum()) if n else 0
    out["rmssd_median"] = round(float(rr.median()), 2) if len(rr) else np.nan
    out["rmssd_iqr"] = round(float(rr.quantile(0.75) - rr.quantile(0.25)), 2) if len(rr) else np.nan
    out["hrv_sufficient"] = bool(n_hrv >= cfg.hrv_min_nightly_epochs)
    if not out["hrv_sufficient"]:
        # Too few QC-passed epochs for a representative nightly RMSSD.
        out["rmssd_median"] = np.nan
        out["rmssd_iqr"] = np.nan
    out["hrv_valid_frac"] = round(float(col("hrv_valid").fillna(0).mean()), 4) if n else np.nan

    sp = col("spo2_mean").dropna()
    out["spo2_artefact"] = False
    if len(sp):
        out["spo2_median"] = round(float(sp.median()), 2)
        out["spo2_iqr"] = round(float(sp.quantile(0.75) - sp.quantile(0.25)), 2)
        out["spo2_p5"] = round(float(sp.quantile(0.05)), 2)
        sp_lo, sp_hi = cfg.spo2_nightly_plausible_pct
        if not (sp_lo <= out["spo2_median"] <= sp_hi):
            out["spo2_artefact"] = True
            for k in ("spo2_median", "spo2_iqr", "spo2_p5"):
                out[k] = np.nan
    else:
        for k in ("spo2_median", "spo2_iqr", "spo2_p5"):
            out[k] = np.nan
    spo2_min = col("spo2_min").dropna()
    out["spo2_min"] = round(float(spo2_min.min()), 2) if len(spo2_min) else np.nan

    duration = col("duration_s", float(cfg.epoch_seconds)).fillna(cfg.epoch_seconds)
    below = col("spo2_below90_frac").fillna(0.0)
    cov = col("spo2_coverage").fillna(0.0)
    has_spo2 = len(sp) > 0  # any epoch with a usable SpO2 value
    if n and has_spo2:
        out["spo2_below90_frac"] = round(float(below.mean()), 4)
        t90_h = float((below * duration).sum() / 3600.0)
        valid_h = float((cov * duration).sum() / 3600.0)
    else:
        # No usable SpO2 in the night: T90 is undefined, NOT zero. Reporting 0
        # would let a model read sensor dropout as "no desaturation".
        out["spo2_below90_frac"] = np.nan
        t90_h, valid_h = np.nan, np.nan
    out["spo2_t90_h"] = round(t90_h, 4) if pd.notna(t90_h) else np.nan
    out["spo2_valid_h"] = round(valid_h, 4) if pd.notna(valid_h) else np.nan
    out["spo2_t90_per_valid_h"] = round(t90_h / valid_h, 4) if valid_h and pd.notna(valid_h) and valid_h > 0 else np.nan
    out["spo2_coverage"] = round(float(cov.mean()), 4) if n else np.nan
    if out["spo2_artefact"]:
        # Desaturation statistics from an implausible channel are withdrawn too.
        for k in ("spo2_min", "spo2_below90_frac", "spo2_t90_h", "spo2_t90_per_valid_h"):
            out[k] = np.nan

    act = col("activity_mean").dropna()
    out["activity_mean"] = round(float(act.mean()), 5) if len(act) else np.nan
    out["activity_std"] = round(float(col("activity_std").mean()), 5) if n else np.nan
    act_p95 = col("activity_p95").dropna()
    out["activity_p95"] = round(float(act_p95.quantile(0.95)), 5) if len(act_p95) else np.nan

    pos = col("posangle_deg").dropna()
    if len(pos):
        out["posangle_median"] = round(circular_mean_deg(pos.to_numpy()), 2)
        # Circular IQR proxy: spread of epoch orientations around their mean.
        dev = np.abs(circular_delta_deg(pos.to_numpy(), out["posangle_median"]))
        out["posangle_iqr"] = round(float(np.percentile(dev, 75) - np.percentile(dev, 25)), 2)
    else:
        out["posangle_median"] = np.nan
        out["posangle_iqr"] = np.nan
    delta = col("posangle_delta_deg").dropna().abs()
    out["posangle_change_count"] = int((delta > cfg.posture_change_deg).sum())
    out["posangle_delta_median"] = round(float(delta.median()), 2) if len(delta) else np.nan
    r = col("posangle_r").dropna()
    out["posangle_r_mean"] = round(float(r.mean()), 4) if len(r) else np.nan

    return out


def extract_night(night, cfg: FeatureConfig, reader_factory) -> NightResult:
    """Extract epoch and nightly features for one participant-night read-only."""
    audit: dict = {
        "night_id": night.night_id,
        "participant_id": night.participant,
        "root": night.root_name,
        "status": "ok",
    }
    with reader_factory(night.edf) as f:
        labels = list(f.getSignalLabels())
        fs_by_index = {i: float(f.getSampleFrequency(i)) for i in range(len(labels))}
        edf_start = pd.Timestamp(f.getStartdatetime())
        duration_s = float(f.getFileDuration())
        audit["n_channels"] = len(labels)
        audit["edf_start"] = str(edf_start)
        audit["edf_duration_h"] = round(duration_s / 3600, 4)

        # Stage grid first: a night with no scored sleep needs no signal reads.
        epochs, meta = sleep_epochs(night.csv, edf_start, cfg)
        audit.update(meta)
        if len(epochs) == 0:
            audit["status"] = "no_sleep_epochs"
            return NightResult(
                night.night_id,
                night.participant,
                empty_epoch_frame(),
                aggregate_nightly(empty_epoch_frame(), cfg),
                audit,
            )

        choices: dict[str, ChannelChoice] = resolve_channels(labels)
        audit["ecg_label"] = choices["ecg"].label or ""
        audit["spo2_label"] = choices["spo2"].label or ""
        audit["activity_label"] = choices["activity"].label or ""
        audit["posangle_label"] = choices["posangle"].label or ""

        def read_index(idx: int) -> np.ndarray:
            return f.readSignal(idx)

        hr_idx, hr_label, hr_audit = choose_hr_channel(read_index, choices["hr"].candidates, cfg)
        audit["hr_label"] = hr_label or ""
        audit["hr_channel_index"] = int(hr_idx) if hr_idx is not None else -1
        audit["hr_duplicate_resolved"] = bool(len(choices["hr"].candidates) > 1)
        audit["hr_candidates"] = hr_audit

        if len(choices["spo2"].candidates) > 1:
            agree = spo2_agreement(read_index, [i for i, _ in choices["spo2"].candidates], cfg)
            if agree:
                audit["spo2_agreement"] = agree

        hr = read_index(hr_idx) if hr_idx is not None else None
        hr_fs = int(round(fs_by_index[hr_idx])) if hr_idx is not None else 0

        spo2 = spo2_fs = None
        if choices["spo2"].index is not None:
            spo2 = read_index(choices["spo2"].index)
            spo2_fs = int(round(fs_by_index[choices["spo2"].index]))

        act, act_fs = None, 0
        if choices["activity"].index is not None:
            act = read_index(choices["activity"].index)
            act_fs = int(round(fs_by_index[choices["activity"].index]))

        pos, pos_fs = None, 0
        if choices["posangle"].index is not None:
            pos = read_index(choices["posangle"].index)
            pos_fs = int(round(fs_by_index[choices["posangle"].index]))

        ecg, ecg_fs = None, 0
        if choices["ecg"].index is not None:
            ecg = read_index(choices["ecg"].index)
            ecg_fs = int(round(fs_by_index[choices["ecg"].index]))

        offsets = epochs["offset_s"].to_numpy()
        hr_source = "heart_rate_channel" if hr is not None else "missing"

        hr_df = epochwise(hr, hr_fs, offsets, cfg, valid_hr)
        for src, dst in [
            ("mean", "hr_mean"),
            ("min", "hr_min"),
            ("max", "hr_max"),
            ("std", "hr_std"),
            ("median", "hr_median"),
            ("coverage", "hr_coverage"),
        ]:
            epochs[dst] = hr_df[src].to_numpy()

        if ecg is not None:
            # QRS detection only on the scored-sleep span: cheaper and avoids
            # artefact-rich pre-sleep/post-sleep segments dominating thresholds.
            span_a = max(0, int(offsets.min()) - cfg.epoch_seconds)
            span_b = min(len(ecg), int((offsets.max() + 2 * cfg.epoch_seconds) * ecg_fs))
            peaks = detect_r_peaks(ecg[span_a:span_b], ecg_fs, cfg) + span_a
            rmssd, hrv_ok, beats, det_hr = epoch_hrv(peaks, ecg_fs, offsets, hr, hr_fs, cfg)
            epochs["rmssd_ms"] = rmssd
            epochs["hrv_valid"] = hrv_ok
            epochs["hrv_beats"] = beats
            audit["ecg_peaks"] = int(len(peaks))
            audit["ecg_usable_window_pct"] = round(100 * float(hrv_ok.mean()), 1)
            if hr is None:
                # Only night without a Heart Rate channel: use gated QRS rate.
                epochs["hr_median"] = np.where(hrv_ok, det_hr, np.nan)
                epochs["hr_coverage"] = hrv_ok.astype(float)
                hr_source = "ecg_fallback" if hrv_ok.any() else "missing"
        else:
            epochs["rmssd_ms"] = np.nan
            epochs["hrv_valid"] = False
            epochs["hrv_beats"] = 0
            audit["ecg_peaks"] = 0
            audit["ecg_usable_window_pct"] = 0.0
        audit["hr_source"] = hr_source

        sp_df = epochwise(spo2, spo2_fs or 0, offsets, cfg, valid_spo2)
        epochs["spo2_mean"] = sp_df["mean"].to_numpy()
        epochs["spo2_min"] = sp_df["min"].to_numpy()
        epochs["spo2_coverage"] = sp_df["coverage"].to_numpy()
        below = []
        for off in offsets:
            a, b = slice_bounds(float(off), spo2_fs or 1, len(spo2) if spo2 is not None else 0, cfg.epoch_seconds)
            seg = spo2[a:b] if spo2 is not None else np.zeros(0)
            v = valid_spo2(seg, cfg) if len(seg) else np.zeros(0, dtype=bool)
            below.append(float(np.mean(seg[v] < cfg.desat_threshold_pct)) if v.any() else np.nan)
        epochs["spo2_below90_frac"] = below

        act_df = epochwise(act, act_fs, offsets, cfg, lambda s, c: np.isfinite(s))
        epochs["activity_mean"] = act_df["mean"].to_numpy()
        epochs["activity_std"] = act_df["std"].to_numpy()
        epochs["activity_p95"] = act_df["p95"].to_numpy()

        pa = posangle_epochs(pos, pos_fs, offsets, cfg)
        epochs["posangle_deg"] = pa["posangle_deg"].to_numpy()
        epochs["posangle_r"] = pa["posangle_r"].to_numpy()
        epochs["posangle_delta_deg"] = pa["posangle_delta_deg"].to_numpy()

        epochs.insert(0, "night_id", night.night_id)
        epochs.insert(1, "participant_id", night.participant)
        epochs["start_time"] = epochs["Start Time"].astype(str)
        epochs = epochs.drop(columns=["Start Time"])

        nightly = aggregate_nightly(epochs, cfg)
        audit["n_hr_epochs"] = int(epochs["hr_median"].notna().sum())
        audit["n_spo2_epochs"] = int(epochs["spo2_mean"].notna().sum())
        audit["n_hrv_epochs"] = int(epochs["hrv_valid"].sum())
        audit["n_posangle_epochs"] = int(epochs["posangle_deg"].notna().sum())
        audit["activity_at_ceiling_frac"] = round(
            float((act >= ACTIVITY_SATURATION).mean()), 4
        ) if act is not None and len(act) else np.nan
        return NightResult(night.night_id, night.participant, epochs, nightly, audit)


def check_no_leakage(columns: list[str]) -> list[str]:
    """Return any column names that could leak PSG-derived label information."""
    bad = []
    for c in columns:
        low = c.lower()
        if any(p in low for p in FORBIDDEN_FEATURE_PATTERNS):
            bad.append(c)
    return bad
