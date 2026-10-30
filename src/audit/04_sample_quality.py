"""Signal-quality and dashboard-feature feasibility checks on a sample of nights.

Reads selective channels only for 14 representative nights across both roots
and handles the device-specific missing-value codes discovered in the audit:
  - SpO2 / Pulse: missing encoded as -0.05 (and other negative artefacts)
  - Heart Rate: missing encoded as 0.01
  - ECG/EKG: flat segments and variable quality; QRS usability is gated
  - Pulse Waveform (oximeter PPG): raw int16-like values with flat dropouts

Writes outputs/paaws_sample_signal_quality.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyedflib
from scipy import signal as sps

from common import OUTPUTS, configure_logging, discover_nights

log = configure_logging("04_sample_quality")

SAMPLE_IDS = [
    "DS_11-N1",     # root A, standard layout
    "DS_108-N1",    # root A, 100% wake, 16 h recording
    "DS_41-N2",     # root A, annotation clock drift 263 s
    "DS_97-N1",     # root A, no SpO2 / no Pulse
    "DS_109-N1",    # root A, 1.18 h short recording
    "DS_176-N1",    # root A, long unscored gaps
    "DS_237-N2",    # root A, 100% wake
    "DS_102-N1",    # root B (only N2 exists) -> corrected below
    "DS_103-N1",    # root B, EKG-only layout
    "DS_118-N1",    # root B, clean EKG in spot check
    "DS_148-N1",    # root B
    "DS_211-N1",    # root B
    "DS_30-N1",     # root B, mixed ECG quality
    "DS_79-N1",     # root B, EKG-only
]
SAMPLE_IDS[SAMPLE_IDS.index("DS_102-N1")] = "DS_102-N2"

SPO2_NAMES = ["Saturation", "SpO2 B-B"]
HR_NAMES = ["Heart Rate", "Pulse"]
ECG_NAMES = ["ECG", "EKG"]
PPG_NAMES = ["Pulse Waveform"]
ACT_NAMES = ["Activity"]
POS_NAMES = ["PosAngle"]
FLOW_NAMES = ["Flow", "Nasal Pressure"]


def _pick(labels: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in labels:
            return c
    return None


def _valid_spo2(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x >= 50) & (x <= 120)


def _valid_hr(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 1) & (x >= 20) & (x <= 250)


def _flat_run(x: np.ndarray, fs: float) -> float:
    if len(x) < 2:
        return len(x) / fs
    d = np.diff(x) == 0
    if not d.any():
        return 0.0
    change = np.flatnonzero(np.diff(np.r_[False, d, False]))
    return float((change[1::2] - change[::2]).max() / fs)


def _ecg_quality(x: np.ndarray, fs: int, hr: np.ndarray | None, hr_fs: int) -> dict:
    """QRS usability on evenly spaced 60-s windows, gated against recorded HR."""
    window = 60 * fs
    n_windows = max(1, len(x) // window)
    idx = np.unique(np.linspace(0, n_windows - 1, min(n_windows, 60)).astype(int))
    sos = sps.butter(3, [5, 15], btype="bandpass", fs=fs, output="sos")
    usable = 0
    rmssd_vals = []
    agreements = []
    for w in idx:
        seg = x[w * window : (w + 1) * window]
        if len(seg) < window or np.std(seg) == 0:
            continue
        y = sps.sosfiltfilt(sos, seg)
        sd = np.std(y)
        if sd == 0:
            continue
        peaks, _ = sps.find_peaks(np.abs(y), height=4 * sd, distance=int(0.3 * fs))
        rr = np.diff(peaks) / fs
        rr = rr[(rr >= 0.3) & (rr <= 2.0)]
        expected = np.nan
        if hr is not None and len(hr):
            a = min(len(hr) - 1, int(w * 60 * hr_fs))
            b = min(len(hr), int((w + 1) * 60 * hr_fs))
            hv = hr[a:b]
            hv = hv[_valid_hr(hv)]
            expected = float(np.median(hv)) if len(hv) else np.nan
        if len(rr) >= 10 and np.isfinite(expected) and expected >= 20:
            detected_hr = 60.0 / np.median(rr)
            if 0.7 * expected <= detected_hr <= 1.3 * expected:
                usable += 1
                rmssd_vals.append(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000)
                agreements.append(detected_hr / expected)
    total = len(idx)
    flat_pct = 100 * float(np.mean(np.diff(x) == 0)) if len(x) > 1 else 100.0
    return {
        "ecg_flat_pct": round(flat_pct, 2),
        "ecg_longest_flat_s": round(_flat_run(x, fs), 1),
        "ecg_usable_window_pct": round(100 * usable / total, 1) if total else 0.0,
        "ecg_rmssd_ms": round(float(np.median(rmssd_vals)), 1) if rmssd_vals else np.nan,
        "ecg_rmssd_windows": len(rmssd_vals),
        "ecg_hr_agreement_ratio": round(float(np.median(agreements)), 3) if agreements else np.nan,
    }


def main() -> None:
    nights = {n.night_id: n for n in discover_nights()}
    stages = pd.read_csv(OUTPUTS / "paaws_night_stage_metrics.csv").set_index("night_id")

    rows = []
    for night_id in SAMPLE_IDS:
        n = nights[night_id]
        st = stages.loc[night_id]
        row = {"night_id": night_id, "root": n.root_name, "participant_id": n.participant}
        sleep_start = pd.Timestamp(st.first_sleep_start) if isinstance(st.first_sleep_start, str) and st.first_sleep_start else None
        sleep_end = pd.Timestamp(st.last_sleep_end) if isinstance(st.last_sleep_end, str) and st.last_sleep_end else None

        with pyedflib.EdfReader(str(n.edf)) as f:
            labels = list(f.getSignalLabels())
            fs_map = {lab: f.getSampleFrequency(i) for i, lab in enumerate(labels)}
            edf_start = pd.Timestamp(f.getStartdatetime())
            duration_s = float(f.getFileDuration())

            def read(name: str) -> tuple[np.ndarray, int]:
                idx = labels.index(name)
                return f.readSignal(idx), int(round(fs_map[name]))

            def sleep_mask(fs: int) -> np.ndarray | None:
                if sleep_start is None or sleep_end is None:
                    return None
                mask = np.zeros(int(duration_s * fs), dtype=bool)
                a = max(0, int((sleep_start - edf_start).total_seconds() * fs))
                b = min(len(mask), int((sleep_end - edf_start).total_seconds() * fs))
                mask[a:b] = True
                return mask

            # ---- SpO2 -------------------------------------------------------
            spo2_name = _pick(labels, SPO2_NAMES)
            row["spo2_channel"] = spo2_name or ""
            if spo2_name:
                x, fs = read(spo2_name)
                v = _valid_spo2(x)
                row["spo2_valid_pct"] = round(100 * float(v.mean()), 1)
                if v.any():
                    row["spo2_median"] = round(float(np.median(x[v])), 2)
                    row["spo2_p5"] = round(float(np.percentile(x[v], 5)), 2)
                    row["spo2_p95"] = round(float(np.percentile(x[v], 95)), 2)
                    row["spo2_pct_below90_valid"] = round(100 * float(np.mean(x[v] < 90)), 2)
                    row["spo2_pct_above100"] = round(100 * float(np.mean(x[v] > 100)), 2)
                    row["spo2_t90_valid_h"] = round(float(np.sum(x[v] < 90) / fs / 3600), 4)
                mask = sleep_mask(fs)
                if mask is not None:
                    xs = x[: len(mask)][mask[: len(x)]]
                    vs = _valid_spo2(xs)
                    row["spo2_t90_sleep_h"] = (
                        round(float(np.sum(xs[vs] < 90) / fs / 3600), 4) if vs.any() else np.nan
                    )

            # ---- HR / Pulse -------------------------------------------------
            hr_name = _pick(labels, HR_NAMES)
            row["hr_channel"] = hr_name or ""
            hr_arr = None
            hr_fs = 0
            if hr_name:
                x, fs = read(hr_name)
                hr_arr, hr_fs = x, fs
                v = _valid_hr(x)
                row["hr_valid_pct"] = round(100 * float(v.mean()), 1)
                if v.any():
                    row["hr_median"] = round(float(np.median(x[v])), 1)
                    row["hr_iqr"] = round(float(np.percentile(x[v], 75) - np.percentile(x[v], 25)), 1)
                    row["hr_outside_30_220_pct"] = round(
                        100 * float(np.mean((x[v] < 30) | (x[v] > 220))), 2
                    )

            pulse_name = _pick(labels, ["Pulse"])  # oximeter pulse only
            if pulse_name:
                x, fs = read(pulse_name)
                v = _valid_hr(x)
                row["pulse_valid_pct"] = round(100 * float(v.mean()), 1)

            # ---- ECG / EKG --------------------------------------------------
            ecg_name = _pick(labels, ECG_NAMES)
            row["ecg_channel"] = ecg_name or ""
            if ecg_name:
                x, fs = read(ecg_name)
                row.update(_ecg_quality(x, fs, hr_arr, hr_fs))

            # ---- Oximeter PPG ----------------------------------------------
            ppg_name = _pick(labels, PPG_NAMES)
            row["ppg_channel"] = ppg_name or ""
            if ppg_name:
                x, fs = read(ppg_name)
                nz = np.isfinite(x) & (x != 0)
                row["ppg_nonzero_pct"] = round(100 * float(nz.mean()), 1)
                row["ppg_std_raw"] = round(float(np.std(x[nz])), 1) if nz.any() else np.nan

            # ---- Motion / position -----------------------------------------
            act_name = _pick(labels, ACT_NAMES)
            pos_name = _pick(labels, POS_NAMES)
            if act_name:
                x, fs = read(act_name)
                row["activity_std"] = round(float(np.nanstd(x)), 5)
                row["activity_p95"] = round(float(np.nanpercentile(x, 95)), 5)
            if pos_name:
                x, fs = read(pos_name)
                row["posangle_median"] = round(float(np.nanmedian(x)), 2)
                row["posangle_iqr"] = round(
                    float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)), 2
                )

            # ---- Airflow sanity --------------------------------------------
            flow_name = _pick(labels, FLOW_NAMES)
            if flow_name:
                x, fs = read(flow_name)
                row["flow_longest_flat_s"] = round(_flat_run(x, fs), 2)

        rows.append(row)
        log.info(
            "  %s done (ECG=%s usable=%.0f%%, SpO2=%.0f%% valid, HR=%.0f%% valid)",
            night_id,
            row.get("ecg_channel", ""),
            row.get("ecg_usable_window_pct", 0),
            row.get("spo2_valid_pct", 0),
            row.get("hr_valid_pct", 0),
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUTS / "paaws_sample_signal_quality.csv", index=False)
    log.info("Wrote %s (%d nights x %d columns)", OUTPUTS / "paaws_sample_signal_quality.csv", len(out), out.shape[1])

    cols = [
        "night_id", "ecg_channel", "ecg_usable_window_pct", "ecg_rmssd_ms", "ecg_rmssd_windows",
        "spo2_valid_pct", "spo2_median", "spo2_pct_below90_valid", "spo2_t90_valid_h",
        "hr_valid_pct", "hr_median", "ppg_nonzero_pct", "activity_std", "posangle_iqr",
    ]
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
