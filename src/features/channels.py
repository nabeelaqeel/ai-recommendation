"""Canonical channel resolution for PAAWS nights.

Channel *labels* are not unique identifiers in this dataset: `Heart Rate`
appears twice on 30 nights, ECG and EKG are alternate labels for the same
200 Hz cardiac channel, and SpO2 has two parallel channels. Resolution is by
role with a deterministic, logged rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FeatureConfig

ECG_LABELS = ("ECG", "EKG")
HR_LABELS = ("Heart Rate",)
SPO2_LABELS = ("Saturation", "SpO2 B-B")
PULSE_LABELS = ("Pulse",)
ACTIVITY_LABELS = ("Activity",)
POSANGLE_LABELS = ("PosAngle",)


@dataclass
class ChannelChoice:
    role: str
    label: str | None
    index: int | None
    candidates: list[tuple[int, str]] = field(default_factory=list)
    note: str = ""


def pick_first(labels: list[str], candidates: tuple[str, ...]) -> tuple[int | None, str | None]:
    for name in candidates:
        if name in labels:
            return labels.index(name), name
    return None, None


def all_indices(labels: list[str], name: str) -> list[int]:
    return [i for i, lab in enumerate(labels) if lab == name]


def valid_hr(x: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    lo, hi = cfg.hr_valid_bpm
    return np.isfinite(x) & (x >= lo) & (x <= hi)


def valid_spo2(x: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    lo, hi = cfg.spo2_valid_pct
    return np.isfinite(x) & (x >= lo) & (x <= hi)


def resolve_channels(labels: list[str]) -> dict[str, ChannelChoice]:
    """Map canonical roles to channel indices without reading signal data."""
    choices: dict[str, ChannelChoice] = {}

    ecg_idx, ecg_name = pick_first(labels, ECG_LABELS)
    choices["ecg"] = ChannelChoice(
        "ecg", ecg_name, ecg_idx, [(i, labels[i]) for i in all_indices(labels, "ECG") + all_indices(labels, "EKG")]
    )

    hr_candidates = [(i, labels[i]) for i in all_indices(labels, "Heart Rate")]
    choices["hr"] = ChannelChoice(
        "hr",
        hr_candidates[0][1] if hr_candidates else None,
        hr_candidates[0][0] if hr_candidates else None,
        hr_candidates,
    )

    spo2_idx, spo2_name = pick_first(labels, SPO2_LABELS)
    choices["spo2"] = ChannelChoice(
        "spo2",
        spo2_name,
        spo2_idx,
        [(i, labels[i]) for i in all_indices(labels, "Saturation") + all_indices(labels, "SpO2 B-B")],
    )

    pulse_idx, pulse_name = pick_first(labels, PULSE_LABELS)
    choices["pulse"] = ChannelChoice("pulse", pulse_name, pulse_idx)

    act_idx, act_name = pick_first(labels, ACTIVITY_LABELS)
    choices["activity"] = ChannelChoice("activity", act_name, act_idx)

    pos_idx, pos_name = pick_first(labels, POSANGLE_LABELS)
    choices["posangle"] = ChannelChoice("posangle", pos_name, pos_idx)

    return choices


def choose_hr_channel(
    reader, candidates: list[tuple[int, str]], cfg: FeatureConfig
) -> tuple[int | None, str | None, list[dict]]:
    """Pick the Heart Rate copy with the highest valid-sample fraction.

    Returns the chosen index/label plus an audit rows for every candidate.
    """
    audit: list[dict] = []
    best: tuple[float, int, str] | None = None
    for idx, label in candidates:
        x = reader(idx)
        v = valid_hr(x, cfg)
        frac = float(v.mean()) if len(x) else 0.0
        audit.append(
            {
                "channel_index": idx,
                "channel_label": label,
                "n_samples": int(len(x)),
                "valid_fraction": round(frac, 4),
                "median_bpm": round(float(np.median(x[v])), 2) if v.any() else None,
            }
        )
        if best is None or frac > best[0]:
            best = (frac, idx, label)
    if best is None:
        return None, None, audit
    return best[1], best[2], audit


def spo2_agreement(reader, indices: list[int], cfg: FeatureConfig) -> dict | None:
    """Compare Saturation and SpO2 B-B when both exist (audit only)."""
    if len(indices) < 2:
        return None
    a = reader(indices[0])
    b = reader(indices[1])
    va = valid_spo2(a, cfg)
    vb = valid_spo2(b, cfg)
    n = min(len(a), len(b))
    both = va[:n] & vb[:n]
    if both.sum() < 100:
        return {"n_overlap": int(both.sum()), "note": "insufficient overlap"}
    diff = a[:n][both] - b[:n][both]
    return {
        "n_overlap": int(both.sum()),
        "median_diff": round(float(np.median(diff)), 4),
        "p95_abs_diff": round(float(np.percentile(np.abs(diff), 95)), 4),
        "corr": round(float(np.corrcoef(a[:n][both], b[:n][both])[0, 1]), 6),
    }
