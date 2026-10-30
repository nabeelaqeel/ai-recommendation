"""Typed Phase 2 configuration.

The YAML file is the single source of truth: clinical band boundaries, the
participant label rule and every feature threshold are parameters, so the
pipeline can be re-run without code changes after clinical review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase2.yaml"

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
REPORTS = PROJECT_ROOT / "reports"

LABEL_DISPLAY = {
    "healthy": "Healthy band (<5)",
    "monitor": "Monitor band (5-15)",
    "see_doctor": "See Doctor band (>=15)",
    "elevated": "Elevated (>=15)",
    "low": "Low (<15)",
    "low_5": "Low (<5)",
    "elevated_5": "Elevated (>=5)",
}


@dataclass(frozen=True)
class ThreeClassBands:
    healthy_max: float = 5.0
    monitor_max: float = 15.0
    see_doctor_min: float = 15.0

    def __post_init__(self) -> None:
        if not (self.healthy_max < self.monitor_max == self.see_doctor_min):
            raise ValueError("three-class bands must satisfy healthy_max < monitor_max == see_doctor_min")


@dataclass(frozen=True)
class LabelConfig:
    three_class: ThreeClassBands = field(default_factory=ThreeClassBands)
    two_class_elevated_min: float = 15.0
    two_class_sensitivity_elevated_min: float = 5.0
    participant_rule: str = "worst_reliable_night"
    comparison_rule: str = "mean_reliable_night"
    min_tst_h: float = 2.0
    target_recall_see_doctor: float = 0.85
    non_diagnostic: bool = True

    def band3(self, ahi: float) -> str:
        """Research band for a single AHI value."""
        if ahi is None or ahi != ahi:  # NaN
            return "undefined"
        if ahi < self.three_class.healthy_max:
            return "healthy"
        if ahi < self.three_class.monitor_max:
            return "monitor"
        return "see_doctor"

    def band2(self, ahi: float, elevated_min: float | None = None) -> str:
        threshold = self.two_class_elevated_min if elevated_min is None else elevated_min
        if ahi is None or ahi != ahi:
            return "undefined"
        return "elevated" if ahi >= threshold else "low"

    def band2_display(self, ahi: float, elevated_min: float | None = None) -> str:
        threshold = self.two_class_elevated_min if elevated_min is None else elevated_min
        if ahi is None or ahi != ahi:
            return "undefined"
        return f">={threshold:g}" if ahi >= threshold else f"<{threshold:g}"


@dataclass(frozen=True)
class FeatureConfig:
    epoch_seconds: int = 30
    hr_valid_bpm: tuple[float, float] = (20.0, 250.0)
    spo2_valid_pct: tuple[float, float] = (50.0, 120.0)
    desat_threshold_pct: float = 90.0
    hrv_window_seconds: int = 60
    hrv_min_beats: int = 20
    rmssd_bounds_ms: tuple[float, float] = (5.0, 200.0)
    posture_change_deg: float = 30.0
    # Night-level plausibility gates. A night whose sleep HR or SpO2 distribution
    # is physiologically implausible (e.g. DS_97 median 145 bpm) is an artefact:
    # its values are removed and a flag is kept, never silently averaged in.
    hr_nightly_plausible_bpm: tuple[float, float] = (30.0, 120.0)
    spo2_nightly_plausible_pct: tuple[float, float] = (85.0, 100.0)
    hrv_min_nightly_epochs: int = 30


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int = 4
    pilot_nights: int = 30
    pilot_seed: int = 42


@dataclass(frozen=True)
class Phase2Config:
    labels: LabelConfig
    features: FeatureConfig
    runtime: RuntimeConfig
    source_path: Path = CONFIG_PATH

    @classmethod
    def load(cls, path: Path | None = None) -> "Phase2Config":
        path = Path(path) if path else CONFIG_PATH
        raw = yaml.safe_load(path.read_text())
        lab = raw.get("labels", {})
        tc = lab.get("three_class", {})
        labels = LabelConfig(
            three_class=ThreeClassBands(
                healthy_max=float(tc.get("healthy_max", 5)),
                monitor_max=float(tc.get("monitor_max", 15)),
                see_doctor_min=float(tc.get("see_doctor_min", 15)),
            ),
            two_class_elevated_min=float(lab.get("two_class", {}).get("elevated_min", 15)),
            two_class_sensitivity_elevated_min=float(
                lab.get("two_class_sensitivity", {}).get("elevated_min", 5)
            ),
            participant_rule=str(lab.get("participant_rule", "worst_reliable_night")),
            comparison_rule=str(lab.get("comparison_rule", "mean_reliable_night")),
            min_tst_h=float(lab.get("min_tst_h", 2.0)),
            target_recall_see_doctor=float(lab.get("target_recall_see_doctor", 0.85)),
            non_diagnostic=bool(lab.get("non_diagnostic", True)),
        )
        feat = raw.get("features", {})
        features = FeatureConfig(
            epoch_seconds=int(feat.get("epoch_seconds", 30)),
            hr_valid_bpm=tuple(float(v) for v in feat.get("hr_valid_bpm", (20, 250))),
            spo2_valid_pct=tuple(float(v) for v in feat.get("spo2_valid_pct", (50, 120))),
            desat_threshold_pct=float(feat.get("desat_threshold_pct", 90)),
            hrv_window_seconds=int(feat.get("hrv_window_seconds", 60)),
            hrv_min_beats=int(feat.get("hrv_min_beats", 20)),
            rmssd_bounds_ms=tuple(float(v) for v in feat.get("rmssd_bounds_ms", (5, 200))),
            posture_change_deg=float(feat.get("posture_change_deg", 30)),
            hr_nightly_plausible_bpm=tuple(float(v) for v in feat.get("hr_nightly_plausible_bpm", (30, 120))),
            spo2_nightly_plausible_pct=tuple(float(v) for v in feat.get("spo2_nightly_plausible_pct", (85, 100))),
            hrv_min_nightly_epochs=int(feat.get("hrv_min_nightly_epochs", 30)),
        )
        rt = raw.get("runtime", {})
        runtime = RuntimeConfig(
            workers=int(rt.get("workers", 4)),
            pilot_nights=int(rt.get("pilot_nights", 30)),
            pilot_seed=int(rt.get("pilot_seed", 42)),
        )
        return cls(labels=labels, features=features, runtime=runtime, source_path=path)


def load_config(path: Path | None = None) -> Phase2Config:
    return Phase2Config.load(path)
