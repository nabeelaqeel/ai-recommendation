"""Trained-model wrapper used by both evaluation and live inference.

Keeping one prediction interface means the offline reports and the rolling
buffer cannot diverge. Outputs are research guidance bands, never diagnoses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.config import Phase2Config, load_config

DISCLAIMER = (
    "Research guidance band only. Not a diagnosis, not clinically validated, "
    "and not a substitute for medical review."
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "rf_phase2.joblib"


@dataclass
class Phase2Classifier:
    model: object
    feature_names: list[str]
    arm: str
    label_order: list[str]
    label_rule: str
    threshold_2class: float = 0.5
    config_path: str = ""
    target_recall: float = 0.85
    fitted_on: str = ""
    positive_class: str = "see_doctor"  # class the cost-sensitive threshold applies to
    extra: dict = field(default_factory=dict)

    # -- persistence ------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Phase2Classifier":
        return joblib.load(path or MODEL_PATH)

    # -- inference --------------------------------------------------------
    def frame_from_dict(self, features: dict) -> pd.DataFrame:
        df = pd.DataFrame([features])
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise KeyError(f"missing feature values: {missing}")
        return df[self.feature_names]

    def _proba(self, features: dict | pd.DataFrame) -> tuple[np.ndarray, list]:
        """One predict_proba call on a numpy view (feature order is fixed)."""
        df = features if isinstance(features, pd.DataFrame) else self.frame_from_dict(features)
        proba = self.model.predict_proba(df.to_numpy(dtype=float))
        return proba, [str(c) for c in self.model.classes_]

    def predict_full(self, features: dict | pd.DataFrame) -> dict:
        """Single-inference view: band, probabilities and the threshold flag.

        One predict_proba call serves everything — the real-time budget does
        not allow paying model inference twice.
        """
        proba, classes = self._proba(features)
        p = {c: float(proba[0, j]) for j, c in enumerate(classes)}
        band = max(p, key=p.get)
        pos = self.positive_class if self.positive_class in p else classes[-1]
        return {
            "band": band,
            "probabilities": p,
            "p_positive": p[pos],
            "positive_class": pos,
            "flagged_positive": bool(p[pos] >= self.threshold_2class),
            "threshold": self.threshold_2class,
            "disclaimer": DISCLAIMER,
        }

    def predict_bands(self, features: dict | pd.DataFrame) -> list[dict]:
        """Back-compatible batch view (one inference per call)."""
        df = features if isinstance(features, pd.DataFrame) else self.frame_from_dict(features)
        proba = self.model.predict_proba(df.to_numpy(dtype=float))
        classes = [str(c) for c in self.model.classes_]
        out = []
        for i in range(len(df)):
            p = {c: float(proba[i, j]) for j, c in enumerate(classes)}
            out.append({"band": max(p, key=p.get), "probabilities": p, "disclaimer": DISCLAIMER})
        return out

    def predict_elevated(self, features: dict | pd.DataFrame) -> list[dict]:
        """Binary view at the cost-sensitive threshold (one inference per call)."""
        df = features if isinstance(features, pd.DataFrame) else self.frame_from_dict(features)
        proba = self.model.predict_proba(df.to_numpy(dtype=float))
        classes = [str(c) for c in self.model.classes_]
        pos = self.positive_class if self.positive_class in classes else classes[-1]
        j = classes.index(pos)
        return [
            {
                "p_elevated": float(proba[i, j]),
                "flagged_elevated": bool(proba[i, j] >= self.threshold_2class),
                "threshold": self.threshold_2class,
                "disclaimer": DISCLAIMER,
            }
            for i in range(len(df))
        ]


def default_config() -> Phase2Config:
    return load_config()
