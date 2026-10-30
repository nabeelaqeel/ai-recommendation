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

    def predict_bands(self, features: dict | pd.DataFrame) -> list[dict]:
        """Return band, class probabilities and the non-diagnostic disclaimer."""
        df = features if isinstance(features, pd.DataFrame) else self.frame_from_dict(features)
        proba = self.model.predict_proba(df)
        classes = list(self.model.classes_)
        out = []
        for i in range(len(df)):
            p = {self._band(c): float(proba[i, j]) for j, c in enumerate(classes)}
            band = max(p, key=p.get)
            out.append(
                {
                    "band": band,
                    "probabilities": p,
                    "disclaimer": DISCLAIMER,
                }
            )
        return out

    def _band(self, sklearn_class) -> str:
        return str(sklearn_class)

    def predict_elevated(self, features: dict | pd.DataFrame) -> list[dict]:
        """Binary >=threshold view with the cost-sensitive operating point."""
        df = features if isinstance(features, pd.DataFrame) else self.frame_from_dict(features)
        proba = self.model.predict_proba(df)
        classes = list(self.model.classes_)
        pos = classes.index("elevated") if "elevated" in classes else 1
        out = []
        for i in range(len(df)):
            p_pos = float(proba[i, pos])
            flagged = p_pos >= self.threshold_2class
            out.append(
                {
                    "p_elevated": p_pos,
                    "flagged_elevated": bool(flagged),
                    "threshold": self.threshold_2class,
                    "disclaimer": DISCLAIMER,
                }
            )
        return out


def default_config() -> Phase2Config:
    return load_config()
