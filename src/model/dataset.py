"""Build the participant-level modelling table from nightly features + labels.

Labels are recomputed from `outputs/paaws_night_metrics.csv` using the Phase 2
config bands, never copied from Phase 1 strings, so a band change in
`config/phase2.yaml` propagates everywhere on re-run.

Feature construction rules
--------------------------
* One row per participant. Features are the nan-mean over that participant's
  reliable nights (TST >= cfg min, valid AHI, successful extraction).
  Selecting the "worst" night BY AHI would leak the label into feature
  construction, so aggregation is label-blind.
* Artefact flags aggregate as "any night flagged"; sufficiency as "any night
  sufficient"; coverage as nan-mean.
* Model inputs come only from an explicit allow-list (FEATURE_GROUPS), and a
  hard assertion rejects any column matching a leakage pattern.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.audit.common import OUTPUTS, PROJECT_ROOT
from src.features.config import Phase2Config, load_config
from src.features.features import FORBIDDEN_FEATURE_PATTERNS

COMBINED_NIGHTLY = PROJECT_ROOT / "data" / "processed" / "features_night.parquet"
EXTRACTION_AUDIT = OUTPUTS / "phase2_extraction_audit.csv"
NIGHT_METRICS = OUTPUTS / "paaws_night_metrics.csv"
PARTICIPANT_TABLE = PROJECT_ROOT / "data" / "processed" / "participant_table.parquet"

# Explicit allow-list. Nothing enters a model unless listed here.
FEATURE_GROUPS: dict[str, list[str]] = {
    "hr": ["hr_median", "hr_iqr", "hr_p5", "hr_p95", "hr_std"],
    "hrv": ["rmssd_median", "rmssd_iqr"],
    "spo2": [
        "spo2_median",
        "spo2_iqr",
        "spo2_p5",
        "spo2_min",
        "spo2_below90_frac",
        "spo2_t90_per_valid_h",
    ],
    "activity": ["activity_mean", "activity_std", "activity_p95"],
    "posangle": ["posangle_median", "posangle_iqr", "posangle_change_count", "posangle_delta_median", "posangle_r_mean"],
    "coverage": [
        "hr_coverage",
        "spo2_coverage",
        "spo2_valid_h",
        "hrv_valid_frac",
        "hr_artefact",
        "spo2_artefact",
        "hrv_sufficient",
    ],
}

# Ablation arms (Stage 3). `core` is the Garmin-strict set: HR, HRV, SpO2/T90,
# Activity. PosAngle is excluded by default because a wrist device does not
# measure torso angle; coverage flags are excluded so the RF cannot use sensor
# dropout as a severity shortcut.
ARMS: dict[str, list[str]] = {
    "core": FEATURE_GROUPS["hr"] + FEATURE_GROUPS["hrv"] + FEATURE_GROUPS["spo2"] + FEATURE_GROUPS["activity"],
    "core_posangle": FEATURE_GROUPS["hr"] + FEATURE_GROUPS["hrv"] + FEATURE_GROUPS["spo2"] + FEATURE_GROUPS["activity"] + FEATURE_GROUPS["posangle"],
    "core_coverage": FEATURE_GROUPS["hr"] + FEATURE_GROUPS["hrv"] + FEATURE_GROUPS["spo2"] + FEATURE_GROUPS["activity"] + FEATURE_GROUPS["coverage"],
    "full": FEATURE_GROUPS["hr"] + FEATURE_GROUPS["hrv"] + FEATURE_GROUPS["spo2"] + FEATURE_GROUPS["activity"] + FEATURE_GROUPS["posangle"] + FEATURE_GROUPS["coverage"],
}

LABEL_3 = ["healthy", "monitor", "see_doctor"]
LABEL_2 = ["low", "elevated"]

log = logging.getLogger("dataset")


def reliable_night_mask(metrics: pd.DataFrame, cfg: Phase2Config) -> pd.Series:
    return metrics["ahi_valid"].astype(bool) & (metrics["tst_h"] >= cfg.labels.min_tst_h)


def _aggregate_participant(g: pd.DataFrame) -> pd.Series:
    """Label-blind aggregation of one participant's reliable nights."""
    out: dict[str, float] = {}
    mean_cols = (
        FEATURE_GROUPS["hr"]
        + FEATURE_GROUPS["hrv"]
        + FEATURE_GROUPS["spo2"]
        + FEATURE_GROUPS["activity"]
        + FEATURE_GROUPS["posangle"]
        + ["hr_coverage", "spo2_coverage", "spo2_valid_h", "hrv_valid_frac"]
    )
    for c in mean_cols:
        if c in g.columns:
            v = pd.to_numeric(g[c], errors="coerce")
            out[c] = float(v.mean()) if v.notna().any() else np.nan
    for c in ("hr_artefact", "spo2_artefact"):
        out[c] = bool(g[c].astype(bool).any()) if c in g.columns else False
    out["hrv_sufficient"] = bool(g["hrv_sufficient"].astype(bool).any()) if "hrv_sufficient" in g.columns else False
    return pd.Series(out)


def build_participant_table(cfg: Phase2Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    nightly = pd.read_parquet(COMBINED_NIGHTLY)
    metrics = pd.read_csv(NIGHT_METRICS).set_index("night_id")
    audit = pd.read_csv(EXTRACTION_AUDIT)
    extracted_ok = set(audit.loc[audit["status"] == "ok", "night_id"])

    m = metrics.copy()
    m["reliable"] = reliable_night_mask(m, cfg)
    reliable_ids = set(m.index[m["reliable"]]) & extracted_ok
    dropped_unextracted = sorted(set(m.index[m["reliable"]]) - extracted_ok)
    if dropped_unextracted:
        log.warning("Reliable nights without successful extraction (excluded): %s", dropped_unextracted)

    nightly = nightly[nightly["night_id"].isin(reliable_ids)].copy()
    nightly["participant_id"] = nightly["participant_id"].astype(str)

    # AHI per reliable night, for label construction only.
    ahi = m.loc[sorted(reliable_ids), ["ahi_sleep"]]
    nightly = nightly.merge(ahi, left_on="night_id", right_index=True, how="left")
    expected_pid = nightly["night_id"].str.split("-").str[0]
    if not (nightly["participant_id"].astype(str) == expected_pid).all():
        raise AssertionError("participant_id in nightly table disagrees with night_id prefix")

    feat_cols = sorted({c for cols in FEATURE_GROUPS.values() for c in cols})
    agg = nightly.groupby("participant_id", sort=True)[feat_cols].apply(
        _aggregate_participant, include_groups=False
    )

    lab = nightly.groupby("participant_id").agg(
        ahi_max=("ahi_sleep", "max"),
        ahi_mean=("ahi_sleep", "mean"),
        n_reliable_nights=("night_id", "count"),
    )
    table = agg.join(lab)

    rule = cfg.labels.participant_rule
    stat = "ahi_max" if rule == "worst_reliable_night" else "ahi_mean"
    table["label_3class"] = table[stat].map(cfg.labels.band3)
    table["label_3class_comparison"] = table["ahi_mean" if stat == "ahi_max" else "ahi_max"].map(cfg.labels.band3)
    table["label_2class"] = table[stat].map(lambda a: cfg.labels.band2(a))
    table["label_2class_sensitivity"] = table[stat].map(
        lambda a: "elevated_5" if a >= cfg.labels.two_class_sensitivity_elevated_min else "low_5"
    )
    table["label_rule"] = rule
    table = table.reset_index()
    return table


def arm_columns(arm: str, table: pd.DataFrame) -> list[str]:
    cols = ARMS[arm]
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise KeyError(f"arm {arm}: columns missing from table: {missing}")
    leaks = [c for c in cols if any(p in c.lower() for p in FORBIDDEN_FEATURE_PATTERNS)]
    if leaks:
        raise AssertionError(f"arm {arm}: leakage-suspect columns: {leaks}")
    return cols


def assert_no_leakage_columns(cols: list[str]) -> None:
    bad = [c for c in cols if any(p in c.lower() for p in FORBIDDEN_FEATURE_PATTERNS)]
    if bad:
        raise AssertionError(f"leakage-suspect columns: {bad}")


def load_or_build(cfg: Phase2Config | None = None, force: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    if not force and PARTICIPANT_TABLE.exists():
        t = pd.read_parquet(PARTICIPANT_TABLE)
        if t["label_rule"].iloc[0] == cfg.labels.participant_rule:
            return t
    t = build_participant_table(cfg)
    t.to_parquet(PARTICIPANT_TABLE, index=False)
    log.info("Participant table: %d rows -> %s", len(t), PARTICIPANT_TABLE)
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()
    t = build_participant_table(cfg)
    t.to_parquet(PARTICIPANT_TABLE, index=False)
    print(f"participants: {len(t)}")
    print(t["label_3class"].value_counts().to_string())
    print(t["label_2class"].value_counts().to_string())
    print(t["label_2class_sensitivity"].value_counts().to_string())
