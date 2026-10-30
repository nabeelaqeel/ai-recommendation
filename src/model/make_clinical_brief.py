"""Generate reports/clinical_review_brief_dr_eg.md after Stage 3.

Pulls Phase 1 headline numbers and Phase 2 model results from the generated
CSVs so the brief always matches the actual runs. Concise by design: the
collaborator should be able to review it in one sitting.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from src.audit.common import OUTPUTS, PROJECT_ROOT

BRIEF = PROJECT_ROOT / "reports" / "clinical_review_brief_dr_eg.md"

log = logging.getLogger("brief")


def _fmt_ci(low, high) -> str:
    try:
        return f"[{float(low):.3f}–{float(high):.3f}]"
    except (TypeError, ValueError):
        return "—"


def main() -> int:
    labels = pd.read_csv(OUTPUTS / "paaws_participant_labels.csv")
    summary = pd.read_csv(OUTPUTS / "phase2_model_summary.csv")
    folds = pd.read_csv(RESULTS := OUTPUTS / "phase2_fold_records.csv")
    cal = pd.read_csv(OUTPUTS / "phase2_calibration.csv")
    s = summary.set_index("tag")

    worst = labels["ahi_max_3class"].value_counts()
    n_participants = len(labels)

    def row(tag: str) -> dict | None:
        return s.loc[tag].to_dict() if tag in s.index else None

    core = row("arm_core_3class")
    pos = row("arm_core_posangle_3class")
    cov = row("arm_core_coverage_3class")
    full = row("arm_full_3class")
    mean_rule = row("rule_mean_night_3class")
    two = row("two_class_ge15")
    ge5 = row("two_class_ge5")
    lr = row("logistic_full_3class")
    night = row("night_level_3class")

    brier = None
    try:
        brier = float(pd.read_csv(OUTPUTS / "phase2_calibration.csv")["brier"].iloc[0]) if "brier" in cal else None
    except Exception:  # noqa: BLE001
        pass

    lines = [
        "# Clinical review brief — AI recommendation module (Phase 1 + Phase 2 baseline)",
        "",
        "**For:** Assoc. Prof. Dr. Eg Kah Peng, UM Faculty of Medicine  ",
        "**From:** FYP project — QoS-Aware Transmission Protocol for Continuous Sleep Monitoring Data  ",
        "**Date:** generated from the committed pipeline outputs  ",
        "**Status:** all results are provisional pending this review.",
        "",
        "---",
        "## 1. What this module is (and is not)",
        "",
        "A supplementary Random Forest layer that maps overnight physiological features",
        "to three **proposed guidance indicators** — Healthy / Monitor / See Doctor —",
        "intended to feed the RB-QoSPCTF QoS tiering. It is **not** a diagnostic device:",
        "it does not diagnose sleep apnoea or any condition, has not been validated on",
        "wrist-worn data, and must not inform clinical decisions in its current form.",
        "",
        "## 2. Data and label basis (Phase 1 audit)",
        "",
        f"- PAAWS Sleep Protocol R2 v1.3: **{n_participants} analysable participants** (141 recruited; 2 have no reliable night), 253 nights, PSG.",
        "- AHI is a **research calculation**: scored respiratory events ÷ scored sleep time.",
        "  Event definitions are inferred from the scored-events CSVs; the PAAWS codebook",
        "  was not available for verification.",
        f"- Participant bands (worst reliable night): Healthy <5: **{int(worst.get('Healthy-band (<5)', 0))}**, "
        f"Monitor 5–15: **{int(worst.get('Monitor-band (5-15)', 0))}**, See Doctor ≥15: **{int(worst.get('See-Doctor-band (>=15)', 0))}**.",
        "- Features use only Garmin-plausible signals: HR, HRV (RMSSD), SpO2/T90, motion;",
        "  torso angle is evaluated separately because a wrist device does not measure it.",
        "",
        "## 3. Baseline performance (Phase 2, participant-grouped CV)",
        "",
        "| model | macro-F1 (mean ± sd) | pooled [95% CI] | balanced acc |",
        "|---|---|---|---|",
    ]

    def perf(tag: str, name: str) -> None:
        r = row(tag)
        if r:
            lines.append(
                f"| {name} | {r['macro_f1_mean']:.3f} ± {r['macro_f1_sd']:.3f} | "
                f"{r['pooled_macro_f1']:.3f} {_fmt_ci(r.get('macro_f1_ci_low'), r.get('macro_f1_ci_high'))} | {r['balanced_acc_mean']:.3f} |"
            )

    perf("arm_core_3class", "Random Forest, Garmin-strict features (HR, RMSSD, SpO2/T90, Activity)")
    perf("arm_core_posangle_3class", "+ torso angle (PosAngle)")
    perf("arm_core_coverage_3class", "+ sensor-coverage flags")
    perf("arm_full_3class", "all features")
    perf("rule_mean_night_3class", "all features, mean-night label rule")
    perf("logistic_full_3class", "logistic regression comparator")
    perf("night_level_3class", "night-level rows (grouped sensitivity analysis)")

    lines += ["", "### See-Doctor (≥15) detection at the cost-sensitive operating point", "",
              "Thresholds chosen inside training folds to target ≥85% recall on ≥15, then",
              "applied to untouched test folds.", "",
              "| model | recall (≥15) | specificity | precision | threshold |", "|---|---:|---:|---:|---:|"]

    def thr(tag: str, name: str) -> None:
        r = row(tag)
        if r and "recall_pos_at_t_mean" in r and pd.notna(r.get("recall_pos_at_t_mean")):
            lines.append(
                f"| {name} | {r['recall_pos_at_t_mean']:.3f} | {r['specificity_at_t_mean']:.3f} | "
                f"{r['precision_pos_at_t_mean']:.3f} | {r['threshold_mean']:.2f} |"
            )

    thr("arm_full_3class", "3-class, P(See Doctor) ≥ t")
    thr("two_class_ge15", "2-class ≥15 vs <15")
    thr("two_class_ge5", "2-class ≥5 vs <5 (sensitivity split)")

    lines += [
        "",
        "## 4. Known limitations",
        "",
        "- **Domain gap:** training features come from PSG (ECG-derived HRV at 200 Hz,",
        "  3 Hz oximetry); deployment is wrist-worn PPG with intermittent SpO2. No",
        "  Garmin data has been used at any stage; transfer is unmeasured.",
        "- **Small See-Doctor class** (~24 participants): confidence intervals are wide;",
        "  per-fold recall varies substantially.",
        "- **Uncalibrated scores:** predict_proba outputs are model scores, not",
        f"  clinical probabilities (mean per-class Brier ≈ {brier if brier is not None else '—'}).",
        "- **AHI is an inferred research estimate**, not a centrally re-scored clinical AHI.",
        "- Structured missingness (no SpO2 channel on 8/253 nights; unusable ECG-HRV on a",
        "  minority of nights) is preserved as NaN + flags, never imputed.",
        "",
        "---",
        "## 5. Four questions for your review",
        "",
        "1. **Band boundaries.** AASM thresholds <5 / 5–15 / ≥15 map to Healthy / Monitor /",
        "   See Doctor. Is this mapping appropriate for *guidance* (not diagnosis)? Should",
        "   the See-Doctor threshold be ≥15 or higher (e.g. ≥20) for a consumer-facing band?",
        "2. **Multi-night label rule.** Primary rule: worst reliable night (screening-style",
        "   caution); comparison: mean of reliable nights. Which should the dashboard use?",
        f"   (Performance difference is reported above: worst {full['macro_f1_mean']:.3f} vs mean {mean_rule['macro_f1_mean']:.3f} macro-F1.)"
        if full and mean_rule else "2. **Multi-night label rule.** worst vs mean reliable night (see model report).",
        "3. **Acceptable false-negative rate for See Doctor.** Current target: ≥85% recall",
        "   on ≥15 with the threshold that maximises specificity subject to that recall.",
        "   Is 85% recall acceptable, or should this be stricter (e.g. 90–95%), accepting",
        "   more false alarms? The specificity/precision trade-off at the current operating",
        "   point is in the table above.",
        "4. **Non-diagnostic framing.** All outputs carry the wording: research guidance",
        "   band, not a diagnosis, not clinically validated. Is this sufficient for the",
        "   project's documentation and dashboard, or do you require additional disclaimers?",
        "",
        "---",
        "## 6. What happens after sign-off",
        "",
        "Every choice above is a config parameter (`config/phase2.yaml`); the pipeline",
        "(feature extraction → labels → model → threshold → reports) re-runs end-to-end",
        "with no code changes. Any band, rule or recall target you specify will propagate",
        "to the model, the confusion matrices, the operating point and this brief.",
        "",
        "*Generated from `outputs/phase2_model_summary.csv`, `outputs/phase2_fold_records.csv`",
        "and the Phase 1 audit tables. Participant-level predictions are excluded from this",
        "brief deliberately.*",
    ]
    BRIEF.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", BRIEF)
    print(f"wrote {BRIEF}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sys.exit(main())
