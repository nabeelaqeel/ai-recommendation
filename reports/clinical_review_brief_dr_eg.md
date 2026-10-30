# Clinical review brief — AI recommendation module (Phase 1 + Phase 2 baseline)

**For:** Assoc. Prof. Dr. Eg Kah Peng, UM Faculty of Medicine  
**From:** FYP project — QoS-Aware Transmission Protocol for Continuous Sleep Monitoring Data  
**Date:** generated from the committed pipeline outputs  
**Status:** all results are provisional pending this review.

---
## 1. What this module is (and is not)

A supplementary Random Forest layer that maps overnight physiological features
to three **proposed guidance indicators** — Healthy / Monitor / See Doctor —
intended to feed the RB-QoSPCTF QoS tiering. It is **not** a diagnostic device:
it does not diagnose sleep apnoea or any condition, has not been validated on
wrist-worn data, and must not inform clinical decisions in its current form.

## 2. Data and label basis (Phase 1 audit)

- PAAWS Sleep Protocol R2 v1.3: **139 analysable participants** (141 recruited; 2 have no reliable night), 253 nights, PSG.
- AHI is a **research calculation**: scored respiratory events ÷ scored sleep time.
  Event definitions are inferred from the scored-events CSVs; the PAAWS codebook
  was not available for verification.
- Participant bands (worst reliable night): Healthy <5: **68**, Monitor 5–15: **47**, See Doctor ≥15: **24**.
- Features use only Garmin-plausible signals: HR, HRV (RMSSD), SpO2/T90, motion;
  torso angle is evaluated separately because a wrist device does not measure it.

## 3. Baseline performance (Phase 2, participant-grouped CV)

| model | macro-F1 (mean ± sd) | pooled [95% CI] | balanced acc |
|---|---|---|---|
| Random Forest, Garmin-strict features (HR, RMSSD, SpO2/T90, Activity) | 0.514 ± 0.083 | 0.520 [0.436–0.607] | 0.530 |
| + torso angle (PosAngle) | 0.510 ± 0.077 | 0.521 [0.438–0.598] | 0.529 |
| + sensor-coverage flags | 0.527 ± 0.077 | 0.566 [0.482–0.646] | 0.543 |
| all features | 0.547 ± 0.077 | 0.582 [0.501–0.658] | 0.568 |
| all features, mean-night label rule | 0.527 ± 0.098 | 0.580 [0.496–0.659] | 0.551 |
| logistic regression comparator | 0.485 ± 0.092 | 0.492 [0.408–0.569] | 0.508 |
| night-level rows (grouped sensitivity analysis) | 0.575 ± 0.071 | 0.604 [0.530–0.669] | 0.578 |

### See-Doctor (≥15) detection at the cost-sensitive operating point

Thresholds chosen inside training folds to target ≥85% recall on ≥15, then
applied to untouched test folds.

| model | recall (≥15) | specificity | precision | threshold |
|---|---:|---:|---:|---:|
| 3-class core (deployable), P(See Doctor) ≥ t | 0.876 | 0.615 | 0.343 | 0.13 |
| 3-class all features, P(See Doctor) ≥ t | 0.874 | 0.621 | 0.357 | 0.14 |
| 2-class ≥15 vs <15 | 0.866 | 0.633 | 0.344 | 0.18 |
| 2-class ≥5 vs <5 (sensitivity split) | 0.859 | 0.359 | 0.588 | 0.32 |

## 4. Known limitations

- **Domain gap:** training features come from PSG (ECG-derived HRV at 200 Hz,
  3 Hz oximetry); deployment is wrist-worn PPG with intermittent SpO2. No
  Garmin data has been used at any stage; transfer is unmeasured.
- **Small See-Doctor class** (~24 participants): confidence intervals are wide;
  per-fold recall varies substantially.
- **Uncalibrated scores:** predict_proba outputs are model scores, not
  clinical probabilities (mean per-class Brier ≈ 0.1779).
- **AHI is an inferred research estimate**, not a centrally re-scored clinical AHI.
- Structured missingness (no SpO2 channel on 8/253 nights; unusable ECG-HRV on a
  minority of nights) is preserved as NaN + flags, never imputed.
- **Real-time feasibility:** single classification from a 10-minute rolling
  buffer takes p50 16.2 ms / p95 17.4 ms
  (budget 50 ms), using the same feature code as training.

---
## 5. Four questions for your review

1. **Band boundaries.** AASM thresholds <5 / 5–15 / ≥15 map to Healthy / Monitor /
   See Doctor. Is this mapping appropriate for *guidance* (not diagnosis)? Should
   the See-Doctor threshold be ≥15 or higher (e.g. ≥20) for a consumer-facing band?
2. **Multi-night label rule.** Primary rule: worst reliable night (screening-style
   caution); comparison: mean of reliable nights. Which should the dashboard use?
   (Performance difference is reported above: worst 0.547 vs mean 0.527 macro-F1.)
3. **Acceptable false-negative rate for See Doctor.** Current target: ≥85% recall
   on ≥15 with the threshold that maximises specificity subject to that recall.
   Is 85% recall acceptable, or should this be stricter (e.g. 90–95%), accepting
   more false alarms? The specificity/precision trade-off at the current operating
   point is in the table above.
4. **Non-diagnostic framing.** All outputs carry the wording: research guidance
   band, not a diagnosis, not clinically validated. Is this sufficient for the
   project's documentation and dashboard, or do you require additional disclaimers?

---
## 6. What happens after sign-off

Every choice above is a config parameter (`config/phase2.yaml`); the pipeline
(feature extraction → labels → model → threshold → reports) re-runs end-to-end
with no code changes. Any band, rule or recall target you specify will propagate
to the model, the confusion matrices, the operating point and this brief.

*Generated from `outputs/phase2_model_summary.csv`, `outputs/phase2_fold_records.csv`
and the Phase 1 audit tables. Participant-level predictions are excluded from this
brief deliberately.*
