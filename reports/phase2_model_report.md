# Phase 2 model report — participant-level Random Forest (research bands)

> **Non-diagnostic.** Healthy / Monitor / See Doctor are proposed sleep-related guidance
> bands for a supplementary recommendation layer. This model does not diagnose sleep
> apnoea, has not been validated on Garmin data, and `predict_proba` outputs are
> uncalibrated model scores, not clinical probabilities.

- Participants: **139** (label rule: `worst_reliable_night`)
- Class counts (worst reliable night): {'healthy': 68, 'monitor': 47, 'see_doctor': 24}
- CV: StratifiedGroupKFold 5 folds × 5 repeats, participant-grouped;
  hyperparameters selected inside training folds only.

## Ablation arms (3-class, primary label rule)

| arm | features | macro-F1 (mean±sd) | pooled macro-F1 [95% CI] | balanced acc |
|---|---|---|---|---|
| arm_core_3class | HR, RMSSD, SpO2/T90, Activity (Garmin-strict) | 0.514 ± 0.083 | 0.520 [0.436–0.607] | 0.530 |
| arm_core_posangle_3class | core + PosAngle | 0.510 ± 0.077 | 0.521 [0.438–0.598] | 0.529 |
| arm_core_coverage_3class | core + coverage flags | 0.527 ± 0.077 | 0.566 [0.482–0.646] | 0.543 |
| arm_full_3class | core + PosAngle + coverage flags | 0.547 ± 0.077 | 0.582 [0.501–0.658] | 0.568 |
| rule_mean_night_3class | full arm, mean-night label rule (comparison) | 0.527 ± 0.098 | 0.580 [0.496–0.659] | 0.551 |
| two_class_ge15 | full arm, binary >=15 with cost-sensitive threshold | 0.736 ± 0.098 | 0.739 [0.627–0.834] | 0.729 |
| two_class_ge5 | full arm, binary >=5 (sensitivity split) | 0.681 ± 0.072 | 0.691 [0.617–0.763] | 0.683 |
| logistic_full_3class | logistic regression comparator (full arm) | 0.485 ± 0.092 | 0.492 [0.408–0.569] | 0.508 |
| night_level_3class | night-level rows, grouped by participant | 0.575 ± 0.071 | 0.604 [0.530–0.669] | 0.578 |

## Cost-sensitive operating point (See-Doctor recall target 85%)

Thresholds are chosen on inner-CV out-of-fold probabilities within each training fold,
then applied to the untouched test fold — the test fold never influences the threshold.

### 3-class: flag See-Doctor when P(see_doctor) >= t

| arm | metric | mean ± sd |
|---|---|---|
| core (deployable) | decision threshold | 0.129 ± 0.044 |
| core (deployable) | See-Doctor recall at threshold | 0.876 ± 0.155 |
| core (deployable) | specificity (non-See-Doctor correctly not flagged) | 0.615 ± 0.154 |
| core (deployable) | precision of See-Doctor flag | 0.343 ± 0.093 |
| core (deployable) | fraction of participants flagged | 0.469 ± 0.144 |
| core (deployable) | See-Doctor recall at default argmax | 0.576 ± 0.242 |
| full | decision threshold | 0.144 ± 0.051 |
| full | See-Doctor recall at threshold | 0.874 ± 0.173 |
| full | specificity (non-See-Doctor correctly not flagged) | 0.621 ± 0.157 |
| full | precision of See-Doctor flag | 0.357 ± 0.155 |
| full | fraction of participants flagged | 0.465 ± 0.147 |
| full | See-Doctor recall at default argmax | 0.634 ± 0.222 |

### 2-class (>=15 vs <15), full arm

| metric | mean ± sd |
|---|---|
| decision threshold | 0.182 ± 0.064 |
| recall (>=15) at threshold | 0.866 ± 0.181 |
| specificity at threshold | 0.633 ± 0.130 |
| precision at threshold | 0.344 ± 0.097 |
| fraction flagged elevated | 0.453 ± 0.117 |
| recall at default 0.5 | 0.518 ± 0.229 |

## Confusion matrix (pooled out-of-fold, full arm, 3-class)

| true \ pred | healthy | monitor | see_doctor |
|---|---:|---:|---:|
| **healthy** | 57 | 11 | 0 |
| **monitor** | 26 | 12 | 9 |
| **see_doctor** | 2 | 5 | 17 |

## Per-class recall/precision (full arm, mean over folds)

| class | recall | precision |
|---|---:|---:|
| healthy | 0.797 | 0.659 |
| monitor | 0.273 | 0.405 |
| see_doctor | 0.634 | 0.639 |

## Permutation importance (full arm, top 12)

| feature | importance |
|---|---:|
| spo2_median | 0.1351 |
| spo2_p5 | 0.0631 |
| spo2_coverage | 0.0600 |
| posangle_delta_median | 0.0559 |
| spo2_below90_frac | 0.0467 |
| spo2_t90_per_valid_h | 0.0380 |
| posangle_r_mean | 0.0372 |
| hr_p5 | 0.0258 |
| activity_p95 | 0.0252 |
| posangle_iqr | 0.0236 |
| hrv_valid_frac | 0.0227 |
| rmssd_median | 0.0220 |

## Calibration (pooled out-of-fold, full arm)

- Mean per-class Brier score: **0.1779** (lower is better; 3-class)

| class | bin | n | mean predicted | observed |
|---|---|---:|---:|---:|
| healthy | 0 | 28 | 0.1535 | 0.0714 |
| healthy | 1 | 28 | 0.3477 | 0.3929 |
| healthy | 2 | 27 | 0.4777 | 0.6667 |
| healthy | 3 | 28 | 0.5887 | 0.6429 |
| healthy | 4 | 28 | 0.7046 | 0.6786 |
| monitor | 0 | 28 | 0.2322 | 0.3929 |
| monitor | 1 | 28 | 0.302 | 0.4286 |
| monitor | 2 | 27 | 0.3466 | 0.2593 |
| monitor | 3 | 28 | 0.3993 | 0.2143 |
| monitor | 4 | 28 | 0.4827 | 0.3929 |
| see_doctor | 0 | 28 | 0.018 | 0.0 |
| see_doctor | 1 | 28 | 0.0562 | 0.0357 |
| see_doctor | 2 | 27 | 0.1213 | 0.1481 |
| see_doctor | 3 | 28 | 0.2503 | 0.0357 |
| see_doctor | 4 | 28 | 0.5172 | 0.6429 |

## Interpretation notes

- Feature construction is label-blind (nan-mean over reliable nights); the worst night is
  selected only for the label, never for the features.
- Coverage flags are excluded from the core arm so the model cannot use sensor dropout
  as a severity shortcut; the `core_coverage` arm quantifies that risk.
- All scores are research estimates on PSG data. The PSG → wrist-worn domain gap
  (ECG vs PPG HRV, 3 Hz vs intermittent SpO2) is unresolved and unmeasured here.

Research guidance band only. Not a diagnosis, not clinically validated, and not a substitute for medical review.
