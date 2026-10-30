# Phase 2 feature dictionary

Every model input, its definition, and its Garmin analogue. All features are
computed **only over scored sleep epochs** in training (30-s epochs from the
PAAWS `scored_events.csv`; Wake and unscored gaps excluded). At run time the
rolling buffer applies the same statistics to whatever the device supplies
(no sleep staging exists on-device — documented domain difference).

Epoch-level statistics are aggregated twice: per epoch (30 s), then per night
(median/IQR/percentiles), then per participant (nan-mean over reliable nights,
label-blind — the "worst night" is used for the label only, never for features).

## HR — heart rate (channel `Heart Rate`, 3 Hz, bpm, missing code 0.01)

| Feature | Definition | Garmin analogue |
|---|---|---|
| `hr_median` | median of per-epoch median HR over the night | overnight HR summary |
| `hr_iqr` | IQR of per-epoch median HR | HR variability (coarse) |
| `hr_p5`, `hr_p95` | 5th/95th percentile of per-epoch median HR | min/max overnight HR |
| `hr_std` | SD of per-epoch median HR | — |

## HRV — heart-rate variability (ECG/EKG 200 Hz → per-epoch RMSSD)

RMSSD per 30-s epoch from R-peaks (5–15 Hz band-pass, local threshold), gated
by rate agreement with the recorded HR (0.7–1.3×), ≥20 RR intervals (0.3–2.0 s),
and a physiological bound (5–200 ms). Causal — no look-ahead — so live inference
is identical. **PPG is never used as a fallback.**

| Feature | Definition | Garmin analogue |
|---|---|---|
| `rmssd_median` | nightly median of valid per-epoch RMSSD (NaN if <30 valid epochs) | overnight HRV (Garmin computes RMSSD itself) |
| `rmssd_iqr` | IQR of valid per-epoch RMSSD | — |

## SpO2 — oxygen saturation (`Saturation` preferred over `SpO2 B-B`, 3 Hz, %, missing code −0.05)

| Feature | Definition | Garmin analogue |
|---|---|---|
| `spo2_median` | median of per-epoch mean SpO2 | overnight SpO2 average |
| `spo2_iqr`, `spo2_p5` | variability of per-epoch mean SpO2 | — |
| `spo2_min` | nightly minimum of per-epoch minima | lowest SpO2 |
| `spo2_below90_frac` | mean fraction of valid samples <90% per epoch | time below 90% |
| `spo2_t90_per_valid_h` | T90 hours per **valid** hour (normalised by coverage) | Garmin does not expose T90 directly |

## Activity — motion (channel `Activity`, 20 Hz; saturates at 10.0)

| Feature | Definition | Garmin analogue |
|---|---|---|
| `activity_mean` | mean of per-epoch mean activity | sleep-restlessness metrics |
| `activity_std` | mean per-epoch SD | movement variability |
| `activity_p95` | 95th percentile of per-epoch p95 | movement spikes |

## PosAngle — torso angle (20 Hz, signed ±360°, circular statistics)

Excluded from the Garmin-strict arm: a wrist device does not measure torso angle.

| Feature | Definition |
|---|---|
| `posangle_median` | circular mean of per-epoch orientation |
| `posangle_iqr` | circular spread around the nightly mean |
| `posangle_change_count` | epochs with circular change >30° from previous |
| `posangle_delta_median` | median absolute circular change between epochs |
| `posangle_r_mean` | mean resultant length (1 = motionless orientation) |

## Coverage / quality flags (excluded from the core arm by design)

Included so missingness is explicit; excluded from `core` so the RF cannot use
sensor dropout as a severity shortcut (ablation `core_coverage` quantifies that).

| Feature | Definition |
|---|---|
| `hr_coverage` | mean fraction of valid HR samples per epoch |
| `spo2_coverage` | mean fraction of valid SpO2 samples per epoch |
| `spo2_valid_h` | hours of valid SpO2 within sleep |
| `hrv_valid_frac` | fraction of epochs with QC-passed RMSSD |
| `hr_artefact` | nightly HR distribution implausible (e.g. DS_97, median ≈145 bpm) → values withdrawn |
| `spo2_artefact` | nightly SpO2 median outside 85–100% → values withdrawn |
| `hrv_sufficient` | ≥30 QC-passed HRV epochs in at least one night |

## Missingness policy

Absent channels, artefact nights and insufficient-HRV nights are **NaN, never
imputed** (the Random Forest handles NaN natively). A night with no SpO2 has
`spo2_t90_h = NaN`, not 0 — sensor dropout must never look like "no
desaturation".

## Leakage guard

Feature columns come from an explicit allow-list (`src/model/dataset.py`).
Any column matching AHI, respiratory-event, desaturation-event, TST, sleep
efficiency, stage or stage-duration patterns is rejected by assertion. Scored
sleep time never enters the feature matrix.

## Labels (research bands, not diagnoses)

AHI = scored respiratory events ÷ scored sleep time, per night, computed from
the PAAWS scored-events CSVs (research estimate; codebook definitions not
verified). Participant bands (configurable in `config/phase2.yaml`):

- 3-class (AASM thresholds): Healthy <5, Monitor 5–15, See Doctor ≥15
- 2-class primary: <15 vs ≥15; sensitivity split: <5 vs ≥5
- `worst_reliable_night` (primary) vs `mean_reliable_night` (comparison row)
- reliable night = valid AHI and TST ≥ 2 h (configurable)

**No clinical validity is claimed.** Bands are proposed guidance indicators for
review by the project's medical collaborator, pending Assoc. Prof. Dr. Eg Kah
Peng's sign-off on the four clinical questions.
