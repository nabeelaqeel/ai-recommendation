# AI Recommendation Module — QoS-Aware Sleep Monitoring (FYP)

Supplementary machine-learning module for the FYP **“QoS-Aware Transmission Protocol for Continuous Sleep Monitoring Data.”**

The primary research contribution is the **RB-QoSPCTF** rule-based framework, which classifies physiological readings into Critical / Important / Normal / Background tiers and maps them to MQTT QoS levels. This repository is a **separate, supplementary** module that explores a lightweight, explainable Random Forest recommendation layer (Healthy / Monitor / See Doctor indicators).

> **Status: Phase 1 (dataset audit) and Phase 2 (feature extraction + baseline model) complete.**
> All clinical choices (band boundaries, label rule, recall target) are config parameters pending
> review by the project's medical collaborator — see
> [`reports/clinical_review_brief_dr_eg.md`](reports/clinical_review_brief_dr_eg.md).
> Nothing here diagnoses sleep apnoea or any condition.

## Dataset

PAAWS Sleep Protocol, Release 2, version 1.3 — 141 participants, 253 nights, 199.8 GB of EDF polysomnography plus scored-event CSVs.

The raw dataset is **not** included in this repository. Scripts read it read-only from an external drive (default `/mnt/e/FYP Data/PAAWS`; change `PAAWS_ROOT` in `src/audit/common.py` if your mount differs).

## Phase 1 findings (audit)

| Item | Result |
|---|---|
| Participants / nights | 141 / 253 (112 × 2 nights, 29 × 1 night) |
| Channel layouts | 22 distinct layouts, 74–91 channels per night |
| Total sleep time | median 6.46 h (IQR 5.40–7.21) |
| Research AHI estimate | median 4.51 (IQR 1.71–10.44), max 64.18 |
| Participant AHI bands (worst reliable night) | 68 (<5) / 47 (5–15) / 24 (≥15) |
| Window-level event rate | 6.31 % of 183,681 30-s sleep epochs |
| Night-to-night AHI agreement | Spearman 0.886; same 3-class band 83.5 % |
| Signal quality (14-night sample) | ECG usable 0–98 % (median 55 %); SpO2 valid 24–100 % (median 68 %); HR valid 23–100 % (median 81 %) |

Full audit: [`reports/paaws_dataset_audit.md`](reports/paaws_dataset_audit.md).

## Phase 2 findings (features + baseline model)

Features are computed **only over scored sleep epochs**, from the four Garmin-plausible signals
(HR, ECG-HRV → RMSSD, SpO2/T90, motion) plus torso angle as a separate ablation arm.
Missingness is preserved (NaN + explicit flags, never imputed); artefact nights
(e.g. a night whose sleep HR median is 146 bpm) have their values withdrawn.

| Model (participant-grouped 5×5 CV) | macro-F1 | pooled [95 % CI] |
|---|---|---|
| Random Forest, Garmin-strict (HR, RMSSD, SpO2/T90, Activity) | 0.514 ± 0.083 | 0.520 [0.436–0.607] |
| + torso angle | 0.510 ± 0.077 | 0.521 [0.438–0.598] |
| + sensor-coverage flags | 0.527 ± 0.077 | 0.566 [0.482–0.646] |
| all features | 0.547 ± 0.077 | 0.582 [0.501–0.658] |
| logistic regression comparator | 0.485 ± 0.092 | 0.492 [0.408–0.569] |
| 2-class (≥15 vs <15) | 0.736 ± 0.098 | 0.739 [0.627–0.834] |

- **Cost-sensitive operating point** (threshold chosen inside training folds, applied to
  untouched folds): See-Doctor recall **0.876** at specificity 0.615 / precision 0.343
  (deployable core model, threshold 0.13 on P(see_doctor)) — recall target ≥85 % met.
- **SpO2 features dominate** permutation importance (`spo2_median` ≈ 0.13); coverage flags
  contribute only marginally, so sensor dropout is not acting as a strong severity shortcut.
- **Live inference:** `predict_from_rolling_buffer()` classifies a 10-minute rolling buffer in
  **p50 16.2 ms / p95 17.4 ms** (budget 50 ms), using the same feature code as training.
- Full report: [`reports/phase2_model_report.md`](reports/phase2_model_report.md);
  feature definitions: [`reports/phase2_feature_dictionary.md`](reports/phase2_feature_dictionary.md).

**Important:** AHI here is a research calculation from scored respiratory events divided by
scored sleep time. The PAAWS codebook was not available, so event definitions are inferred.
The Healthy / Monitor / See Doctor mapping is a **proposal**, not an approved clinical triage
protocol. Model probability outputs are uncalibrated scores, not clinical probabilities.

## Repository layout

```
config/phase2.yaml              every clinical/feature parameter (bands, label rule, recall target)
src/audit/                      Phase 1 audit pipeline
  01_file_manifest.py … 06_verify_deliverables.py
src/features/                   Phase 2 feature extraction
  config.py channels.py epochs.py features.py
  extract_night_features.py     per-night Parquet, resumable, --pilot / --all / --reaggregate
  pilot_qc.py                   QC gates (row counts, T90 recompute, artefact checks, leakage)
src/model/                      Phase 2 model
  dataset.py                    participant table, config-driven labels, leakage assertions
  train_rf.py                   grouped CV, ablations, threshold analysis, reports
  classifier.py                 saved-model wrapper (bands + threshold flag)
  predict_from_rolling_buffer.py  live bridge + latency benchmark
  make_clinical_brief.py        collaborator sign-off brief
reports/                        audit report + inventories + model reports (metadata only)
outputs/                        generated participant-level tables (git-ignored)
data/processed/                 feature Parquets (git-ignored)
models/                         trained artifacts (git-ignored)
```

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Phase 1 (audit)
cd src/audit && python 01_file_manifest.py && python 02_header_scan.py \
  && python 03_annotation_scan.py && python 04_sample_quality.py \
  && python 05_ahi_preliminary.py && python 06_verify_deliverables.py

# Phase 2 (from the project root)
python -m src.features.extract_night_features --pilot   # 30-night QC pilot
python -m src.features.pilot_qc
python -m src.features.extract_night_features --all     # 253 nights (~86 min)
python -m src.model.dataset
python -m src.model.train_rf                            # ~16 min
python -m src.model.predict_from_rolling_buffer         # latency benchmark
python -m src.model.make_clinical_brief
```

All scripts are read-only with respect to the raw dataset and write only inside the project
directory.

## Data availability and privacy

`outputs/`, `data/processed/` and `models/` contain **derived participant-level** tables and
artifacts. They are generated locally and intentionally **excluded** from this repository.

The committed inventories (`reports/paaws_file_manifest.csv`, `reports/paaws_signal_inventory.csv`,
`reports/paaws_annotation_inventory.csv`) contain pseudonymous participant codes and
file/channel metadata only — no signal values and no clinical measurements.

## Disclaimer

This is a research prototype. It does not diagnose sleep apnoea or any other condition, has
not been validated on Garmin (wrist-worn) data, and must not be used for clinical
decision-making or patient safety.
