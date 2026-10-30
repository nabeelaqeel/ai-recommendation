# AI Recommendation Module — QoS-Aware Sleep Monitoring (FYP)

Supplementary machine-learning module for the FYP **“QoS-Aware Transmission Protocol for Continuous Sleep Monitoring Data.”**

The primary research contribution is the **RB-QoSPCTF** rule-based framework, which classifies physiological readings into Critical / Important / Normal / Background tiers and maps them to MQTT QoS levels. This repository is a **separate, supplementary** module that explores a lightweight, explainable Random Forest recommendation layer (Healthy / Monitor / See Doctor indicators).

> **Status: Phase 1 (dataset audit) complete. No model has been trained yet.**
> Feature extraction and model training are intentionally deferred until the task formulation and the AHI-band mapping are reviewed by the project's medical collaborator.

## Dataset

PAAWS Sleep Protocol, Release 2, version 1.3 — 141 participants, 253 nights, 199.8 GB of EDF polysomnography plus scored-event CSVs.

The raw dataset is **not** included in this repository. Scripts read it read-only from an external drive (default `/mnt/e/FYP Data/PAAWS`; change `PAAWS_ROOT` in `src/audit/common.py` if your mount differs).

## Phase 1 findings

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

The full audit, including data-quality caveats and the recommended task formulation, is in [`reports/paaws_dataset_audit.md`](reports/paaws_dataset_audit.md).

**Important:** AHI here is a research calculation from scored respiratory events divided by scored sleep time. The PAAWS codebook was not available, so event definitions are inferred. The Healthy / Monitor / See Doctor mapping is a **proposal**, not an approved clinical triage protocol.

## Repository layout

```
src/audit/                  Phase 1 audit pipeline
  01_file_manifest.py       pair EDF ↔ CSV, cross-check the data summary
  02_header_scan.py         all 253 EDF headers → signal inventory
  03_annotation_scan.py     all 253 scored-event CSVs → event inventory
  04_sample_quality.py      14-night signal-quality / feature-feasibility sample
  05_ahi_preliminary.py     TST, research AHI, class balance, window counts
  06_verify_deliverables.py cross-check the deliverables and headline numbers
reports/                    audit report + inventories (metadata only)
outputs/                    generated participant-level tables (git-ignored)
```

## Reproducing the audit

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd src/audit
python 01_file_manifest.py
python 02_header_scan.py
python 03_annotation_scan.py
python 04_sample_quality.py     # ~15–20 min: selectively decompresses large EDFs
python 05_ahi_preliminary.py
python 06_verify_deliverables.py
```

All scripts are read-only with respect to the raw dataset and write only inside `reports/` and `outputs/`.

## Data availability and privacy

`outputs/` contains **derived participant-level** tables (per-night TST/AHI/window counts and per-participant AHI bands). These are generated locally and intentionally **excluded** from this public repository.

The committed inventories (`reports/paaws_file_manifest.csv`, `reports/paaws_signal_inventory.csv`, `reports/paaws_annotation_inventory.csv`) contain pseudonymous participant codes and file/channel metadata only — no signal values and no clinical measurements.

## Disclaimer

This is a research prototype. It does not diagnose sleep apnoea or any other condition, has not been validated on Garmin (wrist-worn) data, and must not be used for clinical decision-making or patient safety.
