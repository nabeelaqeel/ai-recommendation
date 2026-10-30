# PAAWS R2 Sleep Dataset — Phase 1 Audit

**Project:** QoS-Aware Transmission Protocol for Continuous Sleep Monitoring Data
**Module:** Supplementary AI Recommendation (Random Forest)
**Dataset:** PAAWS Sleep Protocol, Release 2, version 1.3
**Audit date:** 2026-10-30
**Status:** Phase 1 complete — no model trained; awaiting review before Phase 2 feature extraction

---

## 1. Executive summary

| Item | Finding |
|---|---|
| Participants on SSD | **141** (matches the R2 Sleep release) |
| Recording nights | **253** (112 participants × 2 nights, 29 × 1 night) |
| File pairing | 253 EDF ↔ 253 `scored_events.csv`; **0 missing pairs, 0 orphans** |
| Total PSG volume | **199.8 GB** EDF; 48.7 MB annotation CSVs |
| Channel layouts | **22** distinct layouts, 74–91 channels per night (median 81) |
| SpO2 available | 245/253 nights (137/141 participants); 2 participants have no SpO2 on any night |
| ECG or EKG available | 252/253 nights; ECG in 183, EKG in 69 (device-dependent label) |
| Motion/position available | 253/253 nights (`Activity`, `PosAngle`, X/Y/Z accelerometer) |
| Total sleep time | median **6.46 h** (IQR 5.40–7.21) across reliable nights |
| Estimated AHI (research) | median **4.51** (IQR 1.71–10.44; p90 19.68; max 64.18) |
| Participant AHI bands (worst reliable night) | <5: **68**, 5–15: **47**, ≥15: **24** |
| Sleep windows for a window-level task | 183,681 scored 30-s sleep epochs; **6.31%** contain a respiratory-event onset |
| Usable nights for AHI | 251 defined; **242 reliable** (TST ≥ 2 h); 2 nights 100% Wake |
| Signal quality (14-night sample) | ECG usable **0–98%** (median 55%); SpO2 valid **24–100%** (median 68%); HR valid **23–100%** (median 81%) |

**Bottom line:** the data support a **participant-level 3-class AHI-band task** (Healthy/Monitor/See-Doctor bands, n = 139 participants, 68/47/24) using features that have plausible Garmin analogues (HR, HRV, SpO2, motion/position). A window-level 2-class event detector is technically possible but strongly imbalanced (6.3% positive) and needs a defensible aggregation story; it is not recommended as the primary task. Both framings remain research proposals requiring medical-collaborator review.

---

## 2. Dataset identity and version verification

- **Source:** `/mnt/e/FYP Data/PAAWS` (external SSD, read-only; nothing was extracted, moved or modified).
- **Documentation present:** `PAAWS_Dataset_README_R2_version_1.3.md` and `PAAWS_Data_Summary_R2_version_1.2.csv`.
- **README content:** confirms R2 Sleep release, `n = 141` participants, two independent zip parts extracting into a shared `PAAWS_Sleep/` tree, Noxturnal A1 PSG recordings at home, manually scored wake/sleep/stages/events.
- **Two extraction roots on disk:** `neu_sn009z58v` (86 participants, 172 nights) and `neu_sn009z615` (55 participants, 81 nights). Participant sets are **disjoint**; together they form the 141-participant release. No duplicate participant-night IDs.
- **Version evidence:**
  - EDF durations are normal sleep-study lengths (median 13.8 h, range 1.2–24.0 h), not the 10×-short files described as a v1.1 bug → consistent with v1.2+.
  - All 253 annotation files lie inside their EDF recording window (0 nights flagged outside) → consistent with the v1.2 annotation-alignment fix.
  - The v1.3 PSG resynchronisation cannot be independently confirmed without the pre-1.3 files. One night (`DS_41-N2`) shows a residual ~0.29 s/epoch annotation-clock drift (≈ 4.4 min over the night); see §6.
  - EDF `patient` fields are anonymised (`X X X X`); no per-file version string exists.
- **Provenance caveat:** the supplied data summary is v1.2. The v1.3 changelog states that only PSG data changed in v1.3 and that the summary was updated in v1.2, so the mixed filenames are internally consistent, but this should be noted in any publication.

---

## 3. File inventory and manifest

Full inventory: [`paaws_file_manifest.csv`](paaws_file_manifest.csv) (253 rows).

| Check | Result |
|---|---|
| Participants on disk vs `PAAWS_SLEEP > 0` in data summary | 141 vs 141 — exact match |
| Participants on disk but not in summary | none |
| Participants in summary but not on disk | none |
| Participants with 2 nights | 112 |
| Participants with 1 night | 29 |
| EDF + CSV pairs complete | 253/253 |
| Duplicate participant-night IDs | 0 |

Night files follow `<participant>-Sleep-Night<k>.edf` and `<participant>-Sleep-Night<k>_scored_events.csv`. Recording dates span **2021-11-30 to 2024-09-27**.

---

## 4. Signal inventory

Full per-night × channel inventory: [`paaws_signal_inventory.csv`](paaws_signal_inventory.csv) (20,494 channel rows; channels, sampling rates, physical units, ranges, durations).

### 4.1 Key channel availability

| Feature | Channel(s) | Nights | Participants | Rate | Unit |
|---|---|---:|---:|---:|---|
| SpO2 | `Saturation`, `SpO2 B-B` | 245 | 137 | 3 Hz | % |
| Heart rate | `Heart Rate` | 252 | 141 | 3 Hz | bpm |
| Oximeter pulse | `Pulse` | 245 | 137 | 3 Hz | bpm |
| ECG | `ECG` (183 nights) / `EKG` (69 nights) | 252 | 141 | 200 Hz | V |
| PPG / plethysmograph | `Pulse Waveform` | 245 | 137 | 75 Hz | raw |
| Activity (motion) | `Activity` | 253 | 141 | 20 Hz | g/s |
| Body position angle | `PosAngle` | 253 | 141 | 20 Hz | deg |
| 3-axis accelerometer | `X Axis`, `Y Axis`, `Z Axis` | 253 | 141 | 20 Hz | g |
| Airflow | `Flow`, `Nasal Pressure` | 253 | 141 | 200 Hz | cmH2O |
| Thoracic effort | `Thorax Fast`, `Chest`, `RIP Sum` | 253 | 141 | 25–200 Hz | V |
| Abdominal effort | `Abdomen Fast`, `Abdomen` | 252–253 | 141 | 25–200 Hz | V |
| Respiratory rate | `Resp Rate` | 253 | 141 | 25 Hz | rpm |
| Snore | `Snore` | 252 | 141 | 200 Hz | cmH2O |
| EEG/EOG | C3/C4/F3/F4/O1/O2, E1/E2 | 253 | 141 | 200 Hz | V |
| Leg EMG | `Left Leg`, `Right Leg` | 253 | 141 | 200 Hz | V |

### 4.2 Heterogeneity and gaps

- **22 ordered channel layouts**, from 74 to 91 channels. This is a real data-quality constraint for any pipeline that assumes a fixed column set.
- **ECG vs EKG:** the same 200 Hz cardiac channel appears under two labels. 100 participants have ECG only, 40 EKG only, 1 participant has both across nights. A loader must accept both.
- **SpO2 gaps:** 8 nights lack both `Saturation` and `SpO2 B-B`: `DS_102-N2`, `DS_11-N2`, `DS_46-N2`, `DS_97-N1/N2`, `DS_98-N1/N2`, `DS_99-N2`. The same 8 nights also lack the oximeter `Pulse` channel, but `Heart Rate` remains available. `DS_97` and `DS_98` have no SpO2 on either night.
- **Duplicate channel labels:** 36 nights contain at least one duplicated label, so loading by label alone is unsafe. `Heart Rate` appears twice on **30 nights**; `K` (8 nights) and `E2-M2` (7 nights) are also duplicated; `DS_195-N2` has two copies of each of `X Axis`, `Y Axis` and `Z Axis`. A loader must select by index within a validated layout and compare duplicate signals before choosing one. This is especially important for the primary HR feature.
- **One night without ECG/EKG:** `DS_18-N1` has neither `ECG` nor `EKG`, but `Heart Rate` and `Pulse` are present.
- **One night without the `Heart Rate` channel:** `DS_123-N1`; it still has `Pulse` and `ECG`, so HR can be derived.
- **Physical ranges are device ranges, not measurement ranges** (e.g., `Saturation` header range ±32800 despite `%` unit). Values must be inspected and clipped/validated before use; do not assume the EDF physical min/max are physiological bounds.
- **No wrist PPG.** `Pulse Waveform` is the oximeter plethysmograph at 75 Hz, not Garmin optical PPG. It is useful for PSG-side HR/SpO2 work but is not a substitute for Garmin PPG.

### 4.3 Missing-value and scaling conventions (observed)

The supplied README does not document how invalid samples are stored. The following encodings were established by inspecting raw values on representative nights and are essential for Phase 2:

| Signal | Missing / invalid encoding | Valid values observed | Consequence |
|---|---|---|---|
| `Saturation`, `SpO2 B-B` | **−0.05** for long missing stretches; other negatives down to ≈ −15 | ≈ 80–111 % (values >100 do occur) | A naive mean or "fraction below 90" treats missing samples as severe desaturation. Mask first, then compute T90/coverage. |
| `Heart Rate` | **0.01** for missing; negatives down to ≈ −8 | ≈ 40–253 bpm | Naive <20/>250 plausibility filters flag 30–75 % of samples as "implausible". |
| `Pulse` | **−0.05** and other negatives | ≈ 65–115 bpm | Same masking requirement as SpO2. |
| `Pulse Waveform` | dropouts are a **constant −0.5** (not zero); the non-zero fraction is 100 % even across flat segments | ±32768 (raw, uncalibrated) | A "non-zero" or "NaN" test misses dropouts entirely; use flat-run / local-variance detection before peak detection. |
| `ECG` / `EKG` | true flat segments, 0–67 % of samples in the 14-night sample; longest flat run up to 517 s | µV–mV scale, night-dependent | QRS detection is not uniformly reliable; HRV requires per-window quality gating, not a global "ECG exists" flag. |
| `Activity`, `PosAngle`, `X/Y/Z Axis` | no missing code observed | g/s, degrees, g | Continuous; `PosAngle` is an angle, not a discrete posture label. |

Coverage is highly variable: on `DS_11-N1`, only **30.4 %** of SpO2 samples and **65.8 %** of `Heart Rate` samples fall in the valid range after masking. Per-night coverage must be computed and reported, and features should be marked missing when coverage falls below an agreed threshold.

---

## 5. Annotation inventory

Full inventory: [`paaws_annotation_inventory.csv`](paaws_annotation_inventory.csv) (33 event types with counts, nights/participants present, duration statistics).

### 5.1 Format

```
Event, Duration, Start Time, End Time, Start Epoch, End Epoch
```

- Absolute timestamps (`Start Time`/`End Time`) plus 30-s epoch indices.
- Stage events (`Wake`, `N1`, `N2`, `N3`, `REM`) are exactly **30.0 s** and one per epoch; no duplicate stage epochs in any night.
- One file (`DS_237-N1`) contains a duplicated `Event` column (`Event.1`), identical values; handled and documented.
- Four nights have no `Analysis Start` row; 249/253 have it.
- There is **no `Analysis End` event**; the scored period is bounded by the first and last stage epochs.

### 5.2 Event vocabulary (total rows across 253 nights)

| Group | Events (count) |
|---|---|
| Sleep stages | N2 98,697; Wake 47,762; REM 41,147; N3 29,204; N1 15,626 |
| Respiratory | Hypopnea 5,630; H. Obstructive 3,720; A. Obstructive 2,692; A. Central 508; H. Central 104; A. Mixed 93; Apnea 52 |
| Oxygen | Desat 13,233 |
| Arousal | Arousal 23,780; Spontaneous Arousal 1 |
| Limb | LM 19,133; PLMS 11,069; PLM 802 |
| Snore | Single Snore 261,546; Snore Train 18,885 |
| Motion/position | Movement 17,624; Supine 2,838; Right 1,726; Left 1,522; Upright 1,436; Prone 882 |
| Cardiac | Tachycardia 807; Bradycardia 43 |
| Technical | Artifact 8,795; Warning 224; Notification 416; Analysis Start 249; Unknown 3 |

### 5.3 Annotation quirks

- `Snore Train` contains **negative durations** (min −5,970.9 s) and implausibly large values (max 5,278.6 s). Do not use raw snore-event durations without cleaning.
- `Artifact` includes a single 53,105 s (14.7 h) block and other very long blocks; artefact annotations are not simple short windows.
- Posture events can span hours (e.g., `Supine` max 29,595 s), so posture labels describe intervals, not epochs.
- `Movement` max 6,237 s; `PLM` max 3,777 s.

These quirks affect any model that uses event durations. The AHI calculation below uses only respiratory event **counts** and scored sleep time, which are not affected by the snore/artefact duration anomalies.

---

## 6. Timing alignment

- **All 253 nights:** first and last stage epochs fall inside the corresponding EDF recording window (0 flagged outside). This is consistent with the v1.2/v1.3 alignment fixes.
- **Recording pre-roll varies:** first scored stage begins a median **28.3 min** after EDF start (IQR 7.1–118.5 min; max 910 min). Signals outside the scored period exist and must not be labelled as sleep or wake.
- **Epoch spacing:** median **30.0009 s**. 252 nights are within ±0.1 s of 30 s.
  - `DS_41-N2`: **29.7127 s/epoch**, accumulating **262.9 s** drift across 916 epochs (~4.4 min/night).
  - `DS_41-N1` (17.5 s) and `DS_221-N2` (28.3 s) show smaller drift.
  - For these nights, window labelling should use **epoch indices**, not wall-clock rescaling. This should be validated per night in Phase 2.
- **Stage gaps:** 29 nights have one or more jumps in stage-epoch index. These are genuine unscored periods (wall-clock gaps), e.g., `DS_176-N1` has 1.5 h and 3 h stretches with no stage labels. Unscored periods must be excluded from window-level labels.
- **No-sleep nights:** `DS_108-N1` and `DS_237-N2` are scored as **100% Wake** (TST = 0). AHI is undefined for these nights.
- **Short nights:** 9 further nights have TST < 2 h (`DS_109-N1`, `DS_217-N2`, `DS_244-N2`, `DS_260-N2`, `DS_273-N1`, `DS_64-N2`, `DS_65-N1`, `DS_65-N2`, `DS_72-N1`). `DS_65` has no reliable night; `DS_109` has only one night and it is short.

---

## 7. Total sleep time and AHI feasibility

### 7.1 Method (research calculation, not clinical AHI)

- **TST** = sum of `Wake`-excluded stage epochs (`N1`+`N2`+`N3`+`REM`), each 30 s. Unscored gaps are excluded.
- **Respiratory events** = unique rows among `Apnea`, `A. Obstructive`, `A. Central`, `A. Mixed`, `Hypopnea`, `H. Obstructive`, `H. Central`. Identical-start duplicate rows are collapsed (1 known case, `DS_202`).
- **AHI** = respiratory events whose onset epoch is a scored sleep epoch ÷ TST hours. Events on Wake or unscored epochs are excluded.
- **Reliability floor** = TST ≥ 2 h. Nights below this are reported but excluded from the primary distribution.
- **Not verified:** the PAAWS codebook is not on the SSD (it is distributed externally). The event definitions above are inferred from the labels and Noxturnal conventions; AASM hypopnoea rule variants (3% vs 4% desaturation, arousal requirement) have not been confirmed. This is the main uncertainty in the AHI values.
- **Sensitivity:** counting all respiratory events regardless of onset stage changes the median AHI by <0.01; only 2.8% of events start outside a scored sleep epoch. The sleep-onset rule therefore has little practical effect on this dataset.

### 7.2 Staging results (253 nights)

| Metric | Value |
|---|---|
| Cumulative scored TST | 1,538.9 h |
| TST median (IQR) | 6.46 h (5.40–7.21) |
| Sleep efficiency median (IQR) | 92.6% (85.3–96.0%) |
| WASO median (IQR) | 0.46 h (0.25–0.99) |
| Stage share of TST | N1 8.5%, N2 53.4%, N3 15.8%, REM 22.3% |

The stage distribution is physiologically plausible, which supports the usability of the staging labels.

### 7.3 AHI results (242 reliable nights)

| Metric | Value |
|---|---|
| AHI median (IQR) | 4.51 (1.71–10.44) |
| AHI mean ± SD | 8.08 ± 10.17 |
| AHI p90 / p95 | 19.68 / 28.59 |
| AHI max | 64.18 |
| Nights with zero sleep respiratory events | 8 |

**Night-level distribution (242 reliable nights):**

| Band | 4-band | 3-class collapse | % (3-class) |
|---|---:|---:|---:|
| Normal / Healthy (<5) | 129 | 129 | 53.3% |
| Mild / Monitor (5–15) | 76 | 76 | 31.4% |
| Moderate (15–30) | 26 | — | — |
| Severe (≥30) | 11 | — | — |
| See Doctor (≥15) | — | 37 | 15.3% |

**Participant-level distribution (139 participants with ≥1 reliable night, worst reliable night):**

| Band | 4-band | 3-class | % (3-class) |
|---|---:|---:|---:|
| Healthy (<5) | 68 | 68 | 48.9% |
| Monitor (5–15) | 47 | 47 | 33.8% |
| Moderate (15–30) | 17 | — | — |
| Severe (≥30) | 7 | — | — |
| See Doctor (≥15) | — | 24 | 17.3% |

Participant labels using the mean of reliable nights give a very similar split (71/44/24). The worst-night definition is more consistent with screening semantics; the mean definition is more stable. This choice should be agreed with the medical collaborator.

### 7.4 Two-night consistency (103 participants with two reliable nights)

| Metric | Value |
|---|---|
| Spearman correlation of AHI between nights | **0.886** |
| Same 4-band category on both nights | 81.6% |
| Same 3-class category on both nights | 83.5% |
| Median absolute AHI difference | 1.60 |

AHI is stable enough across nights to support a **participant-level** label; treating the two nights as independent would still be wrong for evaluation because they are not independent samples.

Of the 103 participants with two reliable nights, 13 were ≥15 on both nights, 7 on one night only and 83 on neither. The worst-night definition therefore places 7 participants in the See-Doctor band on the basis of a single night; this is the main argument for reviewing the worst-night versus mean-night rule with the medical collaborator.

### 7.5 Window-level class balance

| Metric | Value |
|---|---|
| Scored sleep epochs (30 s) across reliable nights | 183,681 |
| Epochs with ≥1 respiratory-event onset | 11,593 (**6.31%**) |
| Positive rate per night, median (IQR) | 3.8% (1.4–8.4%) |
| Positive rate by AHI band (median) | Normal 1.5%, Mild 6.8%, Moderate 16.3%, Severe 31.8% |

The strong gradient confirms that window-level signal contains information about event risk, but the 1:15 imbalance, within-night correlation and noisy event onsets make a window-level classifier a harder primary task than a participant-level one.

---

## 8. Signal quality and feature feasibility (14-night sample)

Waveform content was read for **14 of 253 nights (5.5%)**, purposively chosen to cover: both extraction roots, ECG and EKG layouts, missing-SpO2 nights, the two 100%-Wake nights, the shortest recording, the clock-drift night, and nights with long unscored gaps. Availability statistics in §4 come from all 253 headers; the quality numbers below are **indicative, not a prevalence estimate**.

### 8.1 ECG and HRV

| Night | Channel | Flat samples % | Longest flat (s) | Usable windows % | RMSSD (ms) | Verdict |
|---|---|---:|---:|---:|---:|---|
| DS_11-N1 | ECG | 23.8 | 20.9 | 31.7 | 361.0 | Unusable (RMSSD implausible) |
| DS_108-N1 | EKG | 22.6 | 283.1 | 55.0 | 22.8 | Partial |
| DS_41-N2 | ECG | 47.2 | 405.5 | 61.7 | 185.0 | Partial (RMSSD implausible) |
| DS_97-N1 | EKG | 66.8 | 11.0 | **0.0** | — | Unusable |
| DS_109-N1 | EKG | 25.4 | 11.5 | 15.0 | 694.9 | Unusable |
| DS_176-N1 | ECG | 2.0 | 14.7 | 33.3 | 97.3 | Partial |
| DS_237-N2 | ECG | 17.0 | 11.2 | 1.7 | 310.9 | Unusable |
| DS_102-N2 | EKG | 32.9 | 4.4 | 61.7 | 448.3 | Partial (RMSSD implausible) |
| DS_103-N1 | EKG | 17.3 | 12.3 | 30.0 | 46.3 | Partial |
| DS_118-N1 | EKG | 42.4 | 13.7 | **98.3** | 36.8 | Good |
| DS_148-N1 | ECG | 44.5 | 10.8 | 55.0 | 196.6 | Partial |
| DS_211-N1 | ECG | 31.0 | 16.3 | **98.3** | 40.9 | Good |
| DS_30-N1 | ECG | 38.9 | 517.2 | **98.3** | 83.2 | Good |
| DS_79-N1 | EKG | 25.0 | 11.7 | 75.0 | 41.1 | Good |

- **Usable-window fraction ranges 0–98% (median 55%)**; only 3/14 nights reach ≥90%, and 6/14 are ≤35%. `DS_97` has no usable QRS at all.
- On gated windows the detected rate agrees with the recorded `Heart Rate` (agreement ratio 0.98–1.04), so the QRS detector works **when the signal is clean**.
- The agreement gate alone is **not sufficient for RMSSD**: five nights pass it yet produce 185–695 ms, which is not physiological for overnight RMSSD. A usable RMSSD needs ≥90% usable windows *and* a physiologic bound (e.g. 5–200 ms), otherwise it must be recorded as missing.
- **PPG is not a systematic fallback:** where `Pulse Waveform` is clean it tracks HR well (DS_11 hour 1: 44 peaks/30 s → 88.2 bpm vs recorded 88.1 bpm), but it is flat for long stretches (hours 3 and 6 are a constant −0.5), so it cannot fill ECG gaps reliably. It is also oximeter pleth, not wrist PPG, so it does not close the Garmin domain gap.

### 8.2 SpO2, heart rate and motion

| Night | SpO2 valid % | SpO2 median | <90 % of valid | T90 valid (h) | T90 in sleep (h) | HR valid % | HR median |
|---|---:|---:|---:|---:|---:|---:|---:|
| DS_11-N1 | 30.4 | 93.85 | 6.17 | 0.21 | 0.21 | 65.7 | 80.0 |
| DS_108-N1 | 68.8 | 90.95 | 39.26 | 4.37 | — | 70.9 | 54.8 |
| DS_41-N2 | 65.7 | 94.95 | 0.56 | 0.05 | 0.04 | 91.3 | 60.2 |
| DS_97-N1 | — | — | — | — | — | 79.3 | 144.8* |
| DS_109-N1 | 99.0 | 94.95 | 0.40 | 0.005 | 0.000 | 88.9 | 63.9 |
| DS_176-N1 | 24.0 | 91.85 | 33.14 | 1.47 | 1.45 | 79.6 | 92.0 |
| DS_237-N2 | 51.2 | 91.85 | 22.99 | 2.14 | — | 23.2 | 81.6 |
| DS_102-N2 | — | — | — | — | — | 81.5 | 65.6 |
| DS_103-N1 | 87.6 | 94.95 | 0.06 | 0.005 | 0.001 | 64.8 | 73.1 |
| DS_118-N1 | 98.0 | 95.85 | 0.02 | 0.002 | 0.000 | 99.9 | 55.8 |
| DS_148-N1 | 59.5 | 91.85 | 26.39 | 2.33 | 2.06 | 93.9 | 78.3 |
| DS_211-N1 | 100.0 | 92.95 | 6.97 | 0.51 | 0.43 | 99.7 | 61.6 |
| DS_30-N1 | 98.1 | 94.95 | 0.02 | 0.002 | 0.001 | 99.4 | 64.1 |
| DS_79-N1 | 66.6 | 94.35 | 1.16 | 0.11 | 0.08 | 76.7 | 55.4 |

\* `DS_97-N1` is flagged unusable: although 79% of samples fall inside 20–250 bpm, the distribution spans 22–247 bpm with a median of 144, which is an artefact pattern, not a plausible overnight HR.

- **SpO2 coverage is the main constraint:** 24–100% valid (median 68%). T90 computed on masked samples ranges from 0.002 h to 4.37 h, and the "fraction below 90" ranges from 0.02% to 39.3%. Missing SpO2 is therefore a much larger source of error than measurement noise: T90 must be normalised by **valid** time and accompanied by its coverage; a naive T90 divided by recording time understates desaturation, while treating −0.05 as a value grossly overstates it.
- **HR coverage** is 23–100% (median 81%) after masking 0.01; it is the most consistently available dashboard signal, but per-night coverage still needs to be a reported feature flag.
- **Motion/position** is clean on all 14 nights: `Activity` standard deviation 0.08–0.33 g/s and `PosAngle` IQR 10–180° (position genuinely changes during the night), with no missing-value codes.
- **Airflow reference signals** also contain long flat segments (up to 405 s on DS_41-N2 and 283 s on DS_108-N1). This affects confidence in the scored respiratory events, not the deployable feature set.

### 8.3 Feature verdict for the four dashboard indicators

| Feature | Evidence from the sample | Verdict for Phase 2 |
|---|---|---|
| **HR (bpm)** | `Heart Rate` 3 Hz; missing 0.01; valid 23–100% (median 81%); DS_97 pathological | **Feasible.** Mask, add coverage %, exclude nights like DS_97 via distribution checks |
| **SpO2 (incl. T90)** | 24–100% valid (median 68%); 2/14 nights have no channel; values >100 occur | **Conditionally feasible.** Mask −0.05, normalize T90 by valid time, publish coverage per night |
| **HRV (RMSSD)** | ECG usable 0–98% (median 55%); plausible RMSSD only on high-coverage nights (3/14 ≥90%) | **Not uniformly feasible.** Window-level QC + ≥90% coverage + physiologic bounds; record as missing otherwise; PPG cannot fill the gaps reliably |
| **Motion / position** | 20 Hz `Activity`, `PosAngle`, X/Y/Z; 14/14 usable, no missing codes | **Feasible.** `PosAngle` is a continuous angle, so derive posture bands rather than assuming discrete labels |

This means a model using all four indicators will have **structured missingness** on a minority of nights (SpO2 absent, HRV unusable). Phase 2 must represent that explicitly (coverage features + missing-value handling) rather than imputing silently.

## 9. Candidate features with a Garmin analogue

The live system will use a Garmin smartwatch, not clinical PSG. The table below separates features that have a plausible wrist-worn analogue from those that do not. **None of these mappings has been validated against Garmin data**; they are design hypotheses for Phase 2.

| Candidate feature | PSG source (verified) | Garmin analogue | Caveat |
|---|---|---|---|
| Heart rate (mean, min, variability) | `Heart Rate` (3 Hz, bpm, missing = 0.01); `ECG`/`EKG` (200 Hz) | Optical HR | PSG HR is ECG/oximeter-derived; Garmin is PPG. Expect bias during motion and low perfusion. Valid coverage 23–100% in the sample. |
| HRV (RMSSD, SDNN) | `ECG`/`EKG` at 200 Hz | Garmin HRV (PPG-derived, mainly overnight) | ECG usable windows range 0–98% (median 55%); RMSSD is only credible at high coverage, so it must be missing otherwise. PPG HRV is less reliable than ECG at high RMSSD; artefact rejection differs. |
| SpO2 (mean, min, T90) | `Saturation` / `SpO2 B-B` (3 Hz, %, missing = −0.05) | Garmin Pulse Ox (model-dependent, often intermittent) | Valid coverage 24–100% (median 68%); T90 must be normalised by valid time and reported with its coverage. Not all Garmin models expose continuous SpO2; sampling cadence is much lower than 3 Hz. |
| Motion / sleep-wake | `Activity` (20 Hz), `X/Y/Z` accel | Wrist accelerometer | PSG activity is a body-motion channel, not wrist-specific; transfer requires recalibration. |
| Body position | `PosAngle` (20 Hz, continuous angle in degrees) | Not consistently available on Garmin | Not a discrete posture label; derive posture bands from the angle. Position may be inferable from accelerometer orientation but is not a standard Garmin field. |
| Respiratory rate | `Resp Rate` (25 Hz, rpm) | Garmin respiration rate (overnight) | Algorithmic derivation differs; use as a secondary feature. |
| Snoring | `Snore` (200 Hz) + snore events | Garmin does not measure snoring acoustically | Not transferable. |
| Airflow / effort / desaturation events / EEG staging | Flow, RIP, SpO2 events, EEG | Not available | These are label-construction signals, not deployable inputs. |

**Leakage rule:** AHI, respiratory-event counts, desaturation counts and any annotation-derived event variable must never be model inputs. TST and stage fractions are legitimate only if they are derived from the wearable in deployment; using PSG-scored TST as an input while predicting a PSG-derived AHI band would leak information and must be avoided.

---

## 10. Data-quality limitations

1. **Heterogeneous channel sets** (22 layouts; ECG vs EKG; 74–91 channels; duplicate labels on 36 nights). Any loader must map by canonical role and validated index, not by position or label alone.
2. **Missing SpO2** on 8 nights; two participants (DS_97, DS_98) have no SpO2 at all. `DS_18-N1` has no ECG/EKG (HR/Pulse present) and `DS_123-N1` has no `Heart Rate` channel (Pulse/ECG present).
3. **Two 100%-Wake nights** (DS_108-N1, DS_237-N2) and 9 nights with TST < 2 h. Two participants (DS_65, DS_109) have no reliable night.
4. **29 nights with unscored stage gaps**; window-level labels must exclude them.
5. **Annotation clock drift** on DS_41-N2 (4.4 min/night), with smaller drift on DS_41-N1 and DS_221-N2. Epoch-index alignment should be preferred over wall-clock rescaling.
6. **Annotation duration anomalies** for snore, movement, artefact and posture events; do not use those durations without cleaning.
7. **Undocumented missing-value codes** (`−0.05` for SpO2/Pulse, `0.01` for Heart Rate) and uncalibrated raw PPG: mask before any statistic (§4.3).
8. **No local codebook**: AHI event definitions and hypopnoea rules are inferred, not verified. The AHI values are research estimates.
9. **No Garmin data**: the domain gap between clinical PSG and wrist PPG is unquantified. Any reported model performance is a PSG-domain result, not evidence of Garmin performance.
10. **Recordings are much longer than the scored sleep period** (median 13.8 h vs 6.5 h TST). Pipeline must window to the scored/sleep period and not treat pre-roll as sleep.
11. **Two extraction roots**: tooling must traverse both; they are disjoint and must not be assumed to be duplicates.

---

## 11. Recommended task formulation

### Primary recommendation: participant-level 3-class AHI band

- **Label:** worst reliable night's AHI band: `<5` (Healthy band), `5–15` (Monitor band), `≥15` (See-Doctor band).
- **Cohort:** 139 participants; class sizes 68 / 47 / 24 — adequate for a 3-class baseline with participant-grouped cross-validation.
- **Inputs (Phase 2):** windowed/aggregated HR, HRV, SpO2, motion and position features restricted to the overnight period, using only channels with a plausible Garmin analogue. Each signal carries an explicit coverage indicator; HRV is computed only from QC-passed ECG windows; SpO2/T90 are normalised by valid (masked) time; structured missingness is preserved rather than imputed silently.
- **Evaluation:** participant-grouped stratified CV (all nights of a participant in one fold); report macro-F1, balanced accuracy, per-class precision/recall/F1 and the confusion matrix. Accuracy alone is not acceptable.
- **Why this task:** it matches the dashboard indicators directly, has a defensible cohort size, avoids the 1:15 window imbalance, and is robust to the night-to-night variability documented above.

### Simpler fallback: 2-class elevated-risk task

If the 3-class model is unstable (e.g., the 24-person See-Doctor class is too small for reliable estimates), collapse to `AHI ≥ 15` vs `< 15` (24 vs 115). This is the smallest defensible task and should be reported alongside the 3-class result if there is any doubt.

### Not recommended as the primary task: window-level 2-class event detection

A 30-s-window event/no-event classifier is feasible (183,681 windows; 6.31% positive) and the positive rate rises monotonically with AHI band (1.5% → 31.8%), so the signal is there. However:
- the class imbalance is severe;
- adjacent windows are highly correlated, inflating effective sample size;
- event onsets are annotation-dependent and noisy;
- mapping window scores to Healthy/Monitor/See-Doctor requires an aggregation and threshold rule that has not been clinically reviewed.

It could be a **secondary analysis** (e.g., to add temporal resolution or to explain night-level scores), but it should not replace the participant-level task in the first model iteration.

### Medical review required

The band boundaries (`<5`, `5–15`, `≥15`) follow common sleep-medicine conventions but the mapping to **Healthy / Monitor / See Doctor** is a research proposal. Assoc. Prof. Dr. Eg Kah Peng should confirm:
1. the band boundaries and whether `≥15` or `≥30` should trigger "See Doctor";
2. whether participant labels should use the worst night or the mean;
3. the acceptable false-negative rate for the See-Doctor band;
4. that the output is framed as a screening indicator, not a diagnosis.

---

## 12. Reproducibility

All scripts are in `src/audit/` and read the SSD read-only:

| Script | Purpose |
|---|---|
| `common.py` | Paths, night discovery, pure-Python EDF header parser |
| `01_file_manifest.py` | File pairing, data-summary cross-check |
| `02_header_scan.py` | All 253 EDF headers → signal inventory |
| `03_annotation_scan.py` | All 253 CSVs → event inventory, staging, alignment |
| `04_sample_quality.py` | 14-night signal-quality sample (NaN/flatline, T90, HR, RMSSD, motion) |
| `05_ahi_preliminary.py` | TST/AHI/class balance and window-level counts |
| `06_verify_deliverables.py` | Cross-checks the four deliverables and the report's headline numbers |

Supporting outputs (not counted as report deliverables):

- `outputs/paaws_night_headers.csv` — per-night EDF start/duration
- `outputs/paaws_night_stage_metrics.csv` — per-night staging and alignment
- `outputs/paaws_event_counts_by_night.csv` — per-night event counts
- `outputs/paaws_night_metrics.csv` — per-night TST/AHI/window metrics
- `outputs/paaws_participant_labels.csv` — participant-level AHI bands
- `outputs/ahi_class_distribution.csv` — all distribution tables
- `outputs/paaws_sample_signal_quality.csv` — sample quality metrics

**Runtime:** header scan ≈ 1 min; annotation scan ≈ 2 s; AHI ≈ 10 s; sample-quality reads dominate (~15–20 min for 14 nights because pyedflib selectively decompresses large EDFs). No large intermediate files were created; no raw file was modified.

---

## 13. Phase 1 conclusion

The dataset is complete, internally consistent and large enough for a supplementary participant-level 3-class sleep-risk band model. The dashboard features split into three tiers: **HR and motion/position are consistently usable**, **SpO2/T90 is usable but coverage-limited**, and **ECG-based HRV is only available on a subset of nights and requires strict window-level QC**. The main risks are the PSG→Garmin domain gap, the inferred AHI definitions, structured feature missingness, and the small See-Doctor class (n = 24). Feature extraction and model training should not begin until the task formulation and band mapping are reviewed.
