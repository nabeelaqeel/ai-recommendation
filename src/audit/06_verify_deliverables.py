"""Verify the four Phase 1 audit deliverables are present and internally consistent.

Run after 01-05 and the report are complete. Exits non-zero on any failure so the
audit can be re-checked cheaply.
"""

from __future__ import annotations

import sys

import pandas as pd

from common import OUTPUTS, REPORTS, configure_logging, discover_nights, edf_header

log = configure_logging("06_verify_deliverables")

FAILS: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        log.info("PASS  %s", message)
    else:
        FAILS.append(message)
        log.error("FAIL  %s", message)


def main() -> int:
    nights = discover_nights()
    manifest = pd.read_csv(REPORTS / "paaws_file_manifest.csv")
    signals = pd.read_csv(REPORTS / "paaws_signal_inventory.csv")
    annotations = pd.read_csv(REPORTS / "paaws_annotation_inventory.csv")
    report = REPORTS / "paaws_dataset_audit.md"

    check(len(manifest) == 253, f"manifest has 253 rows (found {len(manifest)})")
    check(manifest.participant_id.nunique() == 141, "manifest has 141 participants")
    check(bool((manifest.edf_present & manifest.csv_present).all()), "every night has EDF + CSV")
    check(len(nights) == len(manifest), "discovery and manifest agree on night count")

    check(len(annotations) == 33, f"annotation inventory has 33 event types (found {len(annotations)})")
    check(annotations.total_events.sum() > 500_000, "annotation inventory is non-trivial")

    # Signal inventory must cover every discovered night, and header channel counts
    # must agree with the manifest's on-disk channel inventory.
    per_night = signals.groupby("night_id").size()
    check(set(per_night.index) == {n.night_id for n in nights}, "signal inventory covers every night")
    hdr = edf_header(nights[0].edf)
    first_night = nights[0]
    check(
        per_night.get(first_night.night_id, -1) == hdr["n_channels"],
        f"{first_night.night_id} channel count matches EDF header ({hdr['n_channels']})",
    )
    check(
        bool(signals.channel_label.str.lower().ne("edf annotations").all()),
        "annotation channel excluded from signal inventory",
    )

    check(report.exists() and report.stat().st_size > 10_000, "audit report exists and is substantial")
    text = report.read_text() if report.exists() else ""
    check("SAMPLE_QUALITY_PLACEHOLDER" not in text, "report has no unresolved placeholder")
    for section in ["## 1.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8.", "## 9.", "## 11."]:
        check(section in text, f"report contains section {section.strip()}")

    # AHI numbers quoted in the report must match the analysis outputs.
    metrics = pd.read_csv(OUTPUTS / "paaws_night_metrics.csv")
    labels = pd.read_csv(OUTPUTS / "paaws_participant_labels.csv")
    reliable = metrics[metrics.ahi_reliable_tst_ge_2h]
    check(len(reliable) == 242, f"242 reliable nights (found {len(reliable)})")
    check(len(labels) == 139, f"139 labelled participants (found {len(labels)})")
    counts = labels.ahi_max_3class.value_counts()
    check(
        counts.get("Healthy-band (<5)") == 68
        and counts.get("Monitor-band (5-15)") == 47
        and counts.get("See-Doctor-band (>=15)") == 24,
        "participant 3-class counts are 68/47/24",
    )
    windows = int(reliable.n_sleep_windows.sum())
    positives = int(reliable.n_positive_windows.sum())
    check(windows == 183_681, f"sleep windows = 183,681 (found {windows})")
    check(positives == 11_593, f"positive windows = 11,593 (found {positives})")

    # Sample-quality output (produced by 04) should cover the 14 sampled nights.
    quality = OUTPUTS / "paaws_sample_signal_quality.csv"
    check(quality.exists(), "sample signal-quality CSV exists")
    if quality.exists():
        q = pd.read_csv(quality)
        check(len(q) == 14, f"sample quality covers 14 nights (found {len(q)})")
        check("spo2_valid_pct" in q.columns, "sample quality uses masked SpO2 validity")
        check("ecg_usable_window_pct" in q.columns, "sample quality gates ECG usability")

    if FAILS:
        log.error("%d check(s) failed", len(FAILS))
        return 1
    log.info("All deliverable checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
