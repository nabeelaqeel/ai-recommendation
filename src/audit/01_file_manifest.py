"""Build the PAAWS R2 sleep file manifest.

Inventories every participant-night on the external SSD, pairs EDF with
scored-events CSV, cross-checks the supplied data summary and writes
reports/paaws_file_manifest.csv.
"""

from __future__ import annotations

import pandas as pd

from common import DATA_SUMMARY, REPORTS, configure_logging, discover_nights

log = configure_logging("01_file_manifest")


def main() -> None:
    nights = discover_nights()
    log.info("Discovered %d participant-night records", len(nights))

    summary = pd.read_csv(DATA_SUMMARY)
    expected = summary.set_index("DATASET")["PAAWS_SLEEP"].astype(int).to_dict()
    log.info("Data summary rows: %d (PAAWS_SLEEP night counts present)", len(summary))

    rows = []
    for n in nights:
        rows.append(
            {
                "root": n.root_name,
                "participant_id": n.participant,
                "night": n.night,
                "night_id": n.night_id,
                "edf_file": str(n.edf.relative_to(n.edf.parents[2])) if n.edf.exists() else "",
                "csv_file": str(n.csv.relative_to(n.csv.parents[2])) if n.csv.exists() else "",
                "edf_present": n.edf.exists(),
                "csv_present": n.csv.exists(),
                "edf_size_mb": round(n.edf.stat().st_size / 1e6, 3) if n.edf.exists() else None,
                "csv_size_mb": round(n.csv.stat().st_size / 1e6, 3) if n.csv.exists() else None,
                "summary_sleep_nights": expected.get(n.participant),
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["participant_id", "night"]).reset_index(drop=True)

    # Invariant checks
    dup = manifest.duplicated(["participant_id", "night"], keep=False)
    if dup.any():
        log.warning("Duplicate participant-night records: %d", int(dup.sum()))
    orphans = manifest[~(manifest.edf_present & manifest.csv_present)]
    if len(orphans):
        log.warning("Unpaired files found:\n%s", orphans[["night_id", "edf_present", "csv_present"]].to_string(index=False))

    per_participant = manifest.groupby("participant_id").agg(
        nights_on_disk=("night", "count"),
        summary_sleep_nights=("summary_sleep_nights", "first"),
        edf_gb=("edf_size_mb", lambda s: round(s.sum() / 1e3, 3)),
    )
    mismatch = per_participant[per_participant.nights_on_disk != per_participant.summary_sleep_nights]
    if len(mismatch):
        log.warning("Participants whose on-disk nights differ from data summary:\n%s", mismatch.to_string())

    on_disk = set(manifest.participant_id)
    in_summary = {p for p, v in expected.items() if v > 0}
    log.info(
        "Participants on disk: %d | in data summary with sleep nights>0: %d | "
        "on disk but not in summary: %s | in summary but not on disk: %s",
        len(on_disk),
        len(in_summary),
        sorted(on_disk - in_summary),
        sorted(in_summary - on_disk),
    )
    log.info(
        "Nights per participant: %s",
        manifest.groupby("participant_id").size().value_counts().sort_index().to_dict(),
    )
    log.info("Total EDF size: %.1f GB", manifest.edf_size_mb.sum() / 1e3)
    log.info("Total CSV size: %.1f MB", manifest.csv_size_mb.sum())

    out = REPORTS / "paaws_file_manifest.csv"
    manifest.to_csv(out, index=False)
    log.info("Wrote %s (%d rows)", out, len(manifest))


if __name__ == "__main__":
    main()
