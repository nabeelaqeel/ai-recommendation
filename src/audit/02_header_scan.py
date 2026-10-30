"""Scan every PAAWS sleep EDF header (all 253 nights).

Reads only the fixed-size EDF header block; no signal data is loaded.
Writes:
  reports/paaws_signal_inventory.csv  (one row per night x channel)
  outputs/paaws_night_headers.csv     (one row per night, for alignment joins)
"""

from __future__ import annotations

import time

import pandas as pd

from common import OUTPUTS, REPORTS, configure_logging, discover_nights, edf_header

log = configure_logging("02_header_scan")


def main() -> None:
    nights = discover_nights()
    log.info("Scanning headers for %d EDF files", len(nights))

    channel_rows: list[dict] = []
    night_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    t0 = time.time()
    for i, n in enumerate(nights, 1):
        try:
            hdr = edf_header(n.edf)
        except Exception as exc:  # noqa: BLE001 - audit must record every failure
            failures.append((str(n.edf), repr(exc)))
            log.error("Header parse failed for %s: %r", n.edf, exc)
            continue

        durations = [c["channel_duration_h"] for c in hdr["channels"]]
        night_rows.append(
            {
                "root": n.root_name,
                "participant_id": n.participant,
                "night": n.night,
                "night_id": n.night_id,
                "edf_file": n.edf.name,
                "start_datetime": hdr["start_datetime"],
                "n_records": hdr["n_records"],
                "record_duration_s": hdr["record_duration_s"],
                "recording_duration_h": round(hdr["recording_duration_h"], 6),
                "n_channels": hdr["n_channels"],
                "min_channel_duration_h": round(min(durations), 6),
                "max_channel_duration_h": round(max(durations), 6),
                "patient_field": hdr["patient_field"],
                "recording_field": hdr["recording_field"],
            }
        )
        for ch in hdr["channels"]:
            channel_rows.append(
                {
                    "root": n.root_name,
                    "participant_id": n.participant,
                    "night": n.night,
                    "night_id": n.night_id,
                    "edf_file": n.edf.name,
                    "start_datetime": hdr["start_datetime"],
                    "recording_duration_h": round(hdr["recording_duration_h"], 6),
                    "n_channels": hdr["n_channels"],
                    **ch,
                }
            )

        if i % 25 == 0 or i == len(nights):
            log.info("  %d/%d nights (%.1fs elapsed)", i, len(nights), time.time() - t0)

    signals = pd.DataFrame(channel_rows)
    headers = pd.DataFrame(night_rows)

    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    signals.to_csv(REPORTS / "paaws_signal_inventory.csv", index=False)
    headers.to_csv(OUTPUTS / "paaws_night_headers.csv", index=False)

    log.info("Wrote signal inventory: %d rows", len(signals))
    log.info("Wrote night headers: %d rows", len(headers))
    log.info("Failures: %d", len(failures))
    if len(failures):
        log.warning("First failures: %s", failures[:3])

    # ---- summary diagnostics ------------------------------------------------
    log.info("Channels per night distribution: %s", headers.n_channels.value_counts().to_dict())
    log.info(
        "Recording duration (h): min=%.2f median=%.2f max=%.2f",
        headers.recording_duration_h.min(),
        headers.recording_duration_h.median(),
        headers.recording_duration_h.max(),
    )
    short = headers[headers.recording_duration_h < 4]
    if len(short):
        log.warning("Nights shorter than 4 h: %d -> %s", len(short), short.night_id.tolist())

    label_stats = (
        signals.groupby("channel_label")
        .agg(
            nights_present=("night_id", "nunique"),
            participants_present=("participant_id", "nunique"),
            sample_rates=("sample_rate_hz", lambda s: sorted(set(s.round(3)))),
            units=("physical_dimension", lambda s: sorted(set(s))),
        )
        .sort_values("nights_present", ascending=False)
    )
    log.info("Unique channel labels: %d", len(label_stats))
    log.info(
        "Labels present on <100%% of nights:\n%s",
        label_stats[label_stats.nights_present < len(headers)].to_string(),
    )

    # Channel-layout families (exact ordered label set per night)
    layouts = (
        signals.groupby("night_id")["channel_label"]
        .apply(tuple)
        .value_counts()
    )
    log.info("Distinct ordered channel layouts: %d", len(layouts))
    for layout, count in layouts.items():
        log.info("  layout of %d channels used by %d nights", len(layout), count)

    # Same label with multiple sampling rates anywhere
    multi_rate = signals.groupby("channel_label")["sample_rate_hz"].nunique()
    multi_rate = multi_rate[multi_rate > 1]
    if len(multi_rate):
        log.warning("Labels with >1 sampling rate: %s", multi_rate.to_dict())


if __name__ == "__main__":
    main()
