"""Parse all 253 scored-events CSVs.

Produces:
  reports/paaws_annotation_inventory.csv   (event-type level inventory)
  outputs/paaws_night_stage_metrics.csv    (per-night staging + alignment)
  outputs/paaws_event_counts_by_night.csv  (per-night event counts)

No signal data is read.
"""

from __future__ import annotations

import pandas as pd

from common import (
    ALL_STAGES,
    EPOCH_SECONDS,
    OUTPUTS,
    REPORTS,
    RESPIRATORY_EVENTS,
    SLEEP_STAGES,
    configure_logging,
    discover_nights,
)

log = configure_logging("03_annotation_scan")

EXPECTED_COLUMNS = ["Event", "Duration", "Start Time", "End Time", "Start Epoch", "End Epoch"]


def main() -> None:
    nights = discover_nights()
    headers = pd.read_csv(OUTPUTS / "paaws_night_headers.csv").set_index("night_id")

    event_rows: list[dict] = []
    stage_rows: list[dict] = []
    count_rows: list[dict] = []
    schema_issues: list[tuple[str, str]] = []

    for n in nights:
        df = pd.read_csv(n.csv, parse_dates=["Start Time", "End Time"])
        if "Event.1" in df.columns:
            if (df["Event"] != df["Event.1"]).any():
                schema_issues.append((n.night_id, "Event and Event.1 differ"))
            df = df.drop(columns=["Event.1"])
        if list(df.columns) != EXPECTED_COLUMNS:
            schema_issues.append((n.night_id, str(list(df.columns))))
        df["Start Epoch"] = df["Start Epoch"].astype(int)
        df["End Epoch"] = df["End Epoch"].astype(int)
        event_rows.append({"night_id": n.night_id, "participant_id": n.participant, **df["Event"].value_counts().to_dict()})

        stages = df[df.Event.isin(ALL_STAGES)].sort_values("Start Time").copy()
        sleep = stages[stages.Event.isin(SLEEP_STAGES)]
        wake = stages[stages.Event == "Wake"]

        tst_s = float(sleep.Duration.sum())
        first_sleep = sleep["Start Time"].min() if len(sleep) else pd.NaT
        last_sleep_end = sleep["End Time"].max() if len(sleep) else pd.NaT
        spt_s = (last_sleep_end - first_sleep).total_seconds() if len(sleep) else 0.0
        waso_s = 0.0
        if len(sleep) and len(wake):
            in_spt = wake[(wake["Start Time"] >= first_sleep) & (wake["End Time"] <= last_sleep_end)]
            waso_s = float(in_spt.Duration.sum())

        analysis_start = df.loc[df.Event == "Analysis Start", "Start Time"]
        analysis_start = analysis_start.iloc[0] if len(analysis_start) else pd.NaT

        # Epoch-grid consistency: derive the grid anchor from the first stage event.
        if len(stages):
            first = stages.iloc[0]
            epoch0 = first["Start Time"] - pd.Timedelta(seconds=EPOCH_SECONDS * int(first["Start Epoch"]))
            residuals = (stages["Start Time"] - epoch0).dt.total_seconds() - EPOCH_SECONDS * stages["Start Epoch"]
            max_resid = float(residuals.abs().max())
        else:
            epoch0 = pd.NaT
            max_resid = float("nan")

        epochs = sorted(stages["Start Epoch"].unique())
        gaps = sum(1 for a, b in zip(epochs, epochs[1:]) if b - a > 1)
        dup_epochs = int(stages["Start Epoch"].duplicated().sum())
        epoch_spacing = float(stages["Start Time"].diff().dt.total_seconds().median()) if len(stages) > 1 else float("nan")

        hdr = headers.loc[n.night_id]
        edf_start = pd.Timestamp(hdr.start_datetime).tz_localize(None)
        edf_end = edf_start + pd.Timedelta(hours=float(hdr.recording_duration_h))
        first_stage = stages["Start Time"].min() if len(stages) else pd.NaT
        last_stage_end = stages["End Time"].max() if len(stages) else pd.NaT
        start_offset = (first_stage - edf_start).total_seconds() if len(stages) else float("nan")
        end_offset = (edf_end - last_stage_end).total_seconds() if len(stages) else float("nan")

        stage_rows.append(
            {
                "night_id": n.night_id,
                "participant_id": n.participant,
                "night": n.night,
                "n_stage_epochs": len(stages),
                "n_sleep_epochs": len(sleep),
                "n_wake_epochs": len(wake),
                "tst_h": round(tst_s / 3600, 4),
                "spt_h": round(spt_s / 3600, 4),
                "waso_h": round(waso_s / 3600, 4),
                "sleep_efficiency": round(tst_s / spt_s, 4) if spt_s > 0 else float("nan"),
                "n1_h": round(float(sleep.loc[sleep.Event == "N1", "Duration"].sum()) / 3600, 4),
                "n2_h": round(float(sleep.loc[sleep.Event == "N2", "Duration"].sum()) / 3600, 4),
                "n3_h": round(float(sleep.loc[sleep.Event == "N3", "Duration"].sum()) / 3600, 4),
                "rem_h": round(float(sleep.loc[sleep.Event == "REM", "Duration"].sum()) / 3600, 4),
                "first_sleep_start": str(first_sleep) if pd.notna(first_sleep) else "",
                "last_sleep_end": str(last_sleep_end) if pd.notna(last_sleep_end) else "",
                "analysis_start": str(analysis_start) if pd.notna(analysis_start) else "",
                "sol_h": round((first_sleep - analysis_start).total_seconds() / 3600, 4)
                if pd.notna(analysis_start) and pd.notna(first_sleep)
                else float("nan"),
                "edf_start": str(edf_start),
                "edf_duration_h": float(hdr.recording_duration_h),
                "first_stage_minus_edf_start_s": round(start_offset, 2),
                "edf_end_minus_last_stage_end_s": round(end_offset, 2),
                "stage_outside_edf": bool(start_offset < -60 or end_offset < -60),
                "epoch_grid_max_residual_s": round(max_resid, 3) if max_resid == max_resid else float("nan"),
                "epoch_spacing_s": round(epoch_spacing, 4),
                "stage_epoch_gaps": gaps,
                "duplicate_stage_epochs": dup_epochs,
                "no_sleep_epochs": bool(len(sleep) == 0),
            }
        )

        # Respiratory events: de-duplicate identical-start rows (1 known case).
        resp = df[df.Event.isin(RESPIRATORY_EVENTS)].drop_duplicates(subset=["Start Time"], keep="first")
        stage_map = dict(zip(stages["Start Epoch"], stages["Event"]))
        onset_stage = resp["Start Epoch"].map(stage_map)
        resp_sleep = int(onset_stage.isin(SLEEP_STAGES).sum())
        resp_wake = int((onset_stage == "Wake").sum())
        resp_unmapped = int(onset_stage.isna().sum())
        hyp = resp[resp.Event.str.contains("Hypopnea")]
        apn = resp[resp.Event.str.startswith("A") | (resp.Event == "Apnea")]
        desat = df[df.Event == "Desat"]
        plm = df[df.Event.isin(["PLM", "PLMS", "LM"])]
        arousal = df[df.Event.isin(["Arousal", "Spontaneous Arousal"])]
        snore = df[df.Event.isin(["Single Snore", "Snore Train"])]
        posture = df[df.Event.isin(["Supine", "Prone", "Left", "Right", "Upright"])]

        count_rows.append(
            {
                "night_id": n.night_id,
                "participant_id": n.participant,
                "night": n.night,
                "n_rows": len(df),
                "resp_events_total": len(resp),
                "resp_events_sleep": resp_sleep,
                "resp_events_wake": resp_wake,
                "resp_events_unmapped": resp_unmapped,
                "hypopnea_events": len(hyp),
                "apnea_events": len(apn),
                "obstructive_events": int(resp.Event.isin(["A. Obstructive", "H. Obstructive"]).sum()),
                "central_events": int(resp.Event.isin(["A. Central", "H. Central"]).sum()),
                "mixed_events": int((resp.Event == "A. Mixed").sum()),
                "desat_events": len(desat),
                "arousal_events": len(arousal),
                "plm_events": len(plm),
                "snore_events": len(snore),
                "posture_epochs": len(posture),
                "supine_epochs": int((posture.Event == "Supine").sum()),
            }
        )

    stage_metrics = pd.DataFrame(stage_rows).sort_values(["participant_id", "night"]).reset_index(drop=True)
    counts = pd.DataFrame(count_rows).sort_values(["participant_id", "night"]).reset_index(drop=True)

    # Long-format event counts for the inventory's per-night presence checks
    events_long = pd.concat(
        [
            pd.DataFrame({"night_id": r["night_id"], "participant_id": r["participant_id"],
                          "event": list(r.keys())[2:], "count": list(r.values())[2:]})
            for r in event_rows
        ],
        ignore_index=True,
    )
    events_long = events_long[events_long["count"] > 0]

    inventory = (
        events_long.groupby("event")
        .agg(
            total_events=("count", "sum"),
            nights_present=("night_id", "nunique"),
            participants_present=("participant_id", "nunique"),
        )
        .sort_values("total_events", ascending=False)
    )

    # Duration statistics from the raw annotations (single pass)
    dur_frames = []
    for n in nights:
        df = pd.read_csv(n.csv, usecols=["Event", "Duration"])
        dur_frames.append(df)
    durations = pd.concat(dur_frames, ignore_index=True)
    dur_stats = durations.groupby("Event")["Duration"].agg(
        median_duration_s="median", mean_duration_s="mean", min_duration_s="min", max_duration_s="max",
        total_duration_h=lambda s: s.sum() / 3600,
    )
    inventory = inventory.join(dur_stats).reset_index().rename(columns={"index": "event"})
    inventory = inventory.sort_values("total_events", ascending=False)

    stage_metrics.to_csv(OUTPUTS / "paaws_night_stage_metrics.csv", index=False)
    counts.to_csv(OUTPUTS / "paaws_event_counts_by_night.csv", index=False)
    inventory.to_csv(REPORTS / "paaws_annotation_inventory.csv", index=False)

    log.info("Wrote %s (%d event types)", REPORTS / "paaws_annotation_inventory.csv", len(inventory))
    log.info("Nights with schema deviations: %d", len(schema_issues))
    if schema_issues:
        log.warning("Schema deviations: %s", schema_issues[:5])

    log.info(
        "TST (h): min=%.2f median=%.2f max=%.2f",
        stage_metrics.tst_h.min(), stage_metrics.tst_h.median(), stage_metrics.tst_h.max(),
    )
    log.info("Nights with TST < 2 h: %d -> %s", (stage_metrics.tst_h < 2).sum(),
             stage_metrics.loc[stage_metrics.tst_h < 2, "night_id"].tolist())
    log.info("Nights flagged stage_outside_edf: %d -> %s", int(stage_metrics.stage_outside_edf.sum()),
             stage_metrics.loc[stage_metrics.stage_outside_edf, "night_id"].tolist())
    log.info("Max epoch-grid residual across nights: %.3f s", stage_metrics.epoch_grid_max_residual_s.max())
    log.info("Nights with stage epoch gaps: %d | duplicate stage epochs: %d",
             int((stage_metrics.stage_epoch_gaps > 0).sum()), int((stage_metrics.duplicate_stage_epochs > 0).sum()))
    log.info("Total respiratory events (dedup): %d | in sleep: %d",
             counts.resp_events_total.sum(), counts.resp_events_sleep.sum())


if __name__ == "__main__":
    main()
