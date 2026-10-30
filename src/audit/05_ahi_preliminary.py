"""Preliminary TST / respiratory-event / AHI analysis from the annotation CSVs.

All 253 nights are processed. AHI is defined as unique respiratory events
(apnoea + hypopnoea sub-types) with onset in a scored sleep epoch, divided by
scored total sleep time. This is a research calculation pending codebook
confirmation; it is not a clinical AHI.

Writes:
  outputs/paaws_night_metrics.csv
  outputs/ahi_class_distribution.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import EPOCH_SECONDS, OUTPUTS, RESPIRATORY_EVENTS, SLEEP_STAGES, configure_logging, discover_nights

log = configure_logging("05_ahi_preliminary")

AHI_BANDS = [(-np.inf, 5, "Normal (<5)"), (5, 15, "Mild (5-15)"), (15, 30, "Moderate (15-30)"), (30, np.inf, "Severe (>=30)")]
AHI_3 = [(-np.inf, 5, "Healthy-band (<5)"), (5, 15, "Monitor-band (5-15)"), (15, np.inf, "See-Doctor-band (>=15)")]
MIN_TST_H = 2.0  # reliability floor for AHI calculation


def band(v: float, bands) -> str:
    if not np.isfinite(v):
        return "Undefined"
    for lo, hi, name in bands:
        if lo <= v < hi:
            return name
    return "Undefined"


def main() -> None:
    nights = discover_nights()
    stage_metrics = pd.read_csv(OUTPUTS / "paaws_night_stage_metrics.csv").set_index("night_id")
    count_metrics = pd.read_csv(OUTPUTS / "paaws_event_counts_by_night.csv").set_index("night_id")

    rows = []
    for n in nights:
        st = stage_metrics.loc[n.night_id]
        cnt = count_metrics.loc[n.night_id]

        df = pd.read_csv(n.csv, usecols=["Event", "Start Time", "Start Epoch"])
        stages = df[df.Event.isin(["Wake", *SLEEP_STAGES])]
        stage_map = dict(zip(stages["Start Epoch"].astype(int), stages["Event"]))

        resp = df[df.Event.isin(RESPIRATORY_EVENTS)].drop_duplicates(subset=["Start Time"], keep="first")
        onset_stage = resp["Start Epoch"].astype(int).map(stage_map)
        resp_sleep = resp[onset_stage.isin(SLEEP_STAGES)]
        n_sleep_epoch_events = resp_sleep["Start Epoch"].nunique()
        n_sleep_epochs = int(st.n_sleep_epochs)
        tst_h = float(st.tst_h)

        valid = tst_h > 0
        reliable = tst_h >= MIN_TST_H
        ahi = len(resp_sleep) / tst_h if valid else np.nan
        ahi_all = len(resp) / tst_h if valid else np.nan

        rows.append(
            {
                "night_id": n.night_id,
                "participant_id": n.participant,
                "night": n.night,
                "tst_h": round(tst_h, 4),
                "spt_h": float(st.spt_h),
                "waso_h": float(st.waso_h),
                "sleep_efficiency": float(st.sleep_efficiency),
                "resp_events_total": int(cnt.resp_events_total),
                "resp_events_sleep": len(resp_sleep),
                "resp_events_on_unscored_or_wake": int(cnt.resp_events_total - len(resp_sleep)),
                "hypopnea_events": int(cnt.hypopnea_events),
                "apnea_events": int(cnt.apnea_events),
                "ahi_sleep": round(ahi, 3) if np.isfinite(ahi) else np.nan,
                "ahi_all_events": round(ahi_all, 3) if np.isfinite(ahi_all) else np.nan,
                "ahi_category": band(ahi, AHI_BANDS),
                "ahi_3class": band(ahi, AHI_3),
                "ahi_valid": bool(valid),
                "ahi_reliable_tst_ge_2h": bool(reliable),
                "n_sleep_windows": n_sleep_epochs,
                "n_positive_windows": int(n_sleep_epoch_events),
                "window_positive_rate": round(n_sleep_epoch_events / n_sleep_epochs, 4) if n_sleep_epochs else np.nan,
                "edf_duration_h": float(st.edf_duration_h),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["participant_id", "night"]).reset_index(drop=True)
    metrics.to_csv(OUTPUTS / "paaws_night_metrics.csv", index=False)

    # ---------------------------------------------------------------- summaries
    valid = metrics[metrics.ahi_valid]
    reliable = metrics[metrics.ahi_valid & metrics.ahi_reliable_tst_ge_2h]
    log.info("Nights total: %d | AHI defined: %d | reliable (TST>=2h): %d", len(metrics), len(valid), len(reliable))
    log.info("Nights with AHI undefined (TST=0): %s", metrics.loc[~metrics.ahi_valid, "night_id"].tolist())
    log.info(
        "AHI (reliable nights): min=%.2f p25=%.2f median=%.2f p75=%.2f max=%.2f",
        reliable.ahi_sleep.min(), reliable.ahi_sleep.quantile(0.25), reliable.ahi_sleep.median(),
        reliable.ahi_sleep.quantile(0.75), reliable.ahi_sleep.max(),
    )
    log.info("TST (h): median=%.2f IQR=%.2f-%.2f", metrics.tst_h.median(), metrics.tst_h.quantile(0.25), metrics.tst_h.quantile(0.75))

    dist_rows = []
    for level, dfv in [("night (all defined)", valid), ("night (TST>=2h)", reliable)]:
        for cat in [b[2] for b in AHI_BANDS] + ["Undefined"]:
            n = int((dfv.ahi_category == cat).sum())
            dist_rows.append({"level": level, "band": cat, "n": n, "pct": round(100 * n / max(len(dfv), 1), 1)})
        for cat in [b[2] for b in AHI_3]:
            n = int((dfv.ahi_3class == cat).sum())
            dist_rows.append({"level": level + " [3-class]", "band": cat, "n": n, "pct": round(100 * n / max(len(dfv), 1), 1)})

    # Participant level: worst night (max) and mean of valid nights

    def agg_participant(g: pd.DataFrame) -> pd.Series:
        gv = g[g.ahi_valid]
        gr = g[g.ahi_reliable_tst_ge_2h]
        return pd.Series(
            {
                "nights": len(g),
                "nights_valid": len(gv),
                "nights_reliable": len(gr),
                "ahi_max": gr.ahi_sleep.max() if len(gr) else np.nan,
                "ahi_mean": gr.ahi_sleep.mean() if len(gr) else np.nan,
                "tst_mean_h": gr.tst_h.mean() if len(gr) else np.nan,
            }
        )

    p = metrics.groupby("participant_id").apply(agg_participant, include_groups=False).reset_index()
    # Participant labels are derived only from reliable nights (TST >= 2 h).
    p["ahi_max_category"] = p.ahi_max.apply(lambda v: band(v, AHI_BANDS))
    p["ahi_max_3class"] = p.ahi_max.apply(lambda v: band(v, AHI_3))
    p["ahi_mean_category"] = p.ahi_mean.apply(lambda v: band(v, AHI_BANDS))
    p["ahi_mean_3class"] = p.ahi_mean.apply(lambda v: band(v, AHI_3))
    p = p.merge(
        metrics[metrics.ahi_reliable_tst_ge_2h].groupby("participant_id").size().rename("nights_reliable_check").reset_index(),
        on="participant_id", how="left",
    )
    p["nights_reliable_check"] = p.nights_reliable_check.fillna(0).astype(int)
    assert (p.nights_reliable == p.nights_reliable_check).all()
    p = p.drop(columns=["nights_reliable_check"])
    for col in ["nights", "nights_valid", "nights_reliable"]:
        p[col] = p[col].astype(int)
    p["ahi_max"] = p.ahi_max.round(3)
    p["ahi_mean"] = p.ahi_mean.round(3)
    p["tst_mean_h"] = p.tst_mean_h.round(4)
    p_valid = p[p.nights_reliable > 0].copy()
    p_valid.to_csv(OUTPUTS / "paaws_participant_labels.csv", index=False)

    for level, col in [("participant (worst reliable night)", "ahi_max_category"),
                       ("participant (mean reliable nights)", "ahi_mean_category")]:
        for cat in [b[2] for b in AHI_BANDS] + ["Undefined"]:
            n = int((p_valid[col] == cat).sum())
            dist_rows.append({"level": level, "band": cat, "n": n, "pct": round(100 * n / max(len(p_valid), 1), 1)})
    for level, col in [("participant (worst reliable night) [3-class]", "ahi_max_3class"),
                       ("participant (mean reliable nights) [3-class]", "ahi_mean_3class")]:
        for cat in [b[2] for b in AHI_3]:
            n = int((p_valid[col] == cat).sum())
            dist_rows.append({"level": level, "band": cat, "n": n, "pct": round(100 * n / max(len(p_valid), 1), 1)})

    dist = pd.DataFrame(dist_rows)
    dist.to_csv(OUTPUTS / "ahi_class_distribution.csv", index=False)
    log.info("Participant-level (worst night) 4-band counts:\n%s", p_valid.ahi_max_category.value_counts().to_string())
    log.info("Participant-level (worst night) 3-class counts:\n%s", p_valid.ahi_max_3class.value_counts().to_string())

    # Window-level balance
    log.info(
        "Sleep windows: %d | positive: %d (%.2f%%) | per-night positive rate: median=%.3f IQR=%.3f-%.3f",
        reliable.n_sleep_windows.sum(), reliable.n_positive_windows.sum(),
        100 * reliable.n_positive_windows.sum() / reliable.n_sleep_windows.sum(),
        reliable.window_positive_rate.median(), reliable.window_positive_rate.quantile(0.25),
        reliable.window_positive_rate.quantile(0.75),
    )
    log.info("Nights with zero positive windows: %d", int((reliable.n_positive_windows == 0).sum()))

    # Night-to-night consistency for participants with 2 reliable nights
    two = p_valid[p_valid.nights_reliable == 2].copy()
    rel = metrics[metrics.ahi_reliable_tst_ge_2h]
    m = rel.pivot(index="participant_id", columns="night", values="ahi_sleep").dropna()
    if len(m) > 2:
        rho = m.corr(method="spearman").iloc[0, 1]
        night_cat = rel.pivot(index="participant_id", columns="night", values="ahi_category").dropna()
        agree4 = float((night_cat[1] == night_cat[2]).mean()) if len(night_cat) else np.nan
        night_cat3 = rel.pivot(index="participant_id", columns="night", values="ahi_3class").dropna()
        agree3 = float((night_cat3[1] == night_cat3[2]).mean()) if len(night_cat3) else np.nan
        log.info(
            "Two-night participants with both valid: %d | Spearman AHI=%.3f | same 4-band=%.1f%% | same 3-class=%.1f%%",
            len(m), rho, 100 * agree4, 100 * agree3,
        )
        log.info("Median |AHI night1 - night2| = %.2f", float((m[1] - m[2]).abs().median()))

    # AHI completeness vs recording
    log.info(
        "Nights with TST<2h (excluded from reliable AHI): %d -> %s",
        int((metrics.ahi_valid & ~metrics.ahi_reliable_tst_ge_2h).sum()),
        metrics.loc[metrics.ahi_valid & ~metrics.ahi_reliable_tst_ge_2h, "night_id"].tolist(),
    )
    log.info("Wrote %s and %s", OUTPUTS / "paaws_night_metrics.csv", OUTPUTS / "ahi_class_distribution.csv")


if __name__ == "__main__":
    main()
