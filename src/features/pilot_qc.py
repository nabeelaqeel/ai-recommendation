"""Quality-control gates for the Phase 2 pilot extraction.

Run after `extract_night_features --pilot`. Reads only project outputs plus the
raw signals needed for the independent T90 spot-check. Writes
`outputs/phase2_pilot_qc.csv` (per-night checks) and
`outputs/phase2_pilot_qc.md` (summary + verdicts).
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import pyedflib

from src.audit.common import OUTPUTS, PROJECT_ROOT, configure_logging, discover_nights
from src.features.config import load_config
from src.features.features import FORBIDDEN_FEATURE_PATTERNS, NIGHTLY_FEATURES

PROCESSED = PROJECT_ROOT / "data" / "processed"
NIGHT_DIR = PROCESSED / "night_features"
QC_CSV = OUTPUTS / "phase2_pilot_qc.csv"
QC_MD = OUTPUTS / "phase2_pilot_qc.md"

log = logging.getLogger("pilot_qc")


# ---------------------------------------------------------------------------
# independent T90 recomputation (deliberately a different implementation)
# ---------------------------------------------------------------------------
def recompute_t90(night, cfg) -> tuple[float, float]:
    """T90 hours and valid hours inside scored sleep, recomputed from raw SpO2.

    Uses whole-second epoch boundaries and a sample-count integration that does
    not share code with the extractor, so agreement is evidence, not tautology.
    """
    with pyedflib.EdfReader(str(night.edf)) as f:
        labels = list(f.getSignalLabels())
        idx = next((i for i, lab in enumerate(labels) if lab in ("Saturation", "SpO2 B-B")), None)
        if idx is None:
            return float("nan"), float("nan")
        spo2 = f.readSignal(idx)
        fs = float(f.getSampleFrequency(idx))
        start = pd.Timestamp(f.getStartdatetime())
    lo, hi = cfg.features.spo2_valid_pct
    valid = np.isfinite(spo2) & (spo2 >= lo) & (spo2 <= hi)

    ev = pd.read_csv(night.csv, parse_dates=["Start Time"])
    sleep = ev[ev["Event"].isin(["N1", "N2", "N3", "REM"])]
    if sleep.empty:
        return float("nan"), float("nan")

    t90_s = 0.0
    valid_s = 0.0
    for _, row in sleep.iterrows():
        a = int(round((row["Start Time"] - start).total_seconds() * fs))
        b = a + int(round(float(row["Duration"]) * fs))
        a, b = max(0, a), min(len(spo2), b)
        if b <= a:
            continue
        m = valid[a:b]
        n_valid = int(m.sum())
        valid_s += n_valid / fs
        t90_s += int((spo2[a:b][m] < cfg.features.desat_threshold_pct).sum()) / fs
    return t90_s / 3600.0, valid_s / 3600.0


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 pilot QC")
    parser.add_argument("--t90-nights", type=int, default=3, help="how many nights to spot-check T90")
    args = parser.parse_args(argv)
    configure_logging("pilot_qc")
    cfg = load_config()

    selection = pd.read_csv(OUTPUTS / "phase2_pilot_selection.csv")
    audit = pd.read_csv(OUTPUTS / "phase2_extraction_audit.csv")
    nightly = pd.read_parquet(PROCESSED / "features_night.parquet")
    metrics = pd.read_csv(OUTPUTS / "paaws_night_metrics.csv").set_index("night_id")
    nights = {n.night_id: n for n in discover_nights()}

    pilot_ids = list(selection["night_id"])
    audit_p = audit[audit["night_id"].isin(pilot_ids)].set_index("night_id")
    nightly_p = nightly[nightly["night_id"].isin(pilot_ids)].set_index("night_id")
    checks: list[dict] = []
    failures: list[str] = []

    def verdict(ok: bool, name: str, detail: str) -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    # ---- 1. completeness -----------------------------------------------
    missing_files = [nid for nid in pilot_ids if not (NIGHT_DIR / f"{nid}.parquet").exists()]
    verdict(not missing_files, "files", f"missing parquet: {missing_files}")
    missing_nightly = sorted(set(pilot_ids) - set(nightly_p.index))
    verdict(not missing_nightly, "nightly", f"missing nightly rows: {missing_nightly}")

    # ---- 2. row counts vs Phase 1 --------------------------------------
    for nid in pilot_ids:
        n_rows = int(audit_p.loc[nid, "n_rows_written"]) if nid in audit_p.index else -1
        expected = int(metrics.loc[nid, "n_sleep_windows"]) if nid in metrics.index else -1
        p1_defined = nid in metrics.index and pd.notna(metrics.loc[nid, "n_sleep_windows"])
        same = (n_rows == expected) if p1_defined else True
        checks.append(
            {
                "night_id": nid,
                "check": "row_count_matches_phase1",
                "value": n_rows,
                "expected": expected if p1_defined else None,
                "pass": bool(same),
            }
        )
    bad_rows = [c["night_id"] for c in checks if c["check"] == "row_count_matches_phase1" and not c["pass"]]
    verdict(not bad_rows, "row_counts", f"mismatch: {bad_rows}")

    # ---- 3. plausibility ranges + gate verification ---------------------
    for nid, row in nightly_p.iterrows():
        for name, lo, hi in [
            ("hr_median", 30, 120),
            ("spo2_median", 85, 100),
            ("activity_mean", 0, 10),
            ("rmssd_median", 5, 200),
        ]:
            v = row.get(name)
            ok = pd.isna(v) or (lo <= v <= hi)
            checks.append({"night_id": nid, "check": f"range_{name}", "value": v, "expected": f"[{lo},{hi}]", "pass": bool(ok)})
        # A flagged night must actually have its values withdrawn.
        if bool(row.get("hr_artefact", False)):
            checks.append(
                {"night_id": nid, "check": "hr_artefact_withdrawn", "value": row.get("hr_median"),
                 "expected": "NaN", "pass": bool(pd.isna(row.get("hr_median")))}
            )
        if bool(row.get("spo2_artefact", False)):
            checks.append(
                {"night_id": nid, "check": "spo2_artefact_withdrawn", "value": row.get("spo2_median"),
                 "expected": "NaN", "pass": bool(pd.isna(row.get("spo2_median")))}
            )
        if not bool(row.get("hrv_sufficient", False)):
            checks.append(
                {"night_id": nid, "check": "hrv_insufficient_withdrawn", "value": row.get("rmssd_median"),
                 "expected": "NaN", "pass": bool(pd.isna(row.get("rmssd_median")))}
            )
    range_bad = [(c["night_id"], c["check"]) for c in checks if c["check"].startswith(("range_", "hr_", "spo2_", "hrv_")) and not c["pass"]]
    verdict(not range_bad, "ranges_and_gates", f"violations: {range_bad}")

    # ---- 3b. missing-SpO2 nights must be absent, not all-invalid ---------
    sel_indexed = selection.set_index("night_id")
    for nid in pilot_ids:
        if nid not in nightly_p.index:
            continue
        if nid in audit_p.index and audit_p.loc[nid, "status"] == "no_sleep_epochs":
            continue  # no scored sleep: coverage is undefined by construction
        no_channel = bool(sel_indexed.loc[nid, "no_spo2_channel"]) if nid in sel_indexed.index else False
        cov = nightly_p.loc[nid, "spo2_coverage"]
        if no_channel:
            ok = pd.isna(cov) or float(cov) == 0.0
            checks.append({"night_id": nid, "check": "no_spo2_channel_absent", "value": cov, "expected": "0/NaN", "pass": bool(ok)})
            # Dropout must never look like "no desaturation".
            t90 = nightly_p.loc[nid, "spo2_t90_h"]
            checks.append({"night_id": nid, "check": "no_spo2_t90_undefined", "value": t90, "expected": "NaN",
                           "pass": bool(pd.isna(t90))})
        else:
            # A present channel with zero coverage would be a new data-quality finding.
            checks.append({"night_id": nid, "check": "spo2_channel_has_data", "value": cov, "expected": ">0",
                           "pass": bool(pd.notna(cov) and float(cov) > 0)})
    spo2_bad = [(c["night_id"], c["check"], c["value"]) for c in checks
                if (c["check"].startswith("no_spo2") or c["check"] == "spo2_channel_has_data") and not c["pass"]]
    for c in spo2_bad:
        log.warning("SpO2 availability anomaly: %s %s=%s", *c)
    verdict(not [c for c in spo2_bad if c[1] != "spo2_channel_has_data"], "spo2_absence", f"{spo2_bad}")

    # ---- 4. leakage -----------------------------------------------------
    leaks = [c for c in nightly_p.columns if any(p in c.lower() for p in FORBIDDEN_FEATURE_PATTERNS)]
    verdict(not leaks, "leakage", f"forbidden columns present: {leaks}")
    missing_feats = [c for c in NIGHTLY_FEATURES if c not in nightly_p.columns]
    verdict(not missing_feats, "schema", f"missing nightly features: {missing_feats}")

    # ---- 5. coverage distribution --------------------------------------
    cov = nightly_p[["hr_coverage", "spo2_coverage", "hrv_valid_frac"]].astype(float)
    summary_cov = cov.agg(["median", "mean", "min", "max"]).round(3)
    log.info("Coverage summary:\n%s", summary_cov.to_string())

    # ---- 6. independent T90 spot-check ---------------------------------
    t90_ids = [
        nid
        for nid in pilot_ids
        if nid in nightly_p.index and pd.notna(nightly_p.loc[nid, "spo2_t90_h"])
    ]
    # prefer nights with a non-trivial T90, then fill
    t90_ids = sorted(t90_ids, key=lambda n: -float(nightly_p.loc[n, "spo2_t90_h"]))[: args.t90_nights]
    for nid in t90_ids:
        t90, valid_h = recompute_t90(nights[nid], cfg)
        got = float(nightly_p.loc[nid, "spo2_t90_h"])
        rel = abs(got - t90) / t90 if t90 > 0 else abs(got - t90)
        ok = rel <= 0.05
        checks.append(
            {
                "night_id": nid,
                "check": "t90_recompute",
                "value": round(got, 4),
                "expected": round(t90, 4),
                "pass": bool(ok),
                "detail": f"rel_diff={rel:.3f}, valid_h={valid_h:.2f} vs {float(nightly_p.loc[nid, 'spo2_valid_h']):.2f}",
            }
        )
        verdict(ok, "t90", f"{nid}: extracted {got:.3f} h vs recomputed {t90:.3f} h (rel {rel:.3f})")

    # ---- 7. duplicate-HR resolution ------------------------------------
    hr_log_path = OUTPUTS / "phase2_hr_resolution.csv"
    if hr_log_path.exists():
        hr_log = pd.read_csv(hr_log_path)
        unresolved = hr_log.groupby("night_id")["chosen"].sum()
        bad_hr = list(unresolved[unresolved != 1].index)
        verdict(not bad_hr, "hr_duplicates", f"nights without exactly one chosen channel: {bad_hr}")
    else:
        verdict(True, "hr_duplicates", "no duplicated-HR nights in this run")

    # ---- 8. runtime / scalability --------------------------------------
    elapsed = audit_p["elapsed_s"].astype(float)
    per_night = float(elapsed[audit_p["status"] == "ok"].median()) if (audit_p["status"] == "ok").any() else float("nan")
    projected_full_min = per_night * 253 / 4 / 60
    verdict(projected_full_min <= 180, "runtime", f"projected full run {projected_full_min:.0f} min > 180 min budget")

    # ---- 9. structured missingness + gate flags -------------------------
    hr_flag = sorted(nightly_p.index[nightly_p["hr_artefact"].astype(bool)])
    spo2_flag = sorted(nightly_p.index[nightly_p["spo2_artefact"].astype(bool)])
    hrv_insuff = sorted(nightly_p.index[~nightly_p["hrv_sufficient"].astype(bool)])
    missing = {
        "hr_median": int(nightly_p["hr_median"].isna().sum()),
        "rmssd_median": int(nightly_p["rmssd_median"].isna().sum()),
        "spo2_median": int(nightly_p["spo2_median"].isna().sum()),
        "spo2_t90_h": int(nightly_p["spo2_t90_h"].isna().sum()),
        "activity_mean": int(nightly_p["activity_mean"].isna().sum()),
        "posangle_median": int(nightly_p["posangle_median"].isna().sum()),
    }
    high_hrv = nightly_p[nightly_p["hrv_valid_frac"] >= 0.9]
    rmssd_hi = (
        f"{high_hrv['rmssd_median'].min():.1f}-{high_hrv['rmssd_median'].max():.1f} ms"
        if len(high_hrv)
        else "n/a"
    )

    # ---- write outputs --------------------------------------------------
    qc = pd.DataFrame(checks)
    qc.to_csv(QC_CSV, index=False)
    n_pass = int(qc["pass"].sum())
    n_total = len(qc)

    def _md_table(df: pd.DataFrame) -> list[str]:
        header = "| " + " | ".join(str(c) for c in df.reset_index().columns) + " |"
        sep = "|" + "---|" * len(df.reset_index().columns)
        rows = [
            "| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in df.reset_index().to_numpy()
        ]
        return [header, sep, *rows]

    lines = [
        "# Phase 2 pilot QC (30 nights)",
        "",
        f"- Nights extracted: **{len(pilot_ids)}** ({int((audit_p['status'] == 'no_sleep_epochs').sum())} with no scored sleep)",
        f"- QC checks: **{n_pass}/{n_total} passed**",
        f"- Mean extraction time: **{per_night:.1f} s/night** (serial median); projected full run **~{projected_full_min:.0f} min** at 4 workers",
        "",
        "## Coverage over scored sleep epochs",
        "",
        *_md_table(summary_cov),
        "",
        "## T90 independent spot-check",
        "",
    ]
    t90_rows = qc[qc["check"] == "t90_recompute"]
    lines += [
        "| night | extracted h | recomputed h | rel diff |",
        "|---|---:|---:|---:|",
    ]
    for _, r in t90_rows.iterrows():
        rel = abs(float(r["value"]) - float(r["expected"])) / float(r["expected"]) if float(r["expected"]) > 0 else 0
        lines.append(f"| {r['night_id']} | {r['value']} | {r['expected']} | {rel:.3f} |")

    lines += [
        "",
        "## Night-level QC gates",
        "",
        f"- HR artefact nights (values withdrawn, flag kept): **{', '.join(hr_flag) or 'none'}**",
        f"- SpO2 artefact nights: **{', '.join(spo2_flag) or 'none'}**",
        f"- Nights without a representative nightly RMSSD (<30 QC-passed epochs): **{', '.join(hrv_insuff)}**",
        f"- RMSSD range on nights with >=90% HRV coverage: **{rmssd_hi}**",
        "",
        f"## Structured missingness (nights with NaN, of {len(nightly_p)})",
        "",
        "| feature | nights missing |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in missing.items()]
    lines += [
        "",
        "Missingness is preserved, never imputed: absent channels, artefact nights and",
        "insufficient-HRV nights are NaN with an explicit flag, so sensor dropout cannot",
        "masquerade as a physiological value (e.g. T90 = 0 h on a night with no SpO2).",
        "",
        "## Failures",
        ""]
    lines += [f"- {f}" for f in failures] or ["- none"]
    QC_MD.write_text("\n".join(lines) + "\n")

    log.info("QC: %d/%d checks passed -> %s", n_pass, n_total, QC_MD)
    for f in failures:
        log.warning("FAIL %s", f)
    if failures:
        log.warning("QC FAILED — resolve before the 253-night run.")
        return 1
    log.info("QC PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
