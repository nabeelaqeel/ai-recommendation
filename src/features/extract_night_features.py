"""Extract per-epoch and per-night features for PAAWS participant-nights.

Read-only with respect to the raw dataset. One night is held in memory per
worker process; results are written as one Parquet (epochs) plus one JSON
(nightly row) per night, so the run is resumable and the combined tables are
rebuilt from those files.

Usage
-----
    python -m src.features.extract_night_features --pilot
    python -m src.features.extract_night_features --night DS_118-N1 --night DS_30-N1
    python -m src.features.extract_night_features --all
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pyedflib

from src.audit.common import OUTPUTS, PROJECT_ROOT, Night, configure_logging, discover_nights
from src.features.config import FeatureConfig, Phase2Config, load_config
from src.features.features import NIGHTLY_FEATURES, extract_night

PROCESSED = PROJECT_ROOT / "data" / "processed"
NIGHT_DIR = PROCESSED / "night_features"
COMBINED_EPOCHS = PROCESSED / "features_epoch.parquet"
COMBINED_NIGHTLY = PROCESSED / "features_night.parquet"
AUDIT = OUTPUTS / "phase2_extraction_audit.csv"
SELECTION = OUTPUTS / "phase2_pilot_selection.csv"

# Nights that exercise a known data quirk; always included in the pilot.
MANDATORY = {
    "DS_41-N2": "annotation clock drift 262.9 s",
    "DS_108-N1": "100% Wake night",
    "DS_237-N2": "100% Wake night",
    "DS_18-N1": "no ECG/EKG channel",
    "DS_123-N1": "no Heart Rate channel",
    "DS_97-N1": "pathological HR / no usable QRS",
}

log = logging.getLogger("extract")


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
def _worker(payload: dict) -> dict:
    """Run in a worker process: extract one night and write its files."""
    night: Night = payload["night"]
    cfg: FeatureConfig = payload["cfg"]
    t0 = time.perf_counter()
    base = {
        "night_id": night.night_id,
        "participant_id": night.participant,
        "root": night.root_name,
        "status": "ok",
    }
    try:
        if not night.edf.exists():
            raise FileNotFoundError(f"EDF missing: {night.edf}")
        if not night.csv.exists():
            raise FileNotFoundError(f"CSV missing: {night.csv}")
        result = extract_night(night, cfg, lambda p: pyedflib.EdfReader(str(p)))

        result.epochs.to_parquet(NIGHT_DIR / f"{night.night_id}.parquet", index=False)
        nightly = {"night_id": night.night_id, "participant_id": night.participant, **result.nightly}
        (NIGHT_DIR / f"{night.night_id}.nightly.json").write_text(json.dumps(nightly, indent=2, default=str))

        audit = {**base, **result.audit}
        audit.pop("hr_candidates", None)
        audit["n_rows_written"] = int(len(result.epochs))
        audit["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return {
            "audit": audit,
            "nightly": nightly,
            "hr_candidates": result.audit.get("hr_candidates", []),
            "spo2_agreement": result.audit.get("spo2_agreement"),
            "status": "ok",
        }
    except Exception as exc:  # noqa: BLE001 - one bad night must not kill the run
        return {
            "audit": {
                **base,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.perf_counter() - t0, 2),
            },
            "nightly": None,
            "hr_candidates": [],
            "spo2_agreement": None,
            "status": "error",
            "traceback": traceback.format_exc(),
        }


# --------------------------------------------------------------------------
# pilot selection
# --------------------------------------------------------------------------
def _inventory() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "reports" / "paaws_signal_inventory.csv")


def _channel_sets(inv: pd.DataFrame) -> tuple[set[str], set[str]]:
    """(nights with a duplicate Heart Rate, nights with any SpO2 channel)."""
    hr = inv[inv["channel_label"] == "Heart Rate"].groupby("night_id").size()
    dup_hr = set(hr[hr > 1].index)
    has_spo2 = set(inv[inv["channel_label"].isin(["Saturation", "SpO2 B-B"])]["night_id"])
    return dup_hr, has_spo2


def select_pilot(nights: list[Night], cfg: Phase2Config) -> tuple[list[Night], pd.DataFrame]:
    """Deterministic 30-night pilot covering every known data quirk."""
    by_id = {n.night_id: n for n in nights}
    metrics = pd.read_csv(OUTPUTS / "paaws_night_metrics.csv").set_index("night_id")
    inv = _inventory()
    dup_hr, has_spo2 = _channel_sets(inv)
    ekg_nights = set(inv[inv["channel_label"] == "EKG"]["night_id"])
    missing_spo2 = {n for n in by_id if n not in has_spo2}
    roots = sorted({n.root_name for n in by_id.values()})

    picks: list[tuple[str, str]] = []
    seen: set[str] = set()

    def take(night_id: str, reason: str) -> None:
        if night_id in by_id and night_id not in seen:
            picks.append((night_id, reason))
            seen.add(night_id)

    for nid, why in MANDATORY.items():
        take(nid, f"quirk: {why}")

    short_tst = [n for n in metrics.index if n in by_id and pd.notna(metrics.loc[n, "tst_h"]) and metrics.loc[n, "tst_h"] < 2]
    for nid in short_tst[:2]:
        take(nid, "short TST (<2 h)")
    for nid in sorted(missing_spo2)[:2]:
        take(nid, "no SpO2 channel")
    for nid in sorted(dup_hr)[:4]:
        take(nid, "duplicate Heart Rate channel")

    idx = [n for n in by_id if n in metrics.index]
    m = metrics.loc[idx]
    strata = {
        "band_see_doctor": list(m[m["ahi_3class"] == "See-Doctor-band (>=15)"].index),
        "band_monitor": list(m[m["ahi_3class"] == "Monitor-band (5-15)"].index),
        "band_healthy": list(m[m["ahi_3class"] == "Healthy-band (<5)"].index),
        "ekg_layout": [n for n in by_id if n in ekg_nights],
        "second_root": [n for n in by_id if by_id[n].root_name == roots[-1]],
    }
    quotas = {
        "band_see_doctor": 6,
        "band_monitor": 8,
        "band_healthy": 8,
        "ekg_layout": 8,
        "second_root": 8,
    }
    rng = random.Random(cfg.runtime.pilot_seed)
    for name, pool in strata.items():
        have = sum(1 for _, r in picks if r == f"stratum: {name}")
        need = quotas[name] - have
        if need <= 0:
            continue
        remaining = [n for n in pool if n not in seen]
        rng.shuffle(remaining)
        for nid in remaining[:need]:
            take(nid, f"stratum: {name}")

    if len(picks) < cfg.runtime.pilot_nights:
        remaining = [n for n in by_id if n not in seen]
        rng.shuffle(remaining)
        for nid in remaining[: cfg.runtime.pilot_nights - len(picks)]:
            take(nid, "random fill")

    picks = picks[: cfg.runtime.pilot_nights]
    rows = []
    for nid, reason in picks:
        n = by_id[nid]
        row = metrics.loc[nid] if nid in metrics.index else None
        rows.append(
            {
                "night_id": nid,
                "participant_id": n.participant,
                "root": n.root_name,
                "reason": reason,
                "tst_h": _round(row["tst_h"], 3) if row is not None else None,
                "ahi_sleep": _round(row["ahi_sleep"], 2) if row is not None else None,
                "ahi_3class": row["ahi_3class"] if row is not None else None,
                "reliable": bool(row["ahi_reliable_tst_ge_2h"]) if row is not None else None,
                "duplicate_hr": nid in dup_hr,
                "no_spo2_channel": nid in missing_spo2,
            }
        )
    return [by_id[nid] for nid, _ in picks], pd.DataFrame(rows)


def _round(v, nd: int):
    return round(float(v), nd) if pd.notna(v) else None


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PAAWS Phase 2 feature extraction")
    parser.add_argument("--pilot", action="store_true", help="30-night stratified pilot (default)")
    parser.add_argument("--all", action="store_true", help="all 253 nights")
    parser.add_argument("--night", action="append", default=[], help="explicit night id (repeatable)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="re-extract nights that already have output")
    parser.add_argument(
        "--reaggregate",
        action="store_true",
        help="recompute nightly rows from cached epoch Parquets without re-reading EDFs",
    )
    args = parser.parse_args(argv)

    configure_logging("extract")
    cfg = load_config()
    NIGHT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    nights = discover_nights()
    if args.reaggregate:
        return _reaggregate(nights, cfg)
    if args.night:
        wanted = {n.strip() for n in args.night if n.strip()}
        unknown = wanted - {n.night_id for n in nights}
        if unknown:
            log.error("Unknown night id(s): %s", ", ".join(sorted(unknown)))
            return 2
        nights = [n for n in nights if n.night_id in wanted]
    elif not args.all:
        nights, selection = select_pilot(nights, cfg)
        selection.to_csv(SELECTION, index=False)
        log.info("Pilot selection written to %s", SELECTION)

    workers = args.workers or cfg.runtime.workers
    log.info("Extracting %d night(s) with %d workers", len(nights), workers)

    todo, skipped = [], 0
    for n in nights:
        done = (NIGHT_DIR / f"{n.night_id}.parquet").exists() and (
            NIGHT_DIR / f"{n.night_id}.nightly.json"
        ).exists()
        if done and not args.force:
            skipped += 1
            continue
        todo.append(n)
    if skipped:
        log.info("Skipping %d night(s) already extracted (--force to redo)", skipped)

    payloads = [{"night": n, "cfg": cfg.features} for n in todo]
    audits: list[dict] = []
    hr_resolutions: list[dict] = []
    t0 = time.perf_counter()

    if not payloads:
        log.info("Nothing to extract.")
    elif workers <= 1 or len(payloads) <= 2:
        for i, p in enumerate(payloads, 1):
            res = _worker(p)
            audits.append(res["audit"])
            _collect_hr(res, hr_resolutions, night_map={n.night_id: n for n in todo})
            _log_progress(i, len(payloads), res)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                audits.append(res["audit"])
                _collect_hr(res, hr_resolutions, night_map={n.night_id: n for n in todo})
                _log_progress(i, len(payloads), res)

    if hr_resolutions:
        hr_df = pd.DataFrame(hr_resolutions)
        HR_LOG = OUTPUTS / "phase2_hr_resolution.csv"
        if HR_LOG.exists():
            old = pd.read_csv(HR_LOG)
            old = old[~old["night_id"].isin(hr_df["night_id"])]
            hr_df = pd.concat([old, hr_df], ignore_index=True)
        hr_df.sort_values(["night_id", "channel_index"]).to_csv(HR_LOG, index=False)
        log.info("Heart Rate duplicate-resolution log: %s", HR_LOG)

    elapsed = time.perf_counter() - t0
    if audits:
        new = pd.DataFrame(audits)
        if AUDIT.exists():
            old = pd.read_csv(AUDIT)
            old = old[~old["night_id"].isin(new["night_id"])]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(AUDIT, index=False)
        ok = new[new["status"].isin(["ok", "no_sleep_epochs"])]
        failed = new[~new["status"].isin(["ok", "no_sleep_epochs"])]
        log.info(
            "Extraction finished: %d ok (%d with no scored sleep), %d failed, %.1f min, mean %.1f s/night",
            len(ok),
            int((new["status"] == "no_sleep_epochs").sum()),
            len(failed),
            elapsed / 60,
            float(ok["elapsed_s"].mean()) if len(ok) else float("nan"),
        )
        for _, row in failed.iterrows():
            log.error("%s %s", row["night_id"], row.get("error", ""))

    _rebuild_combined()
    return 0


def _collect_hr(res: dict, hr_resolutions: list[dict], night_map: dict[str, Night]) -> None:
    """Log how a duplicated Heart Rate channel was resolved, for the audit."""
    cands = res.get("hr_candidates") or []
    if len(cands) < 2:
        return
    chosen = res["audit"].get("hr_channel_index")
    for c in cands:
        hr_resolutions.append(
            {
                "night_id": res["audit"]["night_id"],
                "root": night_map[res["audit"]["night_id"]].root_name,
                "channel_index": c["channel_index"],
                "channel_label": c["channel_label"],
                "n_samples": c["n_samples"],
                "valid_fraction": c["valid_fraction"],
                "median_bpm": c["median_bpm"],
                "chosen": c["channel_index"] == chosen,
            }
        )


def _reaggregate(nights: list[Night], cfg: Phase2Config) -> int:
    """Rebuild nightly rows from cached epoch Parquets (no raw re-reads)."""
    from src.features.features import aggregate_nightly

    by_id = {n.night_id: n for n in nights}
    targets = sorted(NIGHT_DIR.glob("*.parquet"))
    if not targets:
        log.error("No cached epoch Parquets in %s", NIGHT_DIR)
        return 2
    done = 0
    for path in targets:
        nid = path.name[: -len(".parquet")]
        if nid not in by_id:
            log.warning("No Night record for %s; skipping", nid)
            continue
        epochs = pd.read_parquet(path)
        nightly = {"night_id": nid, "participant_id": by_id[nid].participant, **aggregate_nightly(epochs, cfg.features)}
        (NIGHT_DIR / f"{nid}.nightly.json").write_text(json.dumps(nightly, indent=2, default=str))
        done += 1
    log.info("Re-aggregated %d nightly rows from cached epochs", done)
    _rebuild_combined()
    return 0


def _log_progress(i: int, total: int, res: dict) -> None:
    a = res["audit"]
    n = res["nightly"] or {}
    rows = a.get("n_rows_written", a.get("n_sleep_epochs", "-"))
    log.info(
        "[%2d/%2d] %-11s %-5s rows=%-5s hr=%-5s spo2=%-5s hrv=%-5s %6.1fs",
        i,
        total,
        a["night_id"],
        a["status"],
        rows,
        _fmt_pct(n.get("hr_coverage")),
        _fmt_pct(n.get("spo2_coverage")),
        _fmt_pct(n.get("hrv_valid_frac")),
        a.get("elapsed_s", 0.0),
    )


def _fmt_pct(v) -> str:
    try:
        return f"{100 * float(v):.0f}%"
    except (TypeError, ValueError):
        return "-"


def _rebuild_combined() -> None:
    parquets = sorted(NIGHT_DIR.glob("*.parquet"))
    jsons = sorted(NIGHT_DIR.glob("*.nightly.json"))
    if parquets:
        frames = []
        for f in parquets:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as exc:  # noqa: BLE001
                log.error("Could not read %s: %s", f, exc)
        if frames:
            combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
            combined.to_parquet(COMBINED_EPOCHS, index=False)
            log.info(
                "Combined epoch table: %d rows x %d cols, %d nights -> %s (%.1f MB)",
                len(combined),
                combined.shape[1],
                combined["night_id"].nunique(),
                COMBINED_EPOCHS,
                COMBINED_EPOCHS.stat().st_size / 1e6,
            )
    if jsons:
        rows = [json.loads(p.read_text()) for p in jsons]
        nightly = pd.DataFrame(rows)
        ordered = ["night_id", "participant_id"] + NIGHTLY_FEATURES
        nightly = nightly[[c for c in ordered if c in nightly.columns]]
        nightly = nightly.sort_values("night_id").reset_index(drop=True)
        nightly.to_parquet(COMBINED_NIGHTLY, index=False)
        log.info("Combined nightly table: %d nights -> %s", len(nightly), COMBINED_NIGHTLY)


if __name__ == "__main__":
    sys.exit(main())
