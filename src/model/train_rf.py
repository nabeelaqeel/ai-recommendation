"""Stage 3: participant-level Random Forest baseline + ablations + thresholds.

Evaluation protocol (all participant-grouped, no participant crosses a fold):
- Outer: StratifiedGroupKFold, 5 folds x 5 repeats.
- Inner: 3-fold stratified CV *inside the training participants only* for
  hyperparameter selection, so the test fold never influences any choice.
- Metrics: macro-F1, balanced accuracy, per-class precision/recall/F1,
  confusion matrices, participant-level bootstrap CI on pooled out-of-fold
  predictions, permutation importance, Brier + reliability bins.
- Cost-sensitive operating point: within each outer fold, the decision
  threshold is chosen on inner-CV out-of-fold probabilities to reach the
  configured recall target on the See-Doctor (>=15) class, then applied to the
  untouched test fold; specificity/precision at that point are reported.

Everything is non-diagnostic research banding. Predicted scores are model
scores, not calibrated clinical probabilities.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.audit.common import OUTPUTS, PROJECT_ROOT
from src.features.config import load_config
from src.model.classifier import DISCLAIMER, Phase2Classifier
from src.model.dataset import ARMS, LABEL_2, LABEL_3, arm_columns, build_participant_table

MODELS = PROJECT_ROOT / "models"
RESULTS = OUTPUTS
N_REPEATS = 5
N_FOLDS = 5
SEED = 42
GRID = {"min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5]}
N_TREES = 300

log = logging.getLogger("train")


def rf(params: dict, n_trees: int = N_TREES) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_trees,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
        **params,
    )


def lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)),
        ]
    )


def _pick_params(X_tr, y_tr, seed: int) -> tuple[dict, float]:
    """Inner-CV hyperparameter selection on training participants only.

    Selection uses a lighter forest (100 trees); the final fold model uses the
    full 300. Standard speed practice, does not change the selection criterion.
    """
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best, best_score = None, -1.0
    for leaf in GRID["min_samples_leaf"]:
        for mf in GRID["max_features"]:
            p = {"min_samples_leaf": leaf, "max_features": mf}
            m = rf(p, n_trees=100)
            scores = []
            for tr, va in inner.split(X_tr, y_tr):
                m.fit(X_tr[tr], y_tr[tr])
                scores.append(f1_score(y_tr[va], m.predict(X_tr[va]), average="macro"))
            s = float(np.mean(scores))
            if s > best_score:
                best, best_score = p, s
    return best, best_score


def _threshold_for_recall(y_true: np.ndarray, p_pos: np.ndarray, pos_idx: int, target: float) -> float:
    """Largest threshold whose recall still meets the target (best specificity)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    best_t = None
    for t in np.sort(grid)[::-1]:
        rec = float(np.mean((p_pos >= t)[y_true == pos_idx])) if (y_true == pos_idx).any() else 0.0
        if rec >= target:
            best_t = float(t)
            break
    if best_t is None:  # target unreachable: fall back to the most sensitive point
        best_t = float(np.sort(grid)[0])
    return best_t


def evaluate(
    table: pd.DataFrame,
    cols: list[str],
    label_col: str,
    positive_class: str | None,
    cfg,
    tag: str,
    threshold_target: float | None = None,
    model_fn=rf,
    select_params: bool = True,
) -> dict:
    """Grouped, repeated cross-validation with optional cost-sensitive threshold."""
    X = table[cols].to_numpy(dtype=float)
    y_raw = table[label_col].to_numpy()
    classes = sorted(set(y_raw))
    y = np.array([classes.index(v) for v in y_raw])
    groups = table["participant_id"].to_numpy()
    pos_idx = classes.index(positive_class) if positive_class else None

    records: list[dict] = []
    oof_proba_accum = np.zeros((len(y), len(classes)))
    for repeat in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
            params, inner_score = (None, np.nan)
            if select_params:
                params, inner_score = _pick_params(X[tr], y[tr], SEED + 100 * repeat + fold)
            model = model_fn(params) if params else model_fn({})
            model.fit(X[tr], y[tr])

            proba = model.predict_proba(X[te])
            full_proba = np.zeros((len(te), len(classes)))
            for j, c in enumerate(model.classes_):
                full_proba[:, int(c)] = proba[:, j]
            oof_proba_accum[te] += full_proba
            pred = np.argmax(full_proba, axis=1)

            rec = {
                "tag": tag,
                "repeat": repeat,
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "params": str(params),
                "inner_macro_f1": round(float(inner_score), 4),
                "macro_f1": round(float(f1_score(y[te], pred, average="macro")), 4),
                "balanced_acc": round(float(balanced_accuracy_score(y[te], pred)), 4),
            }
            for i, cname in enumerate(classes):
                prec, rcall, f1c, _ = precision_recall_fscore_support(
                    y[te], pred, labels=[i], average="weighted", zero_division=0
                )
                rec[f"recall_{cname}"] = round(float(rcall), 4)
                rec[f"precision_{cname}"] = round(float(prec), 4)

            if threshold_target is not None and pos_idx is not None:
                # Choose the threshold on inner-CV OOF probabilities of the
                # TRAINING participants only, then apply it to the test fold.
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 200 + fold)
                m2 = model_fn(params) if params else model_fn({})
                p_in = cross_val_predict(m2, X[tr], y[tr], cv=inner_cv, method="predict_proba")
                p_in_pos = p_in[:, int(np.unique(y[tr]).tolist().index(pos_idx))]
                t = _threshold_for_recall(y[tr], p_in_pos, pos_idx, threshold_target)

                p_te_pos = full_proba[:, pos_idx]
                flagged = p_te_pos >= t
                is_pos = y[te] == pos_idx
                tp = int(np.sum(flagged & is_pos))
                fn = int(np.sum(~flagged & is_pos))
                fp = int(np.sum(flagged & ~is_pos))
                tn = int(np.sum(~flagged & ~is_pos))
                rec.update(
                    {
                        "threshold": t,
                        "recall_pos_at_t": round(tp / (tp + fn), 4) if (tp + fn) else np.nan,
                        "specificity_at_t": round(tn / (tn + fp), 4) if (tn + fp) else np.nan,
                        "precision_pos_at_t": round(tp / (tp + fp), 4) if (tp + fp) else np.nan,
                        "flagged_frac": round(float(flagged.mean()), 4),
                        "default_recall_pos": rec[f"recall_{classes[pos_idx]}"],
                    }
                )
            records.append(rec)

    oof_mean = oof_proba_accum / N_REPEATS
    pooled_pred = np.argmax(oof_mean, axis=1)
    pooled = {
        "tag": tag,
        "label_col": label_col,
        "classes": classes,
        "macro_f1": float(f1_score(y, pooled_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y, pooled_pred)),
        "cm": confusion_matrix(y, pooled_pred, labels=range(len(classes))),
        "oof_pred": pooled_pred,
        "y": y,
        "oof_proba": oof_mean,
        "report": classification_report(y, pooled_pred, target_names=classes, labels=range(len(classes)), zero_division=0, output_dict=True),
    }
    per_rep = pd.DataFrame(records)
    # bootstrap CI on pooled OOF predictions
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        boots.append(f1_score(y[idx], pooled_pred[idx], average="macro", zero_division=0))
    pooled["macro_f1_ci"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return {"per_fold": per_rep, "pooled": pooled, "threshold_target": threshold_target}


def summarize(res: dict) -> dict:
    per = res["per_fold"]
    s = {
        "tag": per["tag"].iloc[0],
        "macro_f1_mean": round(float(per["macro_f1"].mean()), 4),
        "macro_f1_sd": round(float(per["macro_f1"].std()), 4),
        "balanced_acc_mean": round(float(per["balanced_acc"].mean()), 4),
        "pooled_macro_f1": round(res["pooled"]["macro_f1"], 4),
        "macro_f1_ci_low": round(res["pooled"]["macro_f1_ci"][0], 4),
        "macro_f1_ci_high": round(res["pooled"]["macro_f1_ci"][1], 4),
    }
    for c in res["pooled"]["classes"]:
        s[f"recall_{c}_mean"] = round(float(per[f"recall_{c}"].mean()), 4) if f"recall_{c}" in per else np.nan
        s[f"precision_{c}_mean"] = round(float(per[f"precision_{c}"].mean()), 4) if f"precision_{c}" in per else np.nan
    if "threshold" in per.columns:
        for k in ("threshold", "recall_pos_at_t", "specificity_at_t", "precision_pos_at_t", "flagged_frac", "default_recall_pos"):
            s[f"{k}_mean"] = round(float(per[k].mean()), 4) if k in per.columns else np.nan
            s[f"{k}_sd"] = round(float(per[k].std()), 4) if k in per.columns else np.nan
    return s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3 RF training/evaluation")
    parser.add_argument("--quick", action="store_true", help="1 repeat, no grid (smoke test)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()
    global N_REPEATS, GRID
    if args.quick:
        N_REPEATS = 1
        GRID = {"min_samples_leaf": [2], "max_features": ["sqrt"]}

    t0 = time.perf_counter()
    table = build_participant_table(cfg)
    log.info("Participants: %d | 3-class: %s", len(table), table["label_3class"].value_counts().to_dict())

    summaries: list[dict] = []
    all_folds: list[pd.DataFrame] = []
    results: dict[str, dict] = {}

    # ---- ablation arms, primary 3-class label --------------------------
    # core and full both get the cost-sensitive threshold: core is the
    # deployable arm and must carry its own operating point.
    for arm in ("core", "core_posangle", "core_coverage", "full"):
        cols = arm_columns(arm, table)
        res = evaluate(
            table, cols, "label_3class", "see_doctor", cfg, f"arm_{arm}_3class",
            threshold_target=cfg.labels.target_recall_see_doctor if arm in ("core", "full") else None,
        )
        results[arm] = res
        summaries.append(summarize(res))
        all_folds.append(res["per_fold"])
        extra = ""
        if arm == "full" and "recall_pos_at_t_mean" in summaries[-1]:
            extra = f" | see-doctor recall@t {summaries[-1]['recall_pos_at_t_mean']:.3f}"
        log.info("arm %-14s macro-F1 %.3f ± %.3f%s", arm, summaries[-1]["macro_f1_mean"], summaries[-1]["macro_f1_sd"], extra)

    # ---- label rule comparison (full arm) -------------------------------
    res = evaluate(table, arm_columns("full", table), "label_3class_comparison", "see_doctor", cfg, "rule_mean_night_3class")
    results["rule_mean"] = res
    summaries.append(summarize(res))
    all_folds.append(res["per_fold"])
    log.info("mean-night rule macro-F1 %.3f", summaries[-1]["macro_f1_mean"])

    # ---- 2-class >=15 with cost-sensitive threshold ---------------------
    res2 = evaluate(
        table, arm_columns("full", table), "label_2class", "elevated", cfg,
        "two_class_ge15", threshold_target=cfg.labels.target_recall_see_doctor,
    )
    results["two_class"] = res2
    summaries.append(summarize(res2))
    all_folds.append(res2["per_fold"])
    log.info("2-class >=15 macro-F1 %.3f | recall@t %.3f", summaries[-1]["macro_f1_mean"], summaries[-1]["recall_pos_at_t_mean"])

    # ---- 2-class sensitivity >=5 ----------------------------------------
    res2b = evaluate(
        table, arm_columns("full", table), "label_2class_sensitivity", "elevated_5", cfg,
        "two_class_ge5", threshold_target=cfg.labels.target_recall_see_doctor,
    )
    results["two_class_ge5"] = res2b
    summaries.append(summarize(res2b))
    all_folds.append(res2b["per_fold"])

    # ---- logistic comparator (full arm, 3-class) ------------------------
    res_lr = evaluate(table, arm_columns("full", table), "label_3class", "see_doctor", cfg,
                      "logistic_full_3class", model_fn=lambda p: lr_pipeline(), select_params=False)
    results["logistic"] = res_lr
    summaries.append(summarize(res_lr))
    all_folds.append(res_lr["per_fold"])

    # ---- night-level sensitivity analysis -------------------------------
    night = _night_table(cfg)
    if night is not None:
        res_n = evaluate(
            night, arm_columns("full", night), "label_3class", "see_doctor", cfg,
            "night_level_3class", select_params=False,
        )
        results["night_level"] = res_n
        summaries.append(summarize(res_n))
        all_folds.append(res_n["per_fold"])

    # ---- permutation importance (core + full, repeat 0) -----------------
    imp_rows = _permutation_importance(table, results, cfg)
    imp_rows.to_csv(RESULTS / "phase2_permutation_importance.csv", index=False)

    # ---- calibration -----------------------------------------------------
    cal_rows = _calibration(results["full"]["pooled"], LABEL_3)
    cal_rows.to_csv(RESULTS / "phase2_calibration.csv", index=False)
    brier = {"arm": "full", "brier_multiclass": _safe_brier(results["full"]["pooled"])}

    # ---- outputs ---------------------------------------------------------
    pd.DataFrame(summaries).to_csv(RESULTS / "phase2_model_summary.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(RESULTS / "phase2_fold_records.csv", index=False)
    _save_oof(table, results)
    if not args.quick:
        _save_classifier(table, results, cfg)
    else:
        log.info("quick mode: skipping classifier artifact save")

    _write_report(cfg, table, summaries, results, imp_rows, cal_rows, brier, args.quick)
    log.info("Stage 3 done in %.1f min", (time.perf_counter() - t0) / 60)
    return 0


def _night_table(cfg) -> pd.DataFrame | None:
    """Night-level rows for the grouped sensitivity analysis."""
    from src.model.dataset import COMBINED_NIGHTLY, EXTRACTION_AUDIT, NIGHT_METRICS, reliable_night_mask

    try:
        nightly = pd.read_parquet(COMBINED_NIGHTLY)
        metrics = pd.read_csv(NIGHT_METRICS).set_index("night_id")
        audit = pd.read_csv(EXTRACTION_AUDIT)
        ok = set(audit.loc[audit["status"] == "ok", "night_id"])
        m = metrics.copy()
        m["reliable"] = reliable_night_mask(m, cfg)
        keep = m.index[m["reliable"] & m.index.isin(ok)]
        t = nightly[nightly["night_id"].isin(keep)].copy()
        t["label_3class"] = t["night_id"].map(lambda n: cfg.labels.band3(float(metrics.loc[n, "ahi_sleep"])))
        t = t[t["label_3class"] != "undefined"]
        return t.reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("Night-level analysis skipped: %s", exc)
        return None


def _permutation_importance(table: pd.DataFrame, results: dict, cfg) -> pd.DataFrame:
    rows = []
    for arm in ("core", "full"):
        cols = arm_columns(arm, table)
        X = table[cols].to_numpy(dtype=float)
        y = table["label_3class"].to_numpy()
        groups = table["participant_id"].to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
            model = rf({"min_samples_leaf": 2, "max_features": "sqrt"})
            model.fit(X[tr], y[tr])
            pi = permutation_importance(model, X[te], y[te], scoring="f1_macro", n_repeats=10, random_state=SEED)
            for c, mean, sd in zip(cols, pi.importances_mean, pi.importances_std):
                rows.append({"arm": arm, "fold": fold, "feature": c, "importance": float(mean), "sd": float(sd)})
    df = pd.DataFrame(rows)
    agg = df.groupby(["arm", "feature"])["importance"].agg(["mean", "std", "sem"]).reset_index()
    agg.columns = ["arm", "feature", "importance_mean", "importance_std", "importance_sem"]
    return agg.sort_values(["arm", "importance_mean"], ascending=[True, False])


def _calibration(pooled: dict, classes: list[str]) -> pd.DataFrame:
    rows = []
    y = pooled["y"]
    proba = pooled["oof_proba"]
    for i, c in enumerate(classes):
        p = proba[:, i]
        bins = np.quantile(p, np.linspace(0, 1, 6))
        bins[0], bins[-1] = -1e-9, 1e9
        for b in range(5):
            m = (p > bins[b]) & (p <= bins[b + 1])
            if m.sum() < 3:
                continue
            rows.append(
                {
                    "class": c,
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_predicted": round(float(p[m].mean()), 4),
                    "observed_fraction": round(float((y[m] == i).mean()), 4),
                }
            )
    return pd.DataFrame(rows)


def _safe_brier(pooled: dict) -> float:
    y = pooled["y"]
    proba = pooled["oof_proba"]
    return round(float(np.mean([brier_score_loss((y == i).astype(int), proba[:, i]) for i in range(proba.shape[1])])), 4)


def _save_oof(table: pd.DataFrame, results: dict) -> None:
    pooled = results["full"]["pooled"]
    df = pd.DataFrame(
        {
            "participant_id": table["participant_id"],
            "y_true": [pooled["classes"][i] for i in pooled["y"]],
            "y_pred": [pooled["classes"][i] for i in pooled["oof_pred"]],
        }
    )
    for j, c in enumerate(pooled["classes"]):
        df[f"p_{c}"] = pooled["oof_proba"][:, j].round(4)
    df.to_csv(RESULTS / "phase2_oof_predictions.csv", index=False)


def _save_classifier(table: pd.DataFrame, results: dict, cfg) -> None:
    """Persist the deployable (Garmin-strict core) model with its threshold."""
    cols = arm_columns("core", table)
    X = table[cols].to_numpy(dtype=float)
    y = table["label_3class"].to_numpy()
    params, _ = _pick_params(X, np.array([{"healthy": 0, "monitor": 1, "see_doctor": 2}[v] for v in y]), SEED)
    model = rf(params)
    model.fit(X, y)
    # Serial prediction: for single-row inference the joblib thread dispatch of
    # n_jobs=-1 costs ~50 ms per call; serial costs a few ms.
    model.n_jobs = 1
    # The core arm's OWN operating point on P(see_doctor) — never a threshold
    # borrowed from a different model's probability scale.
    core_folds = results["core"]["per_fold"]
    t = float(core_folds["threshold"].mean()) if "threshold" in core_folds else 0.5
    clf = Phase2Classifier(
        model=model,
        feature_names=cols,
        arm="core",
        label_order=LABEL_3,
        label_rule=cfg.labels.participant_rule,
        threshold_2class=t,
        config_path=str(cfg.source_path),
        target_recall=cfg.labels.target_recall_see_doctor,
        fitted_on=f"{len(table)} participants, PAAWS R2 v1.3 (research bands, not diagnoses)",
        positive_class="see_doctor",
        extra={"disclaimer": DISCLAIMER},
    )
    path = clf.save()
    log.info("Saved deployable classifier -> %s (threshold %.2f on P(see_doctor))", path, t)


def _write_report(cfg, table, summaries, results, imp_rows, cal_rows, brier, quick: bool) -> None:
    from src.model.dataset import FEATURE_GROUPS

    s = {r["tag"]: r for r in summaries}
    lines = [
        "# Phase 2 model report — participant-level Random Forest (research bands)",
        "",
        "> **Non-diagnostic.** Healthy / Monitor / See Doctor are proposed sleep-related guidance",
        "> bands for a supplementary recommendation layer. This model does not diagnose sleep",
        "> apnoea, has not been validated on Garmin data, and `predict_proba` outputs are",
        "> uncalibrated model scores, not clinical probabilities.",
        "",
        f"- Participants: **{len(table)}** (label rule: `{cfg.labels.participant_rule}`)",
        f"- Class counts (worst reliable night): {table['label_3class'].value_counts().to_dict()}",
        f"- CV: StratifiedGroupKFold {N_FOLDS} folds × {N_REPEATS} repeats, participant-grouped;",
        "  hyperparameters selected inside training folds only.",
        "",
        "## Ablation arms (3-class, primary label rule)",
        "",
        "| arm | features | macro-F1 (mean±sd) | pooled macro-F1 [95% CI] | balanced acc |",
        "|---|---|---|---|---|",
    ]
    arm_desc = {
        "arm_core_3class": "HR, RMSSD, SpO2/T90, Activity (Garmin-strict)",
        "arm_core_posangle_3class": "core + PosAngle",
        "arm_core_coverage_3class": "core + coverage flags",
        "arm_full_3class": "core + PosAngle + coverage flags",
        "rule_mean_night_3class": "full arm, mean-night label rule (comparison)",
        "two_class_ge15": "full arm, binary >=15 with cost-sensitive threshold",
        "two_class_ge5": "full arm, binary >=5 (sensitivity split)",
        "logistic_full_3class": "logistic regression comparator (full arm)",
        "night_level_3class": "night-level rows, grouped by participant",
    }
    for tag, desc in arm_desc.items():
        if tag not in s:
            continue
        r = s[tag]
        lines.append(
            f"| {tag} | {desc} | {r['macro_f1_mean']:.3f} ± {r['macro_f1_sd']:.3f} | "
            f"{r['pooled_macro_f1']:.3f} [{r['macro_f1_ci_low']:.3f}–{r['macro_f1_ci_high']:.3f}] | {r['balanced_acc_mean']:.3f} |"
        )

    lines += [
        "",
        f"## Cost-sensitive operating point (See-Doctor recall target {cfg.labels.target_recall_see_doctor:.0%})",
        "",
        "Thresholds are chosen on inner-CV out-of-fold probabilities within each training fold,",
        "then applied to the untouched test fold — the test fold never influences the threshold.",
        "",
        "### 3-class: flag See-Doctor when P(see_doctor) >= t",
        "",
        "| arm | metric | mean ± sd |",
        "|---|---|---|",
    ]
    for tag, name in [("arm_core_3class", "core (deployable)"), ("arm_full_3class", "full")]:
        if tag not in s:
            continue
        r = s[tag]
        for k, label in [
            ("threshold", "decision threshold"),
            ("recall_pos_at_t", "See-Doctor recall at threshold"),
            ("specificity_at_t", "specificity (non-See-Doctor correctly not flagged)"),
            ("precision_pos_at_t", "precision of See-Doctor flag"),
            ("flagged_frac", "fraction of participants flagged"),
            ("default_recall_pos", "See-Doctor recall at default argmax"),
        ]:
            if k + "_mean" in r:
                lines.append(f"| {name} | {label} | {r[k + '_mean']:.3f} ± {r.get(k + '_sd', float('nan')):.3f} |")

    lines += [
        "",
        "### 2-class (>=15 vs <15), full arm",
        "",
        "| metric | mean ± sd |",
        "|---|---|",
    ]
    if "two_class_ge15" in s:
        r = s["two_class_ge15"]
        for k, label in [
            ("threshold", "decision threshold"),
            ("recall_pos_at_t", "recall (>=15) at threshold"),
            ("specificity_at_t", "specificity at threshold"),
            ("precision_pos_at_t", "precision at threshold"),
            ("flagged_frac", "fraction flagged elevated"),
            ("default_recall_pos", "recall at default 0.5"),
        ]:
            if k + "_mean" in r:
                lines.append(f"| {label} | {r[k + '_mean']:.3f} ± {r.get(k + '_sd', float('nan')):.3f} |")

    lines += ["", "## Confusion matrix (pooled out-of-fold, full arm, 3-class)", ""]
    cm = results["full"]["pooled"]["cm"]
    classes = results["full"]["pooled"]["classes"]
    lines.append("| true \\ pred | " + " | ".join(classes) + " |")
    lines.append("|---|" + "---:|" * len(classes))
    for i, c in enumerate(classes):
        lines.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |")

    lines += ["", "## Per-class recall/precision (full arm, mean over folds)", "",
              "| class | recall | precision |", "|---|---:|---:|"]
    r = s.get("arm_full_3class")
    if r:
        for c in classes:
            lines.append(f"| {c} | {r.get(f'recall_{c}_mean', float('nan')):.3f} | {r.get(f'precision_{c}_mean', float('nan')):.3f} |")

    lines += ["", "## Permutation importance (full arm, top 12)", "",
              "| feature | importance |", "|---|---:|"]
    top = imp_rows[imp_rows["arm"] == "full"].head(12)
    for _, row in top.iterrows():
        lines.append(f"| {row['feature']} | {row['importance_mean']:.4f} |")

    lines += ["", "## Calibration (pooled out-of-fold, full arm)", "",
              f"- Mean per-class Brier score: **{brier['brier_multiclass']}** (lower is better; 3-class)",
              "", "| class | bin | n | mean predicted | observed |", "|---|---|---:|---:|---:|"]
    for _, row in cal_rows.iterrows():
        lines.append(
            f"| {row['class']} | {row['bin']} | {row['n']} | {row['mean_predicted']} | {row['observed_fraction']} |"
        )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Feature construction is label-blind (nan-mean over reliable nights); the worst night is",
        "  selected only for the label, never for the features.",
        "- Coverage flags are excluded from the core arm so the model cannot use sensor dropout",
        "  as a severity shortcut; the `core_coverage` arm quantifies that risk.",
        "- All scores are research estimates on PSG data. The PSG → wrist-worn domain gap",
        "  (ECG vs PPG HRV, 3 Hz vs intermittent SpO2) is unresolved and unmeasured here.",
        "",
        DISCLAIMER,
    ]
    if quick:
        path = RESULTS / "phase2_model_report_QUICK.md"
    else:
        path = PROJECT_ROOT / "reports" / "phase2_model_report.md"
    path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", path)


if __name__ == "__main__":
    sys.exit(main())
