"""EV pack datasets: clean -> features -> model selection -> one final test.

Protocol (the same for every target)
  1. 20 % of batteries / vehicles are set aside as the TEST set before anything is
     fitted (seeded, stratified by chemistry). Their rows are used exactly once, at
     the end, to score the model chosen on the other 80 % (DEV).
  2. On DEV, 5-fold cross-validation grouped by battery / vehicle compares model
     families (baseline, linear, random forest, histogram GB, XGBoost) and a random
     search over XGBoost settings. All preprocessing (category encoding, imputation,
     scaling) is inside the sklearn Pipeline, so it is refitted in every fold.
  3. The best CV model (lowest MAE, or highest macro-F1 for classes) is refitted on
     all of DEV, scored on TEST with 95 % bootstrap intervals (resampling whole units)
     and saved to models/ev_*.joblib.

Also reported (reports/results/ev_*.csv):
  * leakage checks: what the score would be with the columns that re-encode SoH;
  * feature / cleaning ablations on DEV;
  * a cross-dataset experiment: does pooling the two EV datasets help?

Usage:
    python -m src.models.train_ev            # about 10-20 minutes
    python -m src.models.train_ev --quick    # fewer tuning trials (for a smoke test)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import EV_DIR, EV_TEST_FRACTION, FIGURES_DIR, MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from src.data.ev_loaders import injected_outlier_oracle, load_ev_degradation, load_ev_telematics
from src.features.ev_features import (EV1_CATEGORICAL_FEATURES, EV1_RAW_NUMERIC, EV2_CATEGORICAL_FEATURES,
                                      EV2_PRETRIP_CATEGORICAL, EV2_PRETRIP_NUMERIC, EV2_RAW_NUMERIC,
                                      ev1_features, ev2_features, implied_soh_from_range)
from src.models.common import as_object, feature_ranges
from src.preprocessing.ev_cleaning import (clean_ev_degradation, clean_ev_telematics, make_ev1_masker)
from src.visualization.style import CATEGORICAL, INK_2, SEQ_CMAP, apply_style, save

RESULTS = REPORTS_DIR / "results"
FIG = FIGURES_DIR / "ev"
N_FOLDS = 5
N_BOOT = 1000


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def choose_test_units(df: pd.DataFrame, unit: str, stratum: str, frac: float = EV_TEST_FRACTION,
                      seed: int = RANDOM_SEED) -> set:
    """Seeded, stratified choice of whole units for the final test set."""
    per_unit = df.groupby(unit)[stratum].agg(lambda s: s.mode().iat[0] if s.notna().any() else "NA")
    rng = np.random.default_rng(seed)
    test: list = []
    for _, ids in per_unit.groupby(per_unit):
        ids = np.array(sorted(ids.index))
        rng.shuffle(ids)
        test += list(ids[:int(round(len(ids) * frac))])
    return set(test)


def prepare_ev_degradation() -> tuple[pd.DataFrame, list[str], dict, dict]:
    raw = load_ev_degradation()
    test_units = choose_test_units(raw, "battery_id", "chemistry")
    is_test = raw.battery_id.isin(test_units).to_numpy()
    ranged, _ = clean_ev_degradation(raw)
    masker = make_ev1_masker().fit(ranged[~is_test])           # scales from DEV batteries only
    clean, rep = clean_ev_degradation(raw, masker)
    oracle = injected_outlier_oracle()
    flags = clean[[f"flag_outlier_{c}" for c in masker.columns]].to_numpy()
    tp, fp = int((flags & oracle.to_numpy()).sum()), int((flags & ~oracle.to_numpy()).sum())
    rep["outlier_audit_vs_injected"] = {"injected": int(oracle.to_numpy().sum()), "flagged": tp + fp,
                                        "precision": round(tp / max(tp + fp, 1), 3),
                                        "recall": round(tp / oracle.to_numpy().sum(), 3)}
    df, numeric = ev1_features(clean)
    df["split"] = np.where(df.battery_id.isin(test_units), "test", "dev")
    return df, numeric, rep, masker.to_dict()


def prepare_ev_telematics() -> tuple[pd.DataFrame, list[str], dict]:
    raw = load_ev_telematics()
    test_units = choose_test_units(raw, "vehicle_id", "battery_type")
    clean, rep = clean_ev_telematics(raw)
    df, numeric = ev2_features(clean)
    df["split"] = np.where(df.vehicle_id.isin(test_units), "test", "dev")
    df["fire_risk_2"] = np.where(df.fire_risk == "Low", "Low", "Medium/High")   # 'High' has 3 rows
    return df, numeric, rep


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
XGB_DEFAULT = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                   colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED)
XGB_SPACE = dict(n_estimators=[400, 800, 1500], learning_rate=[0.02, 0.03, 0.05],
                 max_depth=[3, 4, 5, 6, 8], min_child_weight=[1, 5, 10, 30], subsample=[0.6, 0.8, 1.0],
                 colsample_bytree=[0.4, 0.6, 0.8], reg_lambda=[0.5, 1, 5, 20], gamma=[0, 0.05, 0.2])


def _ordinal(cats, nums) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", Pipeline([("object", FunctionTransformer(as_object, feature_names_out="one-to-one")),
                          ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                                    encoded_missing_value=np.nan, dtype=float))]), cats),
        ("num", "passthrough", nums)])


def _onehot(cats, nums) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", Pipeline([("object", FunctionTransformer(as_object, feature_names_out="one-to-one")),
                          ("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("scale", StandardScaler())]), nums)])


def make_estimator(family: str, kind: str, cats: list[str], nums: list[str], params: dict | None = None,
                   n_classes: int = 2) -> Pipeline:
    reg = kind == "regression"
    if family == "Baseline":
        return Pipeline([("pre", _ordinal(cats, nums)),
                         ("model", DummyRegressor() if reg else DummyClassifier(strategy="prior"))])
    if family == "Linear":
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)) if reg else LogisticRegression(max_iter=2000, C=1.0)
        return Pipeline([("pre", _onehot(cats, nums)), ("model", model)])
    if family == "Random Forest":
        cls = RandomForestRegressor if reg else RandomForestClassifier
        return Pipeline([("pre", _ordinal(cats, nums)),
                         ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("model", cls(n_estimators=400, min_samples_leaf=3, max_features=0.5, n_jobs=-1,
                                       random_state=RANDOM_SEED))])
    if family == "HistGB":
        cls = HistGradientBoostingRegressor if reg else HistGradientBoostingClassifier
        return Pipeline([("pre", _ordinal(cats, nums)),
                         ("model", cls(max_iter=800, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                       random_state=RANDOM_SEED))])
    if family.startswith("XGBoost"):
        p = {**XGB_DEFAULT, **(params or {})}
        if reg:
            model = XGBRegressor(**p)
        else:
            model = XGBClassifier(**p, objective="binary:logistic" if n_classes == 2 else "multi:softprob")
        return Pipeline([("pre", _ordinal(cats, nums)), ("model", model)])
    raise ValueError(family)


FAMILIES = ["Baseline", "Linear", "Random Forest", "HistGB", "XGBoost"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_scores(y, p, tol: tuple[float, float]) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.abs(p - y)
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p), f"within_{tol[0]:g}": float(np.mean(e <= tol[0])),
            f"within_{tol[1]:g}": float(np.mean(e <= tol[1]))}


def classification_scores(y, proba, n_classes: int) -> dict:
    y = np.asarray(y, int)
    pred = proba.argmax(1)
    auc = (roc_auc_score(y, proba[:, 1]) if n_classes == 2
           else roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=range(n_classes)))
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_F1": f1_score(y, pred, average="macro"), "AUC": auc,
            "log_loss": log_loss(y, proba / proba.sum(1, keepdims=True), labels=range(n_classes)),
            "majority_class_accuracy": float(np.bincount(y, minlength=n_classes).max() / len(y))}


def bootstrap_ci(units: np.ndarray, fn, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> dict:
    """95 % interval of each metric in fn(idx) when whole units are resampled."""
    uniq, inv = np.unique(units, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws.append(fn(np.concatenate([rows_of[k] for k in pick])))
    d = pd.DataFrame(draws)
    return {f"{c}_ci95": [float(d[c].quantile(0.025)), float(d[c].quantile(0.975))] for c in d}


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    key: str                  # file name stem: models/ev_<key>.joblib
    dataset: str
    target: str
    kind: str                 # "regression" | "classification"
    unit: str
    nums: list[str]
    cats: list[str]
    unit_label: str = ""      # e.g. "SoH %-points"
    tol: tuple = (1.0, 2.0)
    classes: list[str] = field(default_factory=list)
    note: str = ""


def oof_predict(task: Task, dev: pd.DataFrame, family: str, params: dict | None = None) -> np.ndarray:
    X, groups = dev[task.cats + task.nums], dev[task.unit].to_numpy()
    y = encode_target(task, dev)
    n_cls = len(task.classes)
    out = np.zeros((len(dev), n_cls)) if task.kind == "classification" else np.zeros(len(dev))
    for tr, te in GroupKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, groups=groups):
        assert not set(groups[tr]) & set(groups[te])
        est = make_estimator(family, task.kind, task.cats, task.nums, params, n_cls)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X.iloc[tr], y[tr])
            out[te] = est.predict_proba(X.iloc[te]) if task.kind == "classification" else est.predict(X.iloc[te])
    return out


def encode_target(task: Task, df: pd.DataFrame) -> np.ndarray:
    if task.kind == "regression":
        return df[task.target].to_numpy(float)
    return df[task.target].map({c: i for i, c in enumerate(task.classes)}).to_numpy(int)


def score(task: Task, y, pred) -> dict:
    if task.kind == "regression":
        return regression_scores(y, pred, task.tol)
    return classification_scores(y, pred, len(task.classes))


def primary(task: Task, s: dict) -> float:
    """Lower is better."""
    return s["MAE"] if task.kind == "regression" else -s["macro_F1"]


def run_task(task: Task, df: pd.DataFrame, n_trials: int) -> dict:
    t0 = time.time()
    dev = df[df.split == "dev"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    y_dev = encode_target(task, dev)
    rows, oof = [], {}
    for fam in FAMILIES:
        oof[fam] = oof_predict(task, dev, fam)
        rows.append({"model": fam, **score(task, y_dev, oof[fam])})
        print(f"    {fam:14s} {_fmt(task, rows[-1])}", flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    best_params, best_val = {}, primary(task, rows[-1])
    for i in range(n_trials):
        p = {k: v[rng.integers(len(v))] for k, v in XGB_SPACE.items()}
        p = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in p.items()}
        pred = oof_predict(task, dev, "XGBoost", p)
        s = score(task, y_dev, pred)
        if primary(task, s) < best_val:
            best_val, best_params, oof["XGBoost (tuned)"] = primary(task, s), p, pred
    if best_params:
        rows.append({"model": "XGBoost (tuned)", **score(task, y_dev, oof["XGBoost (tuned)"])})
        print(f"    {'XGBoost tuned':14s} {_fmt(task, rows[-1])}  {best_params}", flush=True)
    cv = pd.DataFrame(rows)
    cv.insert(0, "task", task.key)

    best_name = cv.loc[cv.apply(lambda r: primary(task, r), axis=1).idxmin(), "model"]
    family = "XGBoost" if best_name.startswith("XGBoost") else best_name
    params = best_params if best_name == "XGBoost (tuned)" else None
    final = make_estimator(family, task.kind, task.cats, task.nums, params, len(task.classes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(dev[task.cats + task.nums], y_dev)
    Xte, y_te = test[task.cats + task.nums], encode_target(task, test)
    p_te = final.predict_proba(Xte) if task.kind == "classification" else final.predict(Xte)
    test_scores = score(task, y_te, p_te)
    if task.kind == "regression":
        boot = lambda idx: {"MAE": mean_absolute_error(y_te[idx], p_te[idx]), "R2": r2_score(y_te[idx], p_te[idx])}
    else:
        lab = p_te.argmax(1)
        boot = lambda idx: {"accuracy": accuracy_score(y_te[idx], lab[idx]),
                            "balanced_accuracy": balanced_accuracy_score(y_te[idx], lab[idx]),
                            "macro_F1": f1_score(y_te[idx], lab[idx], average="macro")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # a resample can miss a rare class
        ci = bootstrap_ci(test[task.unit].to_numpy(), boot)
    base = make_estimator("Baseline", task.kind, task.cats, task.nums, None, len(task.classes))
    base.fit(dev[task.cats + task.nums], y_dev)
    base_te = base.predict_proba(Xte) if task.kind == "classification" else base.predict(Xte)
    print(f"    TEST  {best_name}: {_fmt(task, test_scores)}  ({time.time() - t0:.0f}s)", flush=True)

    meta = {"task": task.key, "dataset": task.dataset, "target": task.target, "kind": task.kind,
            "unit": task.unit, "model": best_name, "xgb_params": params,
            "categorical_features": task.cats, "numeric_features": task.nums, "classes": task.classes,
            "n_dev_rows": len(dev), "n_test_rows": len(test), "n_dev_units": int(dev[task.unit].nunique()),
            "n_test_units": int(test[task.unit].nunique()), "cv_scores": _clean(cv.set_index("model").T.to_dict()),
            "test_scores": _clean(test_scores), "test_ci95": ci,
            "test_baseline_scores": _clean(score(task, y_te, base_te)),
            "feature_ranges": feature_ranges(dev, task.nums), "note": task.note}
    joblib.dump({"pipeline": final, "meta": meta}, MODELS_DIR / f"ev_{task.key}.joblib")
    (MODELS_DIR / f"ev_{task.key}.json").write_text(json.dumps(meta, indent=2, default=str))

    pred_frame = test[[task.unit]].copy()
    pred_frame["task"] = task.key
    if task.kind == "regression":
        pred_frame["y_true"], pred_frame["y_pred"] = y_te, p_te
    else:
        pred_frame["y_true"] = [task.classes[i] for i in y_te]
        pred_frame["y_pred"] = [task.classes[i] for i in p_te.argmax(1)]
        for i, c in enumerate(task.classes):
            pred_frame[f"p_{c}"] = p_te[:, i]
    return {"task": task, "cv": cv, "meta": meta, "pred": pred_frame, "test": test, "pipeline": final}


def _fmt(task: Task, s: dict) -> str:
    if task.kind == "regression":
        return f"MAE {s['MAE']:.4g}  RMSE {s['RMSE']:.4g}  R2 {s['R2']:.4f}"
    return (f"acc {s['accuracy']:.4f}  bal-acc {s['balanced_accuracy']:.4f}  macro-F1 {s['macro_F1']:.4f}"
            f"  AUC {s['AUC']:.4f}")


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    return float(d) if isinstance(d, (np.floating, np.integer)) else d


# --------------------------------------------------------------------------- #
# Extra experiments (DEV cross-validation only)
# --------------------------------------------------------------------------- #
def leakage_and_ablation(ev1: pd.DataFrame, ev1_nums: list[str], ev2: pd.DataFrame,
                         ev2_nums: list[str]) -> pd.DataFrame:
    rows = []

    def add(dataset, label, task, dev):
        s = score(task, encode_target(task, dev), oof_predict(task, dev, "XGBoost"))
        rows.append({"dataset": dataset, "variant": label, **s})
        print(f"    {dataset} | {label}: MAE {s['MAE']:.3f}  R2 {s['R2']:.4f}", flush=True)

    d1 = ev1[ev1.split == "dev"].reset_index(drop=True)
    c1 = EV1_CATEGORICAL_FEATURES
    mk = lambda nums, cats: Task("x", "ev_degradation", "state_of_health_pct", "regression", "battery_id", nums, cats)
    add("ev_degradation", "raw columns only", mk(EV1_RAW_NUMERIC, c1), d1)
    add("ev_degradation", "all features (saved model inputs)", mk(ev1_nums, c1), d1)
    add("ev_degradation", "LEAK: + capacity_fade_pct (= 100 - SoH)", mk(ev1_nums + ["leak_capacity_fade_pct"], c1), d1)

    # Cleaning ablation: blank flagged outliers instead of keeping them.
    raw = load_ev_degradation()
    dev_units = set(d1.battery_id)
    masker = make_ev1_masker().fit(clean_ev_degradation(raw)[0].loc[lambda d: d.battery_id.isin(dev_units)])
    masked, _ = clean_ev_degradation(raw, masker, mask_outliers=True)
    masked, _ = ev1_features(masked)
    add("ev_degradation", "all features, flagged outliers set to NaN",
        mk(ev1_nums, c1), masked[masked.battery_id.isin(dev_units)].reset_index(drop=True))

    d2 = ev2[ev2.split == "dev"].reset_index(drop=True)
    d2["leak_implied_soh"] = implied_soh_from_range(d2)
    c2 = EV2_CATEGORICAL_FEATURES
    mk2 = lambda nums: Task("x", "ev_telematics", "soh_pct", "regression", "vehicle_id", nums, c2)
    add("ev_telematics", "raw columns only", mk2(EV2_RAW_NUMERIC), d2)
    add("ev_telematics", "all features (saved model inputs)", mk2(ev2_nums), d2)
    add("ev_telematics", "LEAK: + range_remaining_km", mk2(ev2_nums + ["leak_range_remaining_km"]), d2)
    add("ev_telematics", "LEAK: + SoH implied by range", mk2(ev2_nums + ["leak_implied_soh"]), d2)
    return pd.DataFrame(rows)


HARMONISED = ["capacity_kwh", "cycles", "age_days", "ambient_c", "charge_kw"]


def harmonise(ev1: pd.DataFrame, ev2: pd.DataFrame) -> pd.DataFrame:
    a = pd.DataFrame({"source": "ev_degradation", "unit": ev1.battery_id, "split": ev1.split,
                      "chemistry": ev1.chemistry.where(ev1.chemistry.isin(["LFP", "NMC"]), "other"),
                      "capacity_kwh": ev1.pack_capacity_kwh, "cycles": ev1.cycle_count,
                      "age_days": ev1.battery_age_days, "ambient_c": ev1.avg_ambient_temp_c,
                      "charge_kw": ev1.avg_charging_power_kw, "soh": ev1.state_of_health_pct})
    b = pd.DataFrame({"source": "ev_telematics", "unit": ev2.vehicle_id, "split": ev2.split,
                      "chemistry": ev2.battery_type, "capacity_kwh": ev2.battery_capacity_kwh,
                      "cycles": ev2.battery_cycles, "age_days": ev2.vehicle_age_years * 365.25,
                      "ambient_c": ev2.ambient_temp_c, "charge_kw": ev2.charging_power_kw, "soh": ev2.soh_pct})
    return pd.concat([a, b], ignore_index=True)


def cross_dataset(ev1: pd.DataFrame, ev2: pd.DataFrame) -> pd.DataFrame:
    """SoH on the features both datasets share: train on one, the other, or both."""
    h = harmonise(ev1, ev2)
    rows = []
    for train_on in ["ev_degradation", "ev_telematics", "both (pooled)"]:
        tr = h[(h.split == "dev") & ((h.source == train_on) | train_on.startswith("both"))]
        est = make_estimator("XGBoost", "regression", ["chemistry"], HARMONISED)
        est.fit(tr[["chemistry"] + HARMONISED], tr.soh)
        for test_on in ["ev_degradation", "ev_telematics"]:
            te = h[(h.split == "test") & (h.source == test_on)]
            s = regression_scores(te.soh, est.predict(te[["chemistry"] + HARMONISED]), (1, 2))
            rows.append({"trained_on": train_on, "tested_on": test_on, "n_train": len(tr), **s})
            print(f"    train {train_on:15s} -> test {test_on:15s} MAE {s['MAE']:.3f}  R2 {s['R2']:.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_regression(results: list[dict]) -> None:
    reg = [r for r in results if r["task"].kind == "regression"]
    fig, axes = plt.subplots(1, len(reg), figsize=(3.6 * len(reg), 3.6))
    for ax, r in zip(np.atleast_1d(axes), reg):
        p, m = r["pred"], r["meta"]
        lo, hi = np.percentile(np.r_[p.y_true, p.y_pred], [0.2, 99.8])
        ax.hexbin(p.y_true, p.y_pred, gridsize=45, cmap=SEQ_CMAP, mincnt=1, bins="log",
                  extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], color=INK_2, lw=0.8, ls="--")
        ax.set(xlabel=f"true {r['task'].target}", ylabel="predicted",
               title=f"{r['task'].key}\nTEST MAE {m['test_scores']['MAE']:.3g}, R² {m['test_scores']['R2']:.3f}")
    fig.tight_layout()
    save(fig, FIG / "ev_regression_pred_vs_true.png")


def plot_confusions(results: list[dict]) -> None:
    cls = [r for r in results if r["task"].kind == "classification"]
    fig, axes = plt.subplots(1, len(cls), figsize=(3.4 * len(cls), 3.4))
    for ax, r in zip(np.atleast_1d(axes), cls):
        t, p = r["task"], r["pred"]
        cm = confusion_matrix(p.y_true, p.y_pred, labels=t.classes, normalize="true")
        ax.imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(len(t.classes)):
            for j in range(len(t.classes)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(t.classes)), t.classes, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(t.classes)), t.classes, fontsize=7)
        s = r["meta"]["test_scores"]
        ax.set(xlabel="predicted", ylabel="true",
               title=f"{t.key}\nacc {s['accuracy']:.3f} · bal {s['balanced_accuracy']:.3f}")
        ax.grid(False)
    fig.tight_layout()
    save(fig, FIG / "ev_confusion_matrices.png")


def plot_importance(results: list[dict]) -> pd.DataFrame:
    rows = []
    soh = [r for r in results if r["task"].key in ("degradation_soh", "telematics_soh")]
    fig, axes = plt.subplots(1, len(soh), figsize=(4.6 * len(soh), 4.2))
    for ax, r in zip(np.atleast_1d(axes), soh):
        model = r["pipeline"].named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            continue
        imp = pd.Series(model.feature_importances_, index=r["task"].cats + r["task"].nums)
        imp = imp.sort_values(ascending=False)
        rows += [{"task": r["task"].key, "feature": f, "gain_share": v} for f, v in imp.items()]
        top = imp.head(12)[::-1]
        ax.barh(top.index, top.values, color=CATEGORICAL[0])
        ax.set(title=f"{r['task'].key}: top features (XGBoost gain)", xlabel="share of gain")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save(fig, FIG / "ev_feature_importance.png")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def build_tasks(ev1_nums: list[str], ev2_nums: list[str]) -> list[Task]:
    c1, c2 = EV1_CATEGORICAL_FEATURES, EV2_CATEGORICAL_FEATURES
    return [
        Task("degradation_soh", "ev_degradation", "state_of_health_pct", "regression", "battery_id", ev1_nums, c1,
             "SoH %-points", (1.0, 2.0)),
        Task("degradation_rul", "ev_degradation", "remaining_useful_life_cycles", "regression", "battery_id",
             ev1_nums, c1, "cycles", (100.0, 250.0)),
        Task("degradation_risk", "ev_degradation", "failure_risk_label", "classification", "battery_id", ev1_nums,
             c1, classes=["Low", "Medium", "High"]),
        Task("degradation_category", "ev_degradation", "degradation_category", "classification", "battery_id",
             ev1_nums, c1, classes=["Normal", "Accelerated", "Critical"]),
        Task("telematics_soh", "ev_telematics", "soh_pct", "regression", "vehicle_id", ev2_nums, c2,
             "SoH %-points", (1.0, 2.0), note="SoH is an integer in the data: rounding alone costs ~0.25 MAE"),
        Task("telematics_failure_type", "ev_telematics", "failure_type", "classification", "vehicle_id", ev2_nums,
             c2, classes=["Normal", "Voltage Drop", "Cell Imbalance", "Overheating"],
             note="labels are drawn at random from a risk score: use probabilities, not the class"),
        Task("telematics_maintenance", "ev_telematics", "maintenance_required", "classification", "vehicle_id",
             ev2_nums, c2, classes=["No", "Yes"], note="random draw from a risk score; near the majority baseline"),
        Task("telematics_fire_risk", "ev_telematics", "fire_risk_2", "classification", "vehicle_id", ev2_nums, c2,
             classes=["Low", "Medium/High"], note="'High' (3 rows) merged into 'Medium/High'"),
        Task("telematics_consumption", "ev_telematics", "wh_per_km", "regression", "vehicle_id",
             EV2_PRETRIP_NUMERIC, EV2_PRETRIP_CATEGORICAL, "Wh/km", (10.0, 20.0),
             note="pre-trip inputs only; feeds RangeAssumptions.consumption_wh_per_km"),
    ]


# Inputs a person can type in (dashboard "manual entry"): age, and equivalent full cycles
# from odometer x consumption / capacity. Chemistry and climate are optional.
QUICK_NUMERIC = ["battery_age_days", "efc", "efc_per_day"]
QUICK_CATEGORICAL = ["chemistry", "climate_zone"]


def build_quick_tasks() -> list[Task]:
    note = ("few inputs for manual entry; trained with chemistry/climate also blanked on a copy "
            "of every training row, so 'unknown' is a valid input")
    return [Task("quick_soh", "ev_degradation", "state_of_health_pct", "regression", "battery_id",
                 QUICK_NUMERIC, QUICK_CATEGORICAL, "SoH %-points", (1.0, 2.0), note=note),
            Task("quick_rul", "ev_degradation", "remaining_useful_life_cycles", "regression", "battery_id",
                 QUICK_NUMERIC, QUICK_CATEGORICAL, "cycles", (100.0, 250.0), note=note)]


def with_unknown_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Training rows plus a copy with the optional categories blanked (dev rows only).
    The copy keeps its battery id, so cross-validation still holds out whole batteries."""
    dev = df[df.split == "dev"]
    blank = dev.copy()
    blank[QUICK_CATEGORICAL] = np.nan
    return pd.concat([dev, blank, df[df.split == "test"]], ignore_index=True)


def run_quick(ev1: pd.DataFrame, n_trials: int = 6) -> list[dict]:
    """Quick (manual-entry) models; TEST is scored with categories known and unknown."""
    out = []
    data = with_unknown_categories(ev1)
    for task in build_quick_tasks():
        print(f"[ev] {task.key}  ({task.target}, manual-entry inputs)", flush=True)
        r = run_task(task, data, n_trials)
        test = ev1[ev1.split == "test"].copy()
        test[QUICK_CATEGORICAL] = np.nan
        p = r["pipeline"].predict(test[task.cats + task.nums])
        unknown = regression_scores(test[task.target], p, task.tol)
        r["meta"]["test_scores_categories_unknown"] = _clean(unknown)
        joblib.dump({"pipeline": r["pipeline"], "meta": r["meta"]}, MODELS_DIR / f"ev_{task.key}.joblib")
        (MODELS_DIR / f"ev_{task.key}.json").write_text(json.dumps(r["meta"], indent=2, default=str))
        print(f"    TEST with chemistry/climate unknown: {_fmt(task, unknown)}", flush=True)
        out.append(r)
    return out


def run(n_trials: int = 12) -> None:
    apply_style()
    for d in (EV_DIR, RESULTS, FIG, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print("[ev] preparing ev_degradation ...", flush=True)
    ev1, ev1_nums, rep1, masker = prepare_ev_degradation()
    print("[ev] preparing ev_telematics ...", flush=True)
    ev2, ev2_nums, rep2 = prepare_ev_telematics()
    ev1.to_parquet(EV_DIR / "ev_degradation_features.parquet", index=False)
    ev2.to_parquet(EV_DIR / "ev_telematics_features.parquet", index=False)
    (RESULTS / "ev_cleaning_report.json").write_text(json.dumps(
        {"ev_degradation": rep1, "ev_telematics": rep2, "ev_degradation_outlier_masker": masker},
        indent=2, default=str))
    (MODELS_DIR / "ev_outlier_masker.json").write_text(json.dumps(masker, indent=2))   # used at inference
    pd.concat([ev1.groupby("battery_id").split.first().rename("split").reset_index().assign(dataset="ev_degradation")
               .rename(columns={"battery_id": "unit"}),
               ev2.groupby("vehicle_id").split.first().rename("split").reset_index().assign(dataset="ev_telematics")
               .rename(columns={"vehicle_id": "unit"})]).to_csv(RESULTS / "ev_split_units.csv", index=False)

    results = []
    for task in build_tasks(ev1_nums, ev2_nums):
        df = ev1 if task.dataset == "ev_degradation" else ev2
        print(f"[ev] {task.key}  ({task.target}, {task.kind})", flush=True)
        results.append(run_task(task, df, n_trials if task.kind == "regression" else max(n_trials // 2, 1)))
    results += run_quick(ev1, max(n_trials // 2, 1))

    pd.concat([r["cv"] for r in results]).to_csv(RESULTS / "ev_cv_model_comparison.csv", index=False)
    for kind in ("regression", "classification"):
        pd.concat([r["pred"] for r in results if r["task"].kind == kind]).to_parquet(
            RESULTS / f"ev_test_predictions_{kind}.parquet", index=False)
    summary = []
    for r in results:
        m = r["meta"]
        row = {"task": m["task"], "target": m["target"], "model": m["model"],
               **{f"test_{k}": v for k, v in m["test_scores"].items()},
               **{f"test_{k}": v for k, v in m["test_ci95"].items()},
               **{f"baseline_{k}": v for k, v in m["test_baseline_scores"].items()}}
        summary.append(row)
    pd.DataFrame(summary).to_csv(RESULTS / "ev_test_metrics.csv", index=False)

    # Test error per chemistry / SoH band for the SoH models.
    per_group = []
    for r in results:
        if r["task"].kind != "regression" or r["task"].key == "telematics_consumption":
            continue
        p, t = r["pred"].copy(), r["test"]
        p["chemistry"] = (t.chemistry if r["task"].dataset == "ev_degradation" else t.battery_type).to_numpy()
        p["soh_band"] = pd.cut(t["state_of_health_pct" if r["task"].dataset == "ev_degradation" else "soh_pct"],
                               [0, 80, 85, 90, 95, 100.01], right=False).astype(str).to_numpy()
        for col in ["chemistry", "soh_band"]:
            for k, g in p.groupby(col):
                per_group.append({"task": r["task"].key, "group_by": col, "group": k, "n": len(g),
                                  **regression_scores(g.y_true, g.y_pred, r["task"].tol)})
    pd.DataFrame(per_group).to_csv(RESULTS / "ev_test_metrics_by_group.csv", index=False)

    print("[ev] leakage checks and ablations (DEV CV) ...", flush=True)
    leakage_and_ablation(ev1, ev1_nums, ev2, ev2_nums).to_csv(RESULTS / "ev_leakage_and_ablation.csv", index=False)
    print("[ev] cross-dataset experiment ...", flush=True)
    cross_dataset(ev1, ev2).to_csv(RESULTS / "ev_cross_dataset.csv", index=False)

    plot_regression(results)
    plot_confusions(results)
    plot_importance(results).to_csv(RESULTS / "ev_feature_importance.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 tuning trials instead of 12")
    ap.add_argument("--only-manual-entry", action="store_true",
                    help="only the manual-entry models, from the saved feature table")
    args = ap.parse_args()
    if args.only_manual_entry:
        apply_style()
        run_quick(pd.read_parquet(EV_DIR / "ev_degradation_features.parquet"))
    else:
        run(n_trials=2 if args.quick else 12)
