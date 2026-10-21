"""Censoring-aware RUL training: let cells that never failed contribute what they do show.

Cells that never reached EOL (B0029, B0031, B0032, RW26-RW28) have no RUL label, only
a lower bound: at each row, RUL > (last observed ageing - current ageing). The
standard models in train_rul.py drop them. Because the cells that fade slowest are
the ones most likely to be censored, dropping them biases RUL downwards (RW27).

Method: Buckley-James-style imputation (Buckley & James, 1979), with any sklearn
regressor.
  1. Fit the model on the labelled rows. Estimate its error distribution from
     cell-grouped cross-validated residuals on those rows (in-sample residuals of a
     forest are far too small).
  2. For a censored row with prediction p and bound b, impute
         E[RUL | RUL > b] ~= mean of (p + r) over residuals r with p + r > b,
     or b if no residual gets above the bound.
  3. Refit on labelled + imputed rows; repeat 2-3 a few times.
The imputed values are never scored. They only enter training.

Evaluation, all with the same held-out rows as the standard models:
  * labelled cells, leave-one-cell-out: MAE / RMSE / R2. Censored cells are added
    to training only.
  * censored cells, leave-one-cell-out: each censored cell is held out in turn. We
    report how often the prediction respects the known bound and the mean
    shortfall below it (0 = never below).

Usage:
    python -m src.models.censored_rul
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GroupKFold

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.evaluation.metrics import summarise_predictions
from src.evaluation.splits import check_split, leave_cells_out
from src.models.common import feature_ranges, make_models, make_pipeline
from src.models.train_rul import DATASETS, load
from src.visualization.style import CATEGORICAL, INK_2, MUTED, apply_style, save

RESULTS = REPORTS_DIR / "results"
# (label, imputation) for the variants compared; None = labelled rows only (train_rul.py).
VARIANTS = (("labelled only", None),
            ("+ censored, bound", "bound"),
            ("+ censored, Buckley-James", "conditional_mean"))


class BuckleyJamesRegressor(BaseEstimator, RegressorMixin):
    """Regressor trained on exact targets plus right-censored lower bounds.

    fit(X, y, censored, groups): for censored rows, y is the lower bound. `groups`
    (cell ids) keeps the residual cross-validation cell-grouped.
    """

    def __init__(self, estimator=None, n_iter: int = 3, n_splits: int = 5, imputation: str = "conditional_mean"):
        self.estimator = estimator
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.imputation = imputation  # "conditional_mean" (Buckley-James) or "bound" (max(pred, bound))

    def _residuals(self, X, y, groups) -> np.ndarray:
        n = min(self.n_splits, len(np.unique(groups)))
        res = []
        for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
            m = make_pipeline(clone(self.estimator)).fit(X.iloc[tr], y[tr])
            res.append(y[te] - m.predict(X.iloc[te]))
        return np.concatenate(res)

    @staticmethod
    def impute(pred: np.ndarray, bound: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        cand = pred[:, None] + residuals[None, :]
        above = cand > bound[:, None]
        n = above.sum(axis=1)
        mean_above = np.where(above, cand, 0.0).sum(axis=1) / np.maximum(n, 1)
        return np.where(n > 0, mean_above, bound)

    def fit(self, X: pd.DataFrame, y, censored, groups):
        y, cens = np.asarray(y, float), np.asarray(censored, bool)
        groups = np.asarray(groups)
        Xl, yl = X[~cens], y[~cens]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.residuals_ = np.round(self._residuals(Xl, yl, groups[~cens]), 6)
            self.model_ = make_pipeline(clone(self.estimator)).fit(Xl, yl)
            y_fit = y.copy()
            for _ in range(self.n_iter if cens.any() else 0):
                # Round: parallel forests differ in the last float bits between runs, and fed back as
                # targets those bits can flip tied splits, so un-rounded imputation is not reproducible.
                pred = np.round(self.model_.predict(X[cens]), 6)
                y_fit[cens] = (np.maximum(pred, y[cens]) if self.imputation == "bound"
                               else self.impute(pred, y[cens], self.residuals_))
                self.model_ = make_pipeline(clone(self.estimator)).fit(X, y_fit)
        self.imputed_ = y_fit[cens]
        return self

    def predict(self, X):
        return self.model_.predict(X)


def _combined(cfg: dict) -> pd.DataFrame:
    """Labelled rows (exact RUL) and censored rows (lower bound) in one frame."""
    lab, cens = load(cfg)
    lab = lab.assign(is_censored=False, y=lab[cfg["target"]])
    cens = cens.assign(is_censored=True, y=cens.rul_lower_bound)
    return pd.concat([lab, cens], ignore_index=True)


def evaluate(name: str, cfg: dict, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _combined(cfg)
    feats = cfg["features"]
    est = make_models()[model_name]
    preds, bounds = [], []
    # Every cell, labelled or censored, is held out once.
    for fold, train, test in leave_cells_out(df):
        check_split(df, train, test, "cells")
        tr, te = df.loc[train], df.loc[test]
        lab_tr = tr[~tr.is_censored]
        for variant, imputation in VARIANTS:
            if imputation is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = make_pipeline(clone(est)).fit(lab_tr[feats], lab_tr.y)
            else:
                model = BuckleyJamesRegressor(est, imputation=imputation).fit(
                    tr[feats], tr.y, tr.is_censored, tr.cell)
            frame = te[["cell", cfg["order"], "is_censored"]].assign(
                fold=fold, model=f"{model_name}, {variant}", y_true=te.y.to_numpy(),
                y_pred=np.clip(model.predict(te[feats]), 0, None))
            (bounds if te.is_censored.iloc[0] else preds).append(frame)
    preds, bounds = pd.concat(preds, ignore_index=True), pd.concat(bounds, ignore_index=True)
    preds["dataset"], bounds["dataset"] = name, name
    return preds, bounds


def bound_summary(b: pd.DataFrame) -> pd.DataFrame:
    b = b.assign(respects=b.y_pred >= b.y_true, shortfall=(b.y_true - b.y_pred).clip(lower=0))
    return b.groupby(["dataset", "model", "cell"]).agg(
        rows=("respects", "size"), frac_pred_above_lower_bound=("respects", "mean"),
        mean_shortfall=("shortfall", "mean")).reset_index()


def fit_final(name: str, cfg: dict, model_name: str, scores: dict) -> None:
    """Optional artifact for inference; the default RUL model stays the labelled-only one."""
    df = _combined(cfg)
    bj = BuckleyJamesRegressor(make_models()[model_name]).fit(
        df[cfg["features"]], df.y, df.is_censored, df.cell)
    joblib.dump(bj, MODELS_DIR / f"rul_{cfg['tag']}_censoring_aware.joblib")
    (MODELS_DIR / f"rul_{cfg['tag']}_censoring_aware.json").write_text(json.dumps({
        "dataset": name, "model": f"{model_name}, + censored, Buckley-James", "target": cfg["target"],
        "unit": cfg["unit"], "features": cfg["features"],
        "censored_cells": sorted(df[df.is_censored].cell.unique()),
        "feature_ranges": feature_ranges(df, cfg["features"])} | scores, indent=2))


def plot(bounds: pd.DataFrame, name: str, cfg: dict, model_name: str) -> None:
    cells = sorted(bounds[bounds.dataset == name].cell.unique())
    fig, axes = plt.subplots(1, len(cells), figsize=(3.2 * len(cells), 2.9), sharey=False)
    for ax, cell in zip(np.atleast_1d(axes), cells):
        d = bounds[(bounds.dataset == name) & (bounds.cell == cell)]
        lb = d[d.model == f"{model_name}, {VARIANTS[0][0]}"].sort_values(cfg["order"])
        ax.plot(lb[cfg["order"]], lb.y_true, color=INK_2, lw=1, ls="--", label="known lower bound")
        for k, (variant, _) in enumerate(VARIANTS):
            s = d[d.model == f"{model_name}, {variant}"].sort_values(cfg["order"])
            ax.plot(s[cfg["order"]], s.y_pred, color=CATEGORICAL[k], lw=1.5, marker="o", ms=2.5,
                    label=variant)
        lo, hi = ax.get_ylim()  # allowed region: everything above the bound, to the top of the axes
        ax.fill_between(lb[cfg["order"]], lb.y_true, hi, color=MUTED, alpha=0.15, lw=0,
                        label="allowed (RUL > bound)", zorder=0)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{cell} (held out, never failed)", fontsize=9)
        ax.set_xlabel("cycle" if cfg["tag"] == "ba" else "EFC")
    np.atleast_1d(axes)[0].set_ylabel(f"RUL ({cfg['unit']})")
    np.atleast_1d(axes)[0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle(f"{name}: predicted RUL vs the known lower bound, {model_name}",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "models" / f"rul_{cfg['tag']}_censored_cells.png")


def run() -> None:
    apply_style()
    all_p, all_b, chosen = [], [], {}
    for name in ("Battery Aging", "Randomized Usage"):
        cfg = DATASETS[name]
        # Same model type as the saved labelled-only RUL model (train_rul.py runs first).
        chosen[name] = json.loads((MODELS_DIR / f"rul_{cfg['tag']}.json").read_text())["model"]
        p, b = evaluate(name, cfg, chosen[name])
        all_p.append(p)
        all_b.append(b)
        print(f"  {name} ({chosen[name]}) done")
    preds, bounds = pd.concat(all_p, ignore_index=True), pd.concat(all_b, ignore_index=True)
    metrics = summarise_predictions(preds, ["dataset", "model"])
    btot = bounds.assign(respects=bounds.y_pred >= bounds.y_true,
                         shortfall=(bounds.y_true - bounds.y_pred).clip(lower=0)).groupby(
        ["dataset", "model"]).agg(censored_frac_above_bound=("respects", "mean"),
                                  censored_mean_shortfall=("shortfall", "mean")).reset_index()
    metrics = metrics.merge(btot, on=["dataset", "model"])
    bsum = bound_summary(bounds)
    metrics.to_csv(RESULTS / "rul_censoring_metrics.csv", index=False)
    bsum.to_csv(RESULTS / "rul_censoring_bounds_per_cell.csv", index=False)
    preds.to_parquet(RESULTS / "rul_censoring_predictions.parquet", index=False)

    for name, model_name in chosen.items():
        cfg = DATASETS[name]
        plot(bounds, name, cfg, model_name)
        row = metrics[(metrics.dataset == name)
                      & (metrics.model == f"{model_name}, {VARIANTS[2][0]}")].iloc[0]
        fit_final(name, cfg, model_name, {
            "loco_MAE": row.MAE, "loco_RMSE": row.RMSE, "loco_R2": row.R2,
            "censored_frac_above_bound": row.censored_frac_above_bound,
            "censored_mean_shortfall": row.censored_mean_shortfall})

    pd.set_option("display.width", 220)
    print(metrics[["dataset", "model", "n", "MAE", "RMSE", "R2", "MAE_fold_std",
                   "censored_frac_above_bound", "censored_mean_shortfall"]].round(3).to_string(index=False))
    print(bsum.round(3).to_string(index=False))


if __name__ == "__main__":
    # Import by package path so the saved model pickles as src.models.censored_rul.BuckleyJamesRegressor,
    # not __main__.BuckleyJamesRegressor (which src/inference could not unpickle).
    from src.models import censored_rul
    censored_rul.run()
