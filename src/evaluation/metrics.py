"""Regression metrics and per-group summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    return {
        "n": int(ok.sum()),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def summarise_predictions(pred: pd.DataFrame, by: list[str], fold_col: str = "fold") -> pd.DataFrame:
    """Pooled metrics over all held-out predictions, plus the spread of per-fold MAE."""
    rows = []
    for key, g in pred.groupby(by, sort=False):
        m = regression_metrics(g.y_true, g.y_pred)
        fold_mae = g.groupby(fold_col).apply(
            lambda f: np.mean(np.abs(f.y_true - f.y_pred)), include_groups=False)
        m.update({"MAE_fold_mean": float(fold_mae.mean()), "MAE_fold_std": float(fold_mae.std(ddof=0))})
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,))) | m)
    return pd.DataFrame(rows)
