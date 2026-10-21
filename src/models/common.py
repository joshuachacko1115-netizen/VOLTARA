"""Model zoo and the cross-validation loop shared by the SoH and RUL experiments.

Every model is wrapped in a Pipeline(imputer -> scaler -> model) and the whole
pipeline is re-fitted inside each fold, so the median used for imputation and the
mean/std used for scaling come from the training cells only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED
from src.evaluation.splits import check_split

try:  # XGBoost needs the OpenMP runtime (bundled on Windows; `brew install libomp` on macOS).
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:  # pragma: no cover - depends on the machine
    HAVE_XGB = False


def make_models() -> dict:
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               n_jobs=-1, random_state=RANDOM_SEED),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                           max_leaf_nodes=15, min_samples_leaf=10,
                                                           random_state=RANDOM_SEED),
    }
    if HAVE_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=RANDOM_SEED, n_jobs=-1)
    return models


def make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def cross_validate(df: pd.DataFrame, features: list[str], target: str, splits, split_kind: str,
                   models: dict, order_col: str | None = None, extra_cols=()) -> pd.DataFrame:
    """Fit every model on every fold; return one row per held-out prediction."""
    out = []
    for fold, train, test in splits:
        check_split(df, train, test, split_kind, order_col=order_col)
        Xtr, ytr = df.loc[train, features], df.loc[train, target]
        Xte = df.loc[test, features]
        for name, model in models.items():
            pipe = make_pipeline(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
            frame = df.loc[test, ["cell", *extra_cols]].copy()
            frame["fold"] = fold
            frame["model"] = name
            frame["y_true"] = df.loc[test, target].to_numpy()
            frame["y_pred"] = pred
            out.append(frame)
    return pd.concat(out, ignore_index=True)


def feature_importance(df: pd.DataFrame, features: list[str], target: str) -> pd.Series:
    """Impurity-based importance of a Random Forest fitted on all rows (descriptive only)."""
    pipe = make_pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                               random_state=RANDOM_SEED))
    pipe.fit(df[features], df[target])
    return pd.Series(pipe.named_steps["model"].feature_importances_, index=features).sort_values(
        ascending=False)


def feature_ranges(df: pd.DataFrame, features: list[str]) -> dict:
    """1st / 99th percentile of each training feature, saved so inference can warn on extrapolation."""
    q = df[features].quantile([0.01, 0.99])
    return {f: [None if pd.isna(q.at[0.01, f]) else float(q.at[0.01, f]),
                None if pd.isna(q.at[0.99, f]) else float(q.at[0.99, f])] for f in features}
