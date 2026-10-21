import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.censored_rul import BuckleyJamesRegressor


def _data(seed=0):
    rng = np.random.default_rng(seed)
    cells = np.repeat([f"C{i}" for i in range(8)], 20)
    x = rng.uniform(0, 1, len(cells))
    y = 100 * x + rng.normal(0, 5, len(cells))
    cens = np.isin(cells, ["C6", "C7"])
    bound = np.where(cens, y - 20, y)  # censored rows only know RUL > bound
    return pd.DataFrame({"x": x}), y, bound, cens, cells


def test_conditional_mean_imputation_is_above_the_bound():
    pred = np.array([10.0, 50.0, 90.0])
    bound = np.array([40.0, 40.0, 40.0])
    res = np.array([-20.0, -5.0, 0.0, 5.0, 20.0])
    imp = BuckleyJamesRegressor.impute(pred, bound, res)
    assert (imp >= bound).all()
    assert imp[0] == 40.0                         # no residual reaches the bound -> the bound itself
    assert np.isclose(imp[1], np.mean([45, 50, 55, 70]))
    assert np.isclose(imp[2], 90.0)               # far above the bound -> plain mean of p + r


def test_censored_targets_never_imputed_below_their_bound():
    X, _, bound, cens, cells = _data()
    for imp in ("bound", "conditional_mean"):
        m = BuckleyJamesRegressor(LinearRegression(), imputation=imp).fit(X, bound, cens, cells)
        assert (m.imputed_ >= bound[cens] - 1e-9).all()


def test_without_censored_rows_it_equals_the_plain_model():
    X, y, _, _, cells = _data()
    none = np.zeros(len(y), bool)
    m = BuckleyJamesRegressor(LinearRegression()).fit(X, y, none, cells)
    ref = LinearRegression().fit(X, y)
    np.testing.assert_allclose(m.predict(X), ref.predict(X), atol=1e-8)


def test_fit_is_reproducible_with_a_parallel_forest():
    X, _, bound, cens, cells = _data(1)
    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, n_jobs=-1, random_state=0)
    a = BuckleyJamesRegressor(rf).fit(X, bound, cens, cells).predict(X)
    b = BuckleyJamesRegressor(rf).fit(X, bound, cens, cells).predict(X)
    np.testing.assert_allclose(a, b, atol=1e-6)
