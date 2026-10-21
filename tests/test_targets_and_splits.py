import numpy as np
import pandas as pd
import pytest

from src.evaluation.splits import check_split, chronological, leave_cells_out, leave_groups_out
from src.features.targets import add_rul, eol_position
from src.evaluation.metrics import regression_metrics


def test_eol_position_and_interpolation():
    cyc = np.arange(1, 11.0)
    soh = np.linspace(1.0, 0.7, 10)
    assert eol_position(cyc, soh, window=1) == cyc[np.argmax(soh <= 0.8)]
    interp = eol_position(cyc, soh, window=1, interpolate=True)
    assert interp == pytest.approx(7.0)   # SoH is exactly 0.8 at the 7th cycle


def test_eol_ignores_single_noisy_dip():
    soh = np.r_[np.linspace(1, 0.9, 20), 0.5, np.linspace(0.9, 0.85, 20)]
    assert np.isnan(eol_position(np.arange(len(soh), dtype=float), soh, window=5))


def test_rul_censoring_and_post_eol():
    df = pd.DataFrame({"cell": ["a"] * 4 + ["b"] * 3, "cycle": [1, 2, 3, 4, 1, 2, 3],
                       "eol": [3.0] * 4 + [np.nan] * 3})
    out = add_rul(df, "cycle", "eol", "rul")
    assert out.rul[:3].tolist() == [2, 1, 0] and np.isnan(out.rul[3]) and out.post_eol[3]
    assert out.rul[4:].isna().all() and out.censored[4:].all()
    assert out.rul_lower_bound[4:].tolist() == [2, 1, 0]


@pytest.fixture
def frame():
    return pd.DataFrame({"cell": list("aaabbbccc"), "grp": list("xxxxxxyyy"),
                         "t": [1, 2, 3] * 3, "y": range(9)})


def test_leave_cells_out_has_no_cell_overlap(frame):
    for _, tr, te in leave_cells_out(frame):
        check_split(frame, tr, te, "cells")


def test_leave_groups_out_has_no_group_overlap(frame):
    for _, tr, te in leave_groups_out(frame, "grp"):
        assert set(frame.loc[tr, "grp"]).isdisjoint(frame.loc[te, "grp"])


def test_chronological_split_is_time_ordered(frame):
    (_, tr, te), = chronological(frame, "t", 0.67)
    check_split(frame, tr, te, "chronological", order_col="t")


def test_check_split_catches_leakage(frame):
    with pytest.raises(AssertionError):
        check_split(frame, frame.index[:5], frame.index[4:], "cells")


def test_metrics():
    m = regression_metrics([1, 2, 3], [1, 2, 4])
    assert m["MAE"] == pytest.approx(1 / 3) and m["RMSE"] == pytest.approx(np.sqrt(1 / 3))
