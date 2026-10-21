"""Validation splits that respect cells and time.

* leave_cells_out    - every fold holds out whole cells (tests generalisation to
                       batteries the model has never seen).
* leave_groups_out   - holds out a whole experiment group (harder: unseen conditions).
* chronological      - per cell, the first `train_frac` of its life trains and the
                       rest tests (tests extrapolation forward in time).

Every splitter yields (fold_name, train_index, test_index) and is checked by
`check_split`, which fails loudly on any overlap or time-order violation.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def leave_cells_out(df: pd.DataFrame, cell_col: str = "cell") -> Iterator[tuple]:
    for cell in sorted(df[cell_col].unique()):
        test = df.index[df[cell_col] == cell]
        train = df.index[df[cell_col] != cell]
        yield cell, train, test


def leave_groups_out(df: pd.DataFrame, group_col: str) -> Iterator[tuple]:
    for grp in sorted(df[group_col].unique()):
        yield grp, df.index[df[group_col] != grp], df.index[df[group_col] == grp]


def chronological(df: pd.DataFrame, order_col: str, train_frac: float = 0.6,
                  cell_col: str = "cell") -> Iterator[tuple]:
    train, test = [], []
    for _, g in df.groupby(cell_col):
        g = g.sort_values(order_col)
        k = int(np.floor(len(g) * train_frac))
        train.append(g.index[:k])
        test.append(g.index[k:])
    yield "chronological", np.concatenate(train), np.concatenate(test)


def check_split(df: pd.DataFrame, train, test, kind: str, cell_col: str = "cell",
                order_col: str | None = None) -> None:
    """Leakage guard: no shared rows; no shared cells for cell-wise splits; for the
    chronological split every training row precedes every test row of the same cell."""
    assert len(set(train) & set(test)) == 0, "train and test share rows"
    if kind in ("cells", "groups"):
        shared = set(df.loc[train, cell_col]) & set(df.loc[test, cell_col])
        assert not shared, f"cells in both train and test: {shared}"
    if kind == "chronological" and order_col:
        tr = df.loc[train].groupby(cell_col)[order_col].max()
        te = df.loc[test].groupby(cell_col)[order_col].min()
        both = tr.index.intersection(te.index)
        assert (tr[both] < te[both]).all(), "a training row is later than a test row"
