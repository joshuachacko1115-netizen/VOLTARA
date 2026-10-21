"""CNN-LSTM regressor over a trailing window of per-cycle features (optional, needs PyTorch).

Input for each row: the features of the last `window` valid cycles/checks of the
same cell, ending at the current one (padded by repeating the first row). A 1-D
convolution extracts local patterns across the window, an LSTM summarises it, and
a linear head outputs the target. Only past rows are used, so it is causal.

The class follows the sklearn fit/predict interface so it can be scored with the
same folds as the tree models. Scaling statistics come from the training rows
only, and early stopping uses cells held out *from the training fold*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def make_windows(df: pd.DataFrame, features: list[str], order_col: str, window: int) -> np.ndarray:
    """Array (n_rows, window, n_features) aligned with df.index."""
    X = np.zeros((len(df), window, len(features)), dtype=np.float32)
    pos = {idx: k for k, idx in enumerate(df.index)}
    for _, g in df.groupby("cell", sort=False):
        g = g.sort_values(order_col)
        vals = g[features].to_numpy(np.float32)
        for j, idx in enumerate(g.index):
            lo = max(0, j - window + 1)
            w = vals[lo:j + 1]
            if len(w) < window:
                w = np.vstack([np.repeat(w[:1], window - len(w), axis=0), w])
            X[pos[idx]] = w
    return X


if HAVE_TORCH:
    class _Net(nn.Module):
        def __init__(self, n_features: int, hidden: int = 32):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(n_features, hidden, kernel_size=3, padding=1), nn.ReLU())
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))

        def forward(self, x):                      # x: (batch, window, features)
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            _, (hn, _) = self.lstm(h)
            return self.head(hn[-1]).squeeze(-1)


class CNNLSTMRegressor:
    def __init__(self, features, order_col, window=10, epochs=300, patience=30, lr=1e-3,
                 n_val_cells=1, seed=42):
        if not HAVE_TORCH:
            raise ImportError("PyTorch is not installed")
        self.features, self.order_col, self.window = features, order_col, window
        self.epochs, self.patience, self.lr = epochs, patience, lr
        self.n_val_cells, self.seed = n_val_cells, seed

    def _prep(self, df):
        X = (df[self.features].to_numpy(float) - self.mu_) / self.sd_
        X = np.nan_to_num(X, nan=0.0)  # 0 = training mean after scaling
        frame = pd.DataFrame(X, index=df.index, columns=self.features)
        frame["cell"], frame[self.order_col] = df.cell.values, df[self.order_col].values
        return torch.tensor(make_windows(frame, self.features, self.order_col, self.window))

    def fit(self, df: pd.DataFrame, y: pd.Series):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu_ = np.nanmean(df[self.features].to_numpy(float), axis=0)
        self.sd_ = np.nanstd(df[self.features].to_numpy(float), axis=0)
        self.mu_, self.sd_ = np.nan_to_num(self.mu_), np.where(np.nan_to_num(self.sd_) > 0, self.sd_, 1.0)
        self.y_scale_ = float(np.nanmax(np.abs(y))) or 1.0

        cells = df.cell.unique()
        val_cells = rng.choice(cells, size=min(self.n_val_cells, len(cells) - 1), replace=False)
        is_val = df.cell.isin(val_cells).to_numpy()
        X = self._prep(df)
        yt = torch.tensor(y.to_numpy(np.float32) / self.y_scale_)
        Xtr, ytr, Xva, yva = X[~is_val], yt[~is_val], X[is_val], yt[is_val]

        self.net_ = _Net(len(self.features))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(Xtr), 64):
                i = perm[b:b + 64]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xtr[i]), ytr[i])
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = loss_fn(self.net_(Xva), yva).item()
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            return self.net_(self._prep(df)).numpy() * self.y_scale_
