"""Prediction-only, file-backed storage for a fitted single-output random forest.

Preserves every learned split and leaf value. Trees are mapped individually so
large forests do not need to be copied into process memory during loading.
"""
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MappedForestRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, directory, n_estimators, n_features):
        self.directory = directory
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.n_features_in_ = n_features

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        raise RuntimeError('This is an inference-only export; train the original forest instead.')

    def predict(self, X):
        from src.config import MODELS_DIR
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError('Input feature count does not match the trained forest.')
        if np.isinf(values).any():
            raise ValueError('Input contains infinity.')
        result = np.zeros(len(values), dtype=np.float64)
        row = np.arange(len(values))
        for index in range(self.n_estimators):
            tree = np.load(Path(MODELS_DIR) / self.directory / f'tree_{index:04d}.npy',
                           mmap_mode='r', allow_pickle=False)
            try:
                nodes = np.zeros(len(values), dtype=np.int32)
                active = tree['left'][nodes] != -1
                while active.any():
                    selected = row[active]
                    current = nodes[active]
                    feature = values[selected, tree['feature'][current]]
                    go_left = np.where(np.isnan(feature), tree['missing_left'][current].astype(bool),
                                       feature <= tree['threshold'][current])
                    nodes[active] = np.where(go_left, tree['left'][current], tree['right'][current])
                    active = tree['left'][nodes] != -1
                result += tree['value'][nodes]
            finally:
                tree._mmap.close()
        return result / self.n_estimators


def export_forest(forest, directory):
    """Export a trusted fitted regressor's exact learned nodes, without retraining."""
    directory = Path(directory)
    if forest.n_outputs_ != 1:
        raise ValueError('Only single-output regression forests are supported.')
    directory.mkdir(parents=True, exist_ok=True)
    dtype = [('left', '<i4'), ('right', '<i4'), ('feature', '<i4'),
             ('threshold', '<f8'), ('value', '<f8'), ('missing_left', 'u1')]
    for index, estimator in enumerate(forest.estimators_):
        tree = estimator.tree_
        if tree.node_count >= 2**31:
            raise ValueError('Tree exceeds supported node count.')
        nodes = np.empty(tree.node_count, dtype=dtype)
        nodes['left'], nodes['right'] = tree.children_left, tree.children_right
        nodes['feature'], nodes['threshold'] = tree.feature, tree.threshold
        nodes['value'] = tree.value[:, 0, 0]
        nodes['missing_left'] = tree.missing_go_to_left
        np.save(directory / f'tree_{index:04d}.npy', nodes, allow_pickle=False)
    return MappedForestRegressor(directory.name, len(forest.estimators_), forest.n_features_in_)
