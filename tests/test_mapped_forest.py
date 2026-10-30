import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.inference.mapped_forest import export_forest


def test_mapped_predictions_match_fitted_forest_including_missing_values(tmp_path, monkeypatch):
    import src.config as config
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 4))
    X[::7, 1] = np.nan
    y = X[:, 0] * 100 + np.nan_to_num(X[:, 1]) * 3
    forest = RandomForestRegressor(n_estimators=11, random_state=7, min_samples_leaf=2).fit(X, y)
    mapped = export_forest(forest, tmp_path / 'ev_test.forest')
    monkeypatch.setattr(config, 'MODELS_DIR', tmp_path)
    unseen = rng.normal(size=(25, 4))
    unseen[::3, 1] = np.nan
    np.testing.assert_allclose(mapped.predict(unseen), forest.predict(unseen), atol=1e-10, rtol=0)
    np.testing.assert_allclose(Pipeline([('model', mapped)]).predict(unseen), forest.predict(unseen), atol=1e-10, rtol=0)
    np.testing.assert_allclose(mapped.predict(X), forest.predict(X), atol=1e-10, rtol=0)
    assert mapped.predict(np.empty((0, 4))).shape == (0,)
    with pytest.raises(ValueError):
        mapped.predict(np.ones((2, 3)))


def test_predictor_does_not_load_other_dataset_models(tmp_path, monkeypatch):
    import json
    from src.inference import ev_predict
    paths = []
    for dataset in ('ev_degradation', 'ev_telematics'):
        path = tmp_path / f'ev_{dataset}.joblib'
        path.touch()
        path.with_suffix('.json').write_text(json.dumps({'dataset': dataset}))
    def load(path):
        paths.append(path.name)
        return {'meta': {'dataset': 'ev_degradation', 'task': 'selected'}}
    monkeypatch.setattr(ev_predict.joblib, 'load', load)
    predictor = ev_predict.EVPredictor('ev_degradation', tmp_path)
    assert list(predictor.models) == ['selected']
    assert paths == ['ev_ev_degradation.joblib']
