"""Export inference-only copies of trusted local EV models without changing originals.

Requires py-ubjson in the conversion environment. XGBoost's portable model format
preserves learned trees; compiler/platform-specific training RNG state is omitted.
The output is for prediction, not for resuming the original training session.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost

ROOT = Path(__file__).resolve().parents[1]


def load_portable(path):
    import ubjson
    original = xgboost.Booster.__setstate__

    def restore(booster, state):
        # py-ubjson's decoder reads past a counted empty array. Normalize its
        # zero-length representation before decoding the trusted snapshot.
        data = bytes(state['handle']).replace(b'[#L' + b'\x00' * 8, b'[]')
        snapshot = ubjson.loadb(data, no_bytes=True)
        portable = xgboost.Booster()
        portable.load_model(bytearray(json.dumps(snapshot['Model']).encode()))
        booster.__dict__.update(portable.__dict__)
        portable.handle = None  # Transfer ownership; do not free the live handle.

    try:
        xgboost.Booster.__setstate__ = restore
        return joblib.load(path)
    finally:
        xgboost.Booster.__setstate__ = original


def verify(obj):
    meta = obj['meta']
    table = ROOT / 'data/processed/ev' / f"{meta['dataset']}_features.parquet"
    frame = pd.read_parquet(table)
    test = frame[frame.split == 'test']
    X = test[meta['categorical_features'] + meta['numeric_features']]
    if meta['kind'] == 'regression':
        pred = obj['pipeline'].predict(X)
        observed = float(np.mean(np.abs(pred - test[meta['target']].to_numpy())))
        expected = meta['test_scores']['MAE']
    else:
        pred = obj['pipeline'].predict_proba(X).argmax(axis=1)
        labels = test[meta['target']].map({c: i for i, c in enumerate(meta['classes'])}).to_numpy()
        observed = float(np.mean(pred == labels))
        expected = meta['test_scores']['accuracy']
    if not np.isclose(observed, expected, atol=1e-8, rtol=0):
        raise ValueError(f"{meta['task']}: score changed ({observed} != {expected})")
    print(f"{meta['task']}: held-out score matches ({observed:.12g})", flush=True)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/render-runtime/models')
    parser.add_argument('--mapped-forest', action='store_true', help='Use exact file-backed storage for large regression forests')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == (ROOT / 'models').resolve():
        raise ValueError('Output must not overwrite the original models directory.')
    output.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / 'models').glob('ev_*.joblib')):
        obj = load_portable(source)
        expected_predictions = verify(obj)
        if args.mapped_forest and obj['meta']['kind'] == 'regression':
            from sklearn.ensemble import RandomForestRegressor
            from src.inference.mapped_forest import export_forest
            model = obj['pipeline'].named_steps['model']
            if isinstance(model, RandomForestRegressor):
                obj['pipeline'].steps[-1] = ('model', export_forest(model, output / (source.stem + '.forest')))
                del model
        import src.config as config
        config.MODELS_DIR = output
        destination = output / source.name
        joblib.dump(obj, destination, compress=3)
        # Check normal loading without the compatibility hook.
        np.testing.assert_allclose(verify(joblib.load(destination)), expected_predictions, rtol=1e-13, atol=1e-9)
        print(f'Exported {destination.name}: {destination.stat().st_size} bytes', flush=True)
    for source in (ROOT / 'models').glob('ev_*.json'):
        (output / source.name).write_bytes(source.read_bytes())


if __name__ == '__main__':
    main()
