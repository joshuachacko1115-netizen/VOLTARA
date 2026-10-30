from starlette.testclient import TestClient

from webapp import server
from scripts.prepare_render import allowed


def test_health_and_unavailable_inference(monkeypatch):
    def missing():
        raise FileNotFoundError('missing model')
    monkeypatch.setattr(server, 'quick_models', missing)
    client = TestClient(server.app)
    assert client.get('/healthz').json() == {'status': 'ok'}
    assert client.get('/readyz').status_code == 503


def test_pages_preflight_and_untrusted_origin():
    client = TestClient(server.app)
    for origin, expected in [('https://joshuachacko1115-netizen.github.io', 200),
                             ('https://untrusted.example', 400)]:
        response = client.options('/api/assess', headers={
            'Origin': origin, 'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'content-type'})
        assert response.status_code == expected


def test_runtime_bundle_rejects_unsafe_and_unrelated_paths():
    assert allowed('models/ev_quick_soh.joblib')
    assert allowed('data/raw/ev/sample.csv')
    assert allowed('reports/results/ev_split_units.csv')
    assert allowed('models/ev_degradation_rul.forest/tree_0000.npy')
    for name in ('../outside.csv', '/models/ev_x.joblib', 'models/../../secret',
                 'webapp/server.py', 'models/ev_x.exe', 'data/raw/../../../secret.csv',
                 'C:/models/ev_x.joblib'):
        assert not allowed(name)
