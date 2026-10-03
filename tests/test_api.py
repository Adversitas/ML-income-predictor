import pytest
from fastapi.testclient import TestClient

from mlworkflow import api
from mlworkflow.data import split
from mlworkflow.features import build_model, feature_columns

RECORD = {
    "age": 45, "workclass": "Private", "education": "Bachelors", "education-num": 13,
    "marital-status": "Married", "occupation": "Tech", "relationship": "Husband",
    "capital-gain": 0, "capital-loss": 0, "hours-per-week": 50,
    "native-country": "United-States", "race": "White", "sex": "Male",
}


@pytest.fixture
def client(adult_like, cfg, monkeypatch):
    s = split(adult_like, cfg)
    numeric, categorical = feature_columns(s.X_train, cfg.data.exclude_features)
    cols = numeric + categorical
    model = build_model("logreg", {}, numeric, categorical, seed=0).fit(s.X_train[cols], s.y_train)
    monkeypatch.setattr(api, "cfg", cfg)
    monkeypatch.setattr(api.ModelHolder, "reload", lambda self, name: None)  # no registry in tests
    api.holder.set(model, "7", 0.4)
    with TestClient(api.app) as c:
        yield c
    api.holder.set(None, None, 0.5)


def test_predict_returns_probabilities_and_logs(client, cfg):
    r = client.post("/predict", json={"records": [RECORD, {**RECORD, "age": 22}]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "7"
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert 0 <= p["probability"] <= 1
        assert p["label"] == int(p["probability"] >= 0.4)
    assert len(cfg.inference_log.read_text().splitlines()) == 2


def test_invalid_input_is_rejected(client):
    assert client.post("/predict", json={"records": [{**RECORD, "age": 5}]}).status_code == 422
    assert client.post("/predict", json={"records": []}).status_code == 422


def test_no_model_returns_503(client):
    api.holder.set(None, None, 0.5)
    assert client.post("/predict", json={"records": [RECORD]}).status_code == 503
    assert client.get("/health").json()["model_loaded"] is False
