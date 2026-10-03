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


@pytest.fixture
def web_client(client, adult_like, cfg):
    """API client with the raw snapshot and training reference on disk, as after a pipeline run."""
    cfg.raw_path.parent.mkdir(parents=True, exist_ok=True)
    adult_like.to_parquet(cfg.raw_path, index=False)
    cfg.reference_path.parent.mkdir(parents=True, exist_ok=True)
    split(adult_like, cfg).X_train.to_parquet(cfg.reference_path, index=False)
    api._test_split.cache_clear()
    yield client
    api._test_split.cache_clear()


def test_index_serves_the_web_app(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>ML Income Predictor</title>" in r.text


def test_schema_lists_model_inputs_only(web_client):
    fields = {f["name"]: f for f in web_client.get("/api/schema").json()["fields"]}
    assert not {"fnlwgt", "race", "sex"} & set(fields)
    assert fields["age"]["type"] == "number"
    assert "Private" in fields["workclass"]["options"]


def test_example_can_be_scored(web_client):
    ex = web_client.get("/api/example").json()
    assert ex["actual"] in (0, 1)
    assert web_client.post("/predict", json={"records": [ex["record"]]}).status_code == 200


def test_traffic_then_drift_check(web_client):
    assert web_client.post("/api/drift").status_code == 409  # nothing logged yet

    web_client.post("/api/traffic", json={"n": 300, "drift": False})
    clean = web_client.post("/api/drift").json()
    assert clean["log"]["requests"] == 300
    assert not clean["report"]["drift_detected"]

    web_client.delete("/api/requests")
    web_client.post("/api/traffic", json={"n": 300, "drift": True})
    shifted = web_client.post("/api/drift").json()
    assert "age" in shifted["report"]["drifted_features"]

    assert web_client.delete("/api/requests").json() == {"log": {"requests": 0}, "report": None}
