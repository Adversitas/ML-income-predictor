"""Prediction service and web app.

Serves the registry's `champion`, logs every request for drift monitoring, and backs the
browser UI at `/` with the `/api/*` endpoints.
"""

from __future__ import annotations

import json
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from mlworkflow import data as data_mod
from mlworkflow.config import load_config
from mlworkflow.monitor import report_path, run_drift_check
from mlworkflow.registry import CHAMPION, load_champion

WEB_DIR = Path(__file__).parent / "web"


class Person(BaseModel):
    """One census record. Field aliases match the dataset's column names."""

    model_config = ConfigDict(populate_by_name=True)

    age: float = Field(ge=16, le=100)
    workclass: str | None = None
    education: str | None = None
    education_num: float = Field(alias="education-num", ge=1, le=16)
    marital_status: str | None = Field(None, alias="marital-status")
    occupation: str | None = None
    relationship: str | None = None
    capital_gain: float = Field(0, alias="capital-gain", ge=0)
    capital_loss: float = Field(0, alias="capital-loss", ge=0)
    hours_per_week: float = Field(alias="hours-per-week", ge=1, le=99)
    native_country: str | None = Field(None, alias="native-country")
    # Not model inputs; accepted so monitoring can audit traffic per group.
    race: str | None = None
    sex: str | None = None


class PredictRequest(BaseModel):
    records: list[Person] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    probability: float
    label: int


class PredictResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


class TrafficRequest(BaseModel):
    n: int = Field(500, ge=1, le=5000)
    drift: bool = False


class ModelHolder:
    def __init__(self) -> None:
        self.model = None
        self.version: str | None = None
        self.threshold = 0.5
        self._lock = threading.Lock()

    def set(self, model, version: str | None, threshold: float) -> None:
        with self._lock:
            self.model, self.version, self.threshold = model, version, threshold

    def reload(self, model_name: str) -> None:
        model, version = load_champion(model_name)
        if model is None:
            self.set(None, None, 0.5)
        else:
            self.set(model, version.version, float(version.tags.get("threshold", 0.5)))


class TrainJob:
    """Runs the train-and-promote flow in a background thread; one at a time."""

    def __init__(self) -> None:
        self.state = {"status": "idle"}
        self._lock = threading.Lock()

    def start(self) -> dict:
        with self._lock:
            if self.state["status"] == "running":
                raise HTTPException(409, "training is already running")
            self.state = {"status": "running", "started": _now()}
        threading.Thread(target=self._run, daemon=True).start()
        return self.state

    def _run(self) -> None:
        from mlworkflow.pipeline import train_flow  # heavy import, only when training

        try:
            result = train_flow()
            holder.reload(cfg.model_name)
            self.state = {**self.state, "status": "succeeded", "finished": _now(), "result": result}
        except Exception as exc:
            self.state = {**self.state, "status": "failed", "finished": _now(), "error": str(exc)}


cfg = load_config()
holder = ModelHolder()
train_job = TrainJob()
_log_lock = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        holder.reload(cfg.model_name)
    except Exception as exc:  # registry unreachable: start anyway, /health reports it
        print(f"could not load champion: {exc}")
    yield


app = FastAPI(title="ML Income Predictor", lifespan=lifespan)


def log_requests(df: pd.DataFrame, proba, version: str | None) -> None:
    cfg.inference_log.parent.mkdir(parents=True, exist_ok=True)
    rows = df.assign(probability=proba, model_version=version, ts=_now())
    with _log_lock, cfg.inference_log.open("a", encoding="utf-8") as f:
        for rec in rows.to_dict(orient="records"):
            f.write(json.dumps(rec, default=str) + "\n")


def score(df: pd.DataFrame) -> PredictResponse:
    model, version, threshold = holder.model, holder.version, holder.threshold
    if model is None:
        raise HTTPException(503, "no champion model in the registry yet; run the pipeline")
    proba = model.predict_proba(df[list(model.feature_names_in_)])[:, 1]
    log_requests(df, proba, version)
    return PredictResponse(
        model_version=str(version),
        threshold=threshold,
        predictions=[Prediction(probability=float(p), label=int(p >= threshold)) for p in proba],
    )


# --- prediction API ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": holder.model is not None,
            "model_version": holder.version}


@app.post("/reload")
def reload() -> dict:
    """Pick up a newly promoted champion without restarting the service."""
    holder.reload(cfg.model_name)
    return health()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    return score(pd.DataFrame([r.model_dump(by_alias=True) for r in req.records]))


# --- web app ----------------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@lru_cache(maxsize=1)
def _test_split() -> tuple[pd.DataFrame, pd.Series]:
    if not cfg.raw_path.exists():
        raise HTTPException(503, "no data yet; run the pipeline first")
    s = data_mod.split(pd.read_parquet(cfg.raw_path), cfg)
    return s.X_test, s.y_test


@app.get("/api/schema")
def schema() -> dict:
    """Form fields, built from the training data the model actually saw."""
    if not cfg.reference_path.exists():
        raise HTTPException(503, "no training data yet; run the pipeline first")
    ref = pd.read_parquet(cfg.reference_path)
    ref = ref.drop(columns=[c for c in cfg.data.exclude_features if c in ref])
    fields = []
    for col in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[col]):
            fields.append({"name": col, "type": "number", "min": float(ref[col].min()),
                           "max": float(ref[col].max()), "default": float(ref[col].median())})
        else:
            counts = ref[col].value_counts()
            fields.append({"name": col, "type": "category", "options": counts.index.tolist(),
                           "default": counts.index[0]})
    # education and education-num encode the same thing; the form keeps them in sync
    edu = ref.groupby("education")["education-num"].agg(lambda s: s.mode().iloc[0]).to_dict()
    return {"fields": fields, "education_num": edu}


@app.get("/api/example")
def example() -> dict:
    """A random real person from the held-out test set, with the true outcome."""
    X, y = _test_split()
    i = X.sample(1).index[0]
    record = json.loads(X.loc[[i]].to_json(orient="records"))[0]
    return {"record": record, "actual": int(y.loc[i])}


def _download_slices(run_id: str) -> list[dict]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="slice_metrics.csv", dst_path=tmp)
            return pd.read_csv(path).to_dict(orient="records")
    except Exception:
        return []


@lru_cache(maxsize=32)
def _run_details(candidate_run_id: str) -> dict:
    """Test metrics, candidate comparison and group audit from the pipeline run behind a version."""
    client = MlflowClient()
    parent_id = client.get_run(candidate_run_id).data.tags.get("mlflow.parentRunId")
    if parent_id is None:
        return {}
    parent = client.get_run(parent_id)
    siblings = client.search_runs([parent.info.experiment_id],
                                  f"tags.mlflow.parentRunId = '{parent_id}'")
    candidates = sorted(
        ({"model": r.data.params.get("model"),
          **{k.removeprefix("val_"): v for k, v in r.data.metrics.items()}} for r in siblings),
        key=lambda c: c.get("roc_auc", 0), reverse=True)
    return {
        "test_metrics": {k.removeprefix("test_"): v for k, v in parent.data.metrics.items()
                         if k.startswith("test_")},
        "data_hash": parent.data.params.get("data_hash"),
        "n_train": parent.data.params.get("n_train"),
        "candidates": candidates,
        "slices": _download_slices(parent_id),
    }


@app.get("/api/model")
def model_info() -> dict:
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{cfg.model_name}'")
    except Exception as exc:
        raise HTTPException(503, f"model registry unreachable: {exc}") from exc
    if not versions:
        return {"champion": None, "history": []}
    rm = client.get_registered_model(cfg.model_name)
    champion_version = rm.aliases.get(CHAMPION)
    history = sorted((
        {"version": v.version, "candidate": v.tags.get("candidate"),
         "test_roc_auc": float(v.tags.get("test_roc_auc", "nan")),
         "promotion": v.tags.get("promotion"),
         "created": datetime.fromtimestamp(v.creation_timestamp / 1000, UTC).isoformat(timespec="seconds"),
         "champion": v.version == champion_version, "run_id": v.run_id}
        for v in versions), key=lambda h: int(h["version"]), reverse=True)
    champion = next((h for h in history if h["champion"]), None)
    if champion:
        champion = {**champion, "threshold": holder.threshold, "serving": holder.version,
                    **_run_details(champion["run_id"])}
    return {"champion": champion, "history": history}


@app.post("/api/train")
def start_training() -> dict:
    return train_job.start()


@app.get("/api/train")
def training_status() -> dict:
    return train_job.state


@app.post("/api/traffic")
def send_traffic(req: TrafficRequest) -> dict:
    """Replay held-out rows through the model, as if they were live requests."""
    if not cfg.raw_path.exists():
        raise HTTPException(503, "no data yet; run the pipeline first")
    rows = data_mod.sample_traffic(cfg, req.n, req.drift)
    result = score(rows)
    probs = [p.probability for p in result.predictions]
    return {"sent": len(rows), "drift": req.drift,
            "positive_rate": sum(p.label for p in result.predictions) / len(probs),
            "mean_probability": sum(probs) / len(probs)}


def _log_summary() -> dict:
    if not cfg.inference_log.exists() or cfg.inference_log.stat().st_size == 0:
        return {"requests": 0}
    log = pd.read_json(cfg.inference_log, lines=True)
    return {"requests": len(log), "mean_probability": float(log["probability"].mean()),
            "positive_rate": float((log["probability"] >= holder.threshold).mean()),
            "last": str(log["ts"].iloc[-1])}


@app.get("/api/monitoring")
def monitoring() -> dict:
    path = report_path(cfg)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return {"log": _log_summary(), "report": report}


@app.post("/api/drift")
def check_drift() -> dict:
    try:
        run_drift_check(cfg)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    return monitoring()


@app.delete("/api/requests")
def clear_requests() -> dict:
    with _log_lock:
        cfg.inference_log.unlink(missing_ok=True)
    report_path(cfg).unlink(missing_ok=True)
    return monitoring()
