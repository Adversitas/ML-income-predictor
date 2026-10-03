"""Prediction service. Serves the registry's `champion` and logs every request for drift monitoring."""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mlworkflow.config import load_config
from mlworkflow.registry import load_champion


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


cfg = load_config()
holder = ModelHolder()
_log_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        holder.reload(cfg.model_name)
    except Exception as exc:  # registry unreachable: start anyway, /health reports it
        print(f"could not load champion: {exc}")
    yield


app = FastAPI(title="Adult income model", lifespan=lifespan)


def log_requests(df: pd.DataFrame, proba) -> None:
    cfg.inference_log.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    rows = df.assign(probability=proba, model_version=holder.version, ts=now)
    with _log_lock, cfg.inference_log.open("a", encoding="utf-8") as f:
        for rec in rows.to_dict(orient="records"):
            f.write(json.dumps(rec, default=str) + "\n")


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
    model, version, threshold = holder.model, holder.version, holder.threshold
    if model is None:
        raise HTTPException(503, "no champion model in the registry yet; run the pipeline")
    df = pd.DataFrame([r.model_dump(by_alias=True) for r in req.records])
    proba = model.predict_proba(df[list(model.feature_names_in_)])[:, 1]
    log_requests(df, proba)
    return PredictResponse(
        model_version=str(version),
        threshold=threshold,
        predictions=[Prediction(probability=float(p), label=int(p >= threshold)) for p in proba],
    )
