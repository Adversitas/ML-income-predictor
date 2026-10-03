"""Prefect flows: train-and-promote, and drift check.

train-and-promote:
  ingest -> validate -> split -> train candidates (one MLflow run each)
  -> pick best on validation -> evaluate on test + per-group audit
  -> register -> promote to `champion` only if it beats the current champion on the same test set
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sklearn.pipeline import Pipeline

from mlworkflow import data as data_mod
from mlworkflow.config import DEFAULT_CONFIG, Config, load_config
from mlworkflow.evaluate import best_f1_threshold, classification_metrics, slice_metrics
from mlworkflow.features import SKOPS_TRUSTED_TYPES, build_model, feature_columns
from mlworkflow.monitor import run_drift_check
from mlworkflow.registry import CHAMPION, get_champion
from mlworkflow.validate import validate


@dataclass
class Candidate:
    name: str
    model: Pipeline
    threshold: float
    val_metrics: dict[str, float]
    run_id: str
    model_uri: str


@task(retries=2, retry_delay_seconds=10, cache_policy=NO_CACHE)
def ingest(cfg: Config) -> pd.DataFrame:
    df = data_mod.fetch_raw(cfg)
    get_run_logger().info("ingested %d rows x %d columns", *df.shape)
    return df


@task(cache_policy=NO_CACHE)
def check_data(df: pd.DataFrame, cfg: Config) -> dict:
    report = validate(df, cfg)
    logger = get_run_logger()
    for w in report.warnings:
        logger.warning(w)
    if not report.ok:
        raise ValueError("data validation failed: " + "; ".join(report.errors))
    return report.to_dict()


@task(cache_policy=NO_CACHE)
def train_candidate(name: str, params: dict, splits: data_mod.Splits, cfg: Config) -> Candidate:
    numeric, categorical = feature_columns(splits.X_train, cfg.data.exclude_features)
    features = numeric + categorical
    X_train, X_val = splits.X_train[features], splits.X_val[features]

    with mlflow.start_run(run_name=name, nested=True) as run:
        model = build_model(name, params, numeric, categorical, cfg.data.seed)
        model.fit(X_train, splits.y_train)
        proba = model.predict_proba(X_val)[:, 1]
        threshold = best_f1_threshold(splits.y_val, proba)
        metrics = classification_metrics(splits.y_val, proba, threshold)

        mlflow.log_params({"model": name, "threshold": round(threshold, 4), **params})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_val, proba),
            input_example=X_val.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    get_run_logger().info("%s: val %s=%.4f", name, cfg.promotion.metric,
                          metrics[cfg.promotion.metric])
    return Candidate(name, model, threshold, metrics, run.info.run_id, info.model_uri)


@task(cache_policy=NO_CACHE)
def evaluate_on_test(best: Candidate, splits: data_mod.Splits, cfg: Config) -> dict[str, float]:
    features = list(best.model.feature_names_in_)
    proba = best.model.predict_proba(splits.X_test[features])[:, 1]
    metrics = classification_metrics(splits.y_test, proba, best.threshold)
    slices = slice_metrics(splits.X_test, splits.y_test, proba, best.threshold,
                           cfg.data.slice_columns)
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "slice_metrics.csv"
        slices.round(4).to_csv(path, index=False)
        mlflow.log_artifact(str(path))
    return metrics


@task(cache_policy=NO_CACHE)
def register_and_promote(best: Candidate, test_metrics: dict[str, float],
                         splits: data_mod.Splits, cfg: Config) -> dict:
    """Champion and challenger are scored on the same test set right now, not on stored numbers."""
    logger = get_run_logger()
    client = MlflowClient()
    metric = cfg.promotion.metric
    challenger_score = test_metrics[metric]

    champion = get_champion(client, cfg.model_name)
    champion_score = None
    if champion is not None:
        model = mlflow.sklearn.load_model(f"models:/{cfg.model_name}/{champion.version}")
        features = list(model.feature_names_in_)
        proba = model.predict_proba(splits.X_test[features])[:, 1]
        threshold = float(champion.tags.get("threshold", 0.5))
        champion_score = classification_metrics(splits.y_test, proba, threshold)[metric]

    promote = champion_score is None or (
        challenger_score >= champion_score + cfg.promotion.min_improvement
    )
    version = mlflow.register_model(best.model_uri, cfg.model_name)
    tags = {
        "candidate": best.name,
        "threshold": f"{best.threshold:.6f}",
        f"test_{metric}": f"{challenger_score:.6f}",
        "promotion": "accepted" if promote else "rejected",
    }
    for k, v in tags.items():
        client.set_model_version_tag(cfg.model_name, version.version, k, v)
    if promote:
        client.set_registered_model_alias(cfg.model_name, CHAMPION, version.version)

    decision = {
        "version": version.version,
        "candidate": best.name,
        "challenger": round(challenger_score, 4),
        "champion": None if champion_score is None else round(champion_score, 4),
        "promoted": promote,
    }
    mlflow.log_dict(decision, "promotion.json")
    logger.info("promotion decision: %s", json.dumps(decision))
    return decision


@flow(name="train-and-promote")
def train_flow(config_path: str = str(DEFAULT_CONFIG)) -> dict:
    cfg = load_config(config_path)
    mlflow.set_experiment(cfg.experiment)

    df = ingest(cfg)
    report = check_data(df, cfg)
    splits = data_mod.split(df, cfg)

    # Reference sample for drift monitoring: what the model saw in training.
    cfg.reference_path.parent.mkdir(parents=True, exist_ok=True)
    splits.X_train.to_parquet(cfg.reference_path, index=False)

    with mlflow.start_run(run_name="train-and-promote"):
        mlflow.log_params({
            "data_hash": data_mod.dataset_hash(df),
            "n_train": len(splits.X_train),
            "n_val": len(splits.X_val),
            "n_test": len(splits.X_test),
            "excluded_features": ",".join(cfg.data.exclude_features),
        })
        mlflow.log_dict(report, "validation_report.json")
        mlflow.log_artifact(config_path)

        candidates = [train_candidate(name, params or {}, splits, cfg)
                      for name, params in cfg.models.items()]
        best = max(candidates, key=lambda c: c.val_metrics[cfg.promotion.metric])
        mlflow.log_param("selected_candidate", best.name)

        test_metrics = evaluate_on_test(best, splits, cfg)
        return register_and_promote(best, test_metrics, splits, cfg)


@flow(name="drift-check")
def drift_flow(config_path: str = str(DEFAULT_CONFIG), last_n: int | None = None) -> dict:
    """Compare logged inference requests with the training reference."""
    cfg = load_config(config_path)
    logger = get_run_logger()
    report = run_drift_check(cfg, last_n)
    if report["drift_detected"]:
        # With fresh labelled data this is where retraining would be triggered.
        logger.warning("drift detected in %s", report["drifted_features"])
    else:
        logger.info("no drift across %d features", len(report["features"]))
    return report
