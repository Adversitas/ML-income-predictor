"""MLflow Model Registry helpers. The served model is whichever version holds the `champion` alias."""

from __future__ import annotations

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

CHAMPION = "champion"


def get_champion(client: MlflowClient, name: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None


def champion_uri(name: str) -> str:
    return f"models:/{name}@{CHAMPION}"


def load_champion(name: str):
    """Returns (sklearn pipeline, model version) or (None, None) when nothing is promoted yet."""
    client = MlflowClient()
    version = get_champion(client, name)
    if version is None:
        return None, None
    return mlflow.sklearn.load_model(champion_uri(name)), version
