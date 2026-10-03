"""Model pipelines. Preprocessing lives inside the sklearn Pipeline so serving can't drift from training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def fill_missing_category(X: pd.DataFrame) -> pd.DataFrame:
    """Missing is informative here (e.g. no workclass recorded), so it becomes its own category."""
    X = pd.DataFrame(X).astype(object)
    return X.where(X.notna(), "missing")


# MLflow serialises sklearn models with skops, which refuses to load types outside its safe
# list, so a tampered model file can't run arbitrary code at load time. These extra types are
# trusted because only this pipeline writes to the artifact store.
SKOPS_TRUSTED_TYPES = [
    f"{__name__}.fill_missing_category",
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def feature_columns(X: pd.DataFrame, exclude: list[str]) -> tuple[list[str], list[str]]:
    cols = [c for c in X.columns if c not in exclude]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("fill", FunctionTransformer(fill_missing_category, feature_names_out="one-to-one")),
        # Rare and unseen categories share one bucket instead of failing at inference time.
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


MODEL_FACTORIES: dict[str, Callable[[dict[str, Any], int], Any]] = {
    "baseline": lambda p, seed: DummyClassifier(strategy="prior", **p),
    "logreg": lambda p, seed: LogisticRegression(max_iter=2000, random_state=seed, **p),
    "hgb": lambda p, seed: HistGradientBoostingClassifier(random_state=seed, **p),
}


def build_model(name: str, params: dict[str, Any], numeric: list[str], categorical: list[str],
                seed: int) -> Pipeline:
    if name not in MODEL_FACTORIES:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_FACTORIES)}")
    return Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("clf", MODEL_FACTORIES[name](params, seed)),
    ])
