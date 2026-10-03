import numpy as np
import pytest

from mlworkflow.data import dataset_hash, split
from mlworkflow.evaluate import best_f1_threshold, classification_metrics, slice_metrics
from mlworkflow.features import build_model, feature_columns


def test_split_is_stratified_and_disjoint(adult_like, cfg):
    s = split(adult_like, cfg)
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == len(adult_like)
    assert not set(s.X_train.index) & set(s.X_test.index)
    assert abs(s.y_train.mean() - s.y_test.mean()) < 0.02


def test_hash_tracks_content(adult_like):
    h = dataset_hash(adult_like)
    assert h == dataset_hash(adult_like.copy())
    adult_like.loc[0, "age"] += 1
    assert h != dataset_hash(adult_like)


def test_excluded_features_are_not_used(adult_like, cfg):
    numeric, categorical = feature_columns(adult_like.drop(columns=["class"]),
                                           cfg.data.exclude_features)
    assert not {"fnlwgt", "race", "sex"} & set(numeric + categorical)


@pytest.mark.parametrize("name", ["logreg", "hgb"])
def test_models_beat_baseline(adult_like, cfg, name):
    s = split(adult_like, cfg)
    numeric, categorical = feature_columns(s.X_train, cfg.data.exclude_features)
    cols = numeric + categorical
    model = build_model(name, {}, numeric, categorical, seed=0).fit(s.X_train[cols], s.y_train)
    proba = model.predict_proba(s.X_test[cols])[:, 1]
    assert classification_metrics(s.y_test, proba, 0.5)["roc_auc"] > 0.75


def test_unseen_category_and_missing_values_are_handled(adult_like, cfg):
    s = split(adult_like, cfg)
    numeric, categorical = feature_columns(s.X_train, cfg.data.exclude_features)
    cols = numeric + categorical
    model = build_model("logreg", {}, numeric, categorical, seed=0).fit(s.X_train[cols], s.y_train)
    row = s.X_test[cols].head(2).copy()
    row.loc[row.index[0], "occupation"] = "Astronaut"
    row.loc[row.index[1], "workclass"] = None
    row.loc[row.index[1], "age"] = np.nan
    assert model.predict_proba(row).shape == (2, 2)


def test_threshold_and_slices():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.3, 0.35, 0.8, 0.2, 0.9])  # separable at 0.35
    t = best_f1_threshold(y, p)
    assert classification_metrics(y, p, t)["f1"] == pytest.approx(1.0)

    import pandas as pd
    X = pd.DataFrame({"sex": ["M", "F", "M", "F", "M", "F"]})
    out = slice_metrics(X, y, p, t, ["sex"])
    assert set(out["value"]) == {"M", "F"}
    assert out["n"].sum() == len(y)
