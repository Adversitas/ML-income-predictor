"""Metrics, decision threshold and per-group audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, proba) -> float:
    """Threshold maximising F1, picked on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # the last precision/recall pair has no threshold
    return float(thresholds[np.argmax(f1[:-1])]) if len(thresholds) else 0.5


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    one_class = len(np.unique(y_true)) < 2
    return {
        "roc_auc": float("nan") if one_class else roc_auc_score(y_true, proba),
        "pr_auc": float("nan") if one_class else average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "positive_rate": float(pred.mean()),
    }


def slice_metrics(X: pd.DataFrame, y_true, proba, threshold: float,
                  columns: list[str]) -> pd.DataFrame:
    """Metrics per group, so gaps between e.g. sexes are reported rather than averaged away."""
    rows = []
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    for col in columns:
        for value, idx in X.groupby(col, dropna=False).indices.items():
            m = classification_metrics(y_true[idx], proba[idx], threshold)
            rows.append({"column": col, "value": str(value), "n": len(idx),
                         "base_rate": float(y_true[idx].mean()), **m})
    return pd.DataFrame(rows)
