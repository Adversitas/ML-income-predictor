"""Feature drift via the Population Stability Index (PSI).

Rule of thumb: PSI < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = reference.dropna().to_numpy(float), current.dropna().to_numpy(float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    # Bin edges from reference quantiles; skewed columns like capital-gain collapse to fewer bins.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    expected = np.histogram(ref, edges)[0] / len(ref)
    actual = np.histogram(cur, edges)[0] / len(cur)
    return _psi(expected, actual)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("missing").astype(str).value_counts(normalize=True)
    cur = current.fillna("missing").astype(str).value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    return _psi(ref.reindex(cats, fill_value=0).to_numpy(), cur.reindex(cats, fill_value=0).to_numpy())


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, threshold: float) -> dict:
    features = {}
    for col in reference.columns:
        if col not in current.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference[col]):
            score = psi_numeric(reference[col], pd.to_numeric(current[col], errors="coerce"))
        else:
            score = psi_categorical(reference[col], current[col])
        features[col] = {"psi": round(score, 4), "drift": bool(score > threshold)}
    drifted = sorted(c for c, f in features.items() if f["drift"])
    return {
        "n_reference": len(reference),
        "n_current": len(current),
        "threshold": threshold,
        "drift_detected": bool(drifted),
        "drifted_features": drifted,
        "features": features,
    }
