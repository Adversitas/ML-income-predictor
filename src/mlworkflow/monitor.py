"""Feature drift via the Population Stability Index (PSI).

Rule of thumb: PSI < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mlworkflow.config import Config

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


def report_path(cfg: Config):
    return cfg.data_dir / "monitoring" / "drift_report.json"


def run_drift_check(cfg: Config, last_n: int | None = None) -> dict:
    """Compare logged inference requests with the training reference and save the report."""
    if not cfg.inference_log.exists() or cfg.inference_log.stat().st_size == 0:
        raise FileNotFoundError(f"no inference requests logged at {cfg.inference_log}; send traffic first")
    reference = pd.read_parquet(cfg.reference_path)
    reference = reference.drop(columns=[c for c in cfg.data.exclude_features if c in reference])
    current = pd.read_json(cfg.inference_log, lines=True)
    if last_n:
        current = current.tail(last_n)
    report = drift_report(reference, current, cfg.monitoring.psi_threshold)
    out = report_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
