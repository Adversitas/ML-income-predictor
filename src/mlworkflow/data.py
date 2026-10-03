"""Ingestion and splitting. The raw snapshot is cached so every run sees identical data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from mlworkflow.config import Config


def fetch_raw(cfg: Config) -> pd.DataFrame:
    if cfg.raw_path.exists():
        return pd.read_parquet(cfg.raw_path)
    frame = fetch_openml(
        data_id=cfg.data.openml_id,
        as_frame=True,
        parser="auto",
        data_home=str(cfg.data_dir / "openml_cache"),
    ).frame
    # Plain object/float columns: pandas categoricals carry their category list into
    # parquet and the API, which makes unseen values harder to reason about.
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            frame[col] = frame[col].astype(object)
        elif pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].astype(float)
    cfg.raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cfg.raw_path, index=False)
    return frame


def dataset_hash(df: pd.DataFrame) -> str:
    """Content hash logged with every run, so a metric can be traced to the exact data."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def make_target(df: pd.DataFrame, cfg: Config) -> pd.Series:
    return (df[cfg.data.target] == cfg.data.positive_label).astype(int)


def split(df: pd.DataFrame, cfg: Config) -> Splits:
    """Stratified train/val/test split. X keeps every column; feature selection is the model's job."""
    d = cfg.data
    y = make_target(df, cfg)
    X = df.drop(columns=[d.target])
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=d.test_size, stratify=y, random_state=d.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=d.val_size, stratify=y_rest, random_state=d.seed
    )
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)
