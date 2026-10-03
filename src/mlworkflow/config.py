"""Typed view of configs/pipeline.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/pipeline.yaml")


@dataclass(frozen=True)
class DataConfig:
    openml_id: int
    target: str
    positive_label: str
    exclude_features: list[str]
    slice_columns: list[str]
    test_size: float
    val_size: float
    seed: int


@dataclass(frozen=True)
class PromotionConfig:
    metric: str
    min_improvement: float


@dataclass(frozen=True)
class MonitoringConfig:
    psi_threshold: float


@dataclass(frozen=True)
class Config:
    experiment: str
    model_name: str
    data: DataConfig
    models: dict[str, dict[str, Any]]
    promotion: PromotionConfig
    monitoring: MonitoringConfig
    data_dir: Path

    @property
    def raw_path(self) -> Path:
        return self.data_dir / "raw" / f"openml_{self.data.openml_id}.parquet"

    @property
    def reference_path(self) -> Path:
        """Training features the drift monitor compares live traffic against."""
        return self.data_dir / "reference" / "train.parquet"

    @property
    def inference_log(self) -> Path:
        return self.data_dir / "inference" / "requests.jsonl"


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        experiment=raw["experiment"],
        model_name=raw["model_name"],
        data=DataConfig(**raw["data"]),
        models=raw["models"],
        promotion=PromotionConfig(**raw["promotion"]),
        monitoring=MonitoringConfig(**raw["monitoring"]),
        data_dir=Path(os.environ.get("MLWF_DATA_DIR", "data")),
    )
