"""Data contract checks run before training. Errors stop the pipeline; warnings are logged."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mlworkflow.config import Config

NUMERIC_RANGES = {
    "age": (16, 100),
    "education-num": (1, 16),
    "capital-gain": (0, 100_000),
    "capital-loss": (0, 5_000),
    "hours-per-week": (1, 99),
}
CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
MIN_ROWS = 1_000
MAX_NULL_RATE = 0.10


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def validate(df: pd.DataFrame, cfg: Config) -> ValidationReport:
    report = ValidationReport()
    expected = set(NUMERIC_RANGES) | set(CATEGORICAL) | {cfg.data.target}
    missing = expected - set(df.columns)
    if missing:
        report.errors.append(f"missing columns: {sorted(missing)}")
        return report

    if len(df) < MIN_ROWS:
        report.errors.append(f"only {len(df)} rows, need at least {MIN_ROWS}")

    labels = set(df[cfg.data.target].dropna().unique())
    if cfg.data.positive_label not in labels or len(labels) != 2:
        report.errors.append(f"target must be binary and contain {cfg.data.positive_label!r}, "
                             f"got {sorted(map(str, labels))}")
    if df[cfg.data.target].isna().any():
        report.errors.append("target has missing values")

    for col, (lo, hi) in NUMERIC_RANGES.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.errors.append(f"{col} should be numeric, got {df[col].dtype}")
            continue
        out = df[col].dropna().between(lo, hi, inclusive="both").eq(False).sum()
        if out:
            report.errors.append(f"{col}: {out} values outside [{lo}, {hi}]")

    for col in [*NUMERIC_RANGES, *CATEGORICAL]:
        rate = df[col].isna().mean()
        if rate > MAX_NULL_RATE:
            report.errors.append(f"{col}: {rate:.1%} missing (limit {MAX_NULL_RATE:.0%})")
        elif rate > 0:
            report.warnings.append(f"{col}: {rate:.1%} missing, imputed downstream")

    dupes = df.duplicated().sum()
    if dupes:
        report.warnings.append(f"{dupes} duplicate rows")
    return report
