import numpy as np

from mlworkflow.validate import validate


def test_clean_data_passes(adult_like, cfg):
    report = validate(adult_like, cfg)
    assert report.ok, report.errors
    assert any("workclass" in w for w in report.warnings)  # small share of missing values


def test_missing_column_fails(adult_like, cfg):
    report = validate(adult_like.drop(columns=["age"]), cfg)
    assert not report.ok
    assert "age" in report.errors[0]


def test_out_of_range_fails(adult_like, cfg):
    adult_like.loc[:4, "age"] = 250
    report = validate(adult_like, cfg)
    assert any("age" in e and "outside" in e for e in report.errors)


def test_too_many_nulls_fails(adult_like, cfg):
    adult_like.loc[: len(adult_like) // 2, "occupation"] = np.nan
    assert not validate(adult_like, cfg).ok


def test_non_binary_target_fails(adult_like, cfg):
    adult_like.loc[0, "class"] = "unknown"
    assert not validate(adult_like, cfg).ok
