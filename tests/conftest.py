import numpy as np
import pandas as pd
import pytest

from mlworkflow.config import load_config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MLWF_DATA_DIR", str(tmp_path))
    return load_config()


@pytest.fixture
def adult_like() -> pd.DataFrame:
    """Small synthetic frame with the Adult schema, where income depends on age, education and hours."""
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.integers(17, 90, n).astype(float)
    edu = rng.integers(1, 17, n).astype(float)
    hours = rng.integers(1, 99, n).astype(float)
    logit = 0.04 * (age - 40) + 0.4 * (edu - 10) + 0.03 * (hours - 40) - 1
    rich = rng.random(n) < 1 / (1 + np.exp(-logit))
    workclass = rng.choice(["Private", "Self-emp", "Gov"], n).astype(object)
    workclass[rng.random(n) < 0.05] = np.nan
    return pd.DataFrame({
        "age": age,
        "workclass": workclass,
        "fnlwgt": rng.integers(10_000, 500_000, n).astype(float),
        "education": rng.choice(["HS-grad", "Bachelors", "Masters"], n),
        "education-num": edu,
        "marital-status": rng.choice(["Married", "Never-married"], n),
        "occupation": rng.choice(["Sales", "Tech", "Craft"], n),
        "relationship": rng.choice(["Husband", "Wife", "Own-child"], n),
        "race": rng.choice(["White", "Black", "Other"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "capital-gain": np.where(rng.random(n) < 0.1, rng.integers(1, 20_000, n), 0).astype(float),
        "capital-loss": np.zeros(n),
        "hours-per-week": hours,
        "native-country": rng.choice(["United-States", "Mexico"], n),
        "class": np.where(rich, ">50K", "<=50K"),
    })
