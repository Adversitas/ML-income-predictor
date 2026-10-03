import numpy as np
import pandas as pd

from mlworkflow.monitor import drift_report, psi_categorical, psi_numeric


def test_same_distribution_has_low_psi():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(40, 10, 5000)), pd.Series(rng.normal(40, 10, 5000))
    assert psi_numeric(a, b) < 0.02


def test_shifted_distribution_has_high_psi():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(40, 10, 5000)), pd.Series(rng.normal(55, 10, 5000))
    assert psi_numeric(a, b) > 0.2


def test_new_category_counts_as_drift():
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    cur = pd.Series(["a"] * 300 + ["c"] * 700)
    assert psi_categorical(ref, cur) > 0.2


def test_report_flags_only_drifted_features():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"age": rng.normal(40, 10, 3000), "job": rng.choice(["x", "y"], 3000)})
    cur = pd.DataFrame({"age": rng.normal(55, 10, 3000), "job": rng.choice(["x", "y"], 3000)})
    report = drift_report(ref, cur, threshold=0.2)
    assert report["drift_detected"]
    assert report["drifted_features"] == ["age"]
