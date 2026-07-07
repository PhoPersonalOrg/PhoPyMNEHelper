"""Tests for phopymnehelper.analysis.computations.gfp_band_power."""

import numpy as np
import pandas as pd
import pytest
from phopymnehelper.analysis.computations import gfp_band_power as gfp


def test_global_field_power_matches_manual() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    g = gfp.global_field_power(x)
    np.testing.assert_allclose(g, np.array([10.0, 20.0]))


def test_baseline_rescale() -> None:
    t = np.arange(10, dtype=float)
    y = np.ones(10) * 2.0
    out = gfp.baseline_rescale(y, t, 0.0, 5.0)
    np.testing.assert_allclose(out, np.zeros(10))


def test_get_filter_sos_caches() -> None:
    cache: dict = {}
    a = gfp.get_filter_sos(8.0, 12.0, 256.0, 4, cache)
    b = gfp.get_filter_sos(8.0, 12.0, 256.0, 4, cache)
    assert a is not None and b is not None
    assert a is b
    assert len(cache) == 1


def test_bootstrap_gfp_ci_deterministic() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, 100))
    lo1, hi1 = gfp.bootstrap_gfp_ci(data, 20, rng=np.random.default_rng(42))
    lo2, hi2 = gfp.bootstrap_gfp_ci(data, 20, rng=np.random.default_rng(42))
    np.testing.assert_allclose(lo1, lo2)
    np.testing.assert_allclose(hi1, hi2)


def test_estimate_sample_rate_from_t() -> None:
    t = np.linspace(0.0, 1.0, 129)
    assert abs(gfp.estimate_sample_rate_from_t(t) - 128.0) < 1e-6


def test_dataframe_to_channel_matrix() -> None:
    df = pd.DataFrame({"t": [0.0, 0.01, 0.02], "A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    mat, t = gfp.dataframe_to_channel_matrix(df, ["A", "B"])
    assert mat.shape == (2, 3)
    np.testing.assert_allclose(t, [0.0, 0.01, 0.02])


def test_bandpass_filter_leaves_nyquist_invalid_band() -> None:
    data = np.random.randn(2, 50)
    cache: dict = {}
    out = gfp.bandpass_filter_channels(data, 40.0, 80.0, 64.0, 4, cache)
    np.testing.assert_array_equal(out, data)
