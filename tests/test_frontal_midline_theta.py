import time

import mne
import numpy as np
import pandas as pd
import pytest

from phopymnehelper.analysis.computations.specific.frontal_midline_theta import (
    compute_frontal_midline_theta_merged_for_intervals,
    compute_frontal_midline_theta_series,
    filter_frontal_midline_theta_params,
    frontal_midline_theta_params_fingerprint,
    resolve_fmt_channel_picks,
)


def _make_raw(ch_names, duration_sec=10.0, sfreq=256.0, freq_hz=7.0):
    n = int(sfreq * duration_sec)
    t = np.arange(n) / sfreq
    data = np.stack([np.sin(2 * np.pi * freq_hz * t) for _ in ch_names], axis=0)
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    return mne.io.RawArray(data, info)


def test_compute_frontal_midline_theta_series_keys_and_finite():
    raw = _make_raw(["AF3", "AF4"])
    result = compute_frontal_midline_theta_series(raw, window_sec=2.0, step_sec=1.0)

    assert result is not None
    assert "times" in result
    assert "fmt_power" in result
    assert "session_mean_fmt_power" in result
    assert "n_windows" in result
    assert "n_valid_windows" in result
    assert "params" in result
    assert result["n_windows"] > 0
    assert result["n_valid_windows"] > 0
    assert np.sum(np.isfinite(result["fmt_power"])) > 0
    assert set(result["params"]["fmt_channels"]) == {"AF3", "AF4"}
    assert result["params"].get("compute_engine") == "stft_spectrogram"


def test_resolve_fmt_channel_picks_prefers_fz():
    raw = _make_raw(["AF3", "AF4", "Fz", "O1"])
    picks, names = resolve_fmt_channel_picks(raw)
    assert names == ["Fz"]
    assert picks.size == 1


def test_resolve_fmt_channel_picks_af_pair():
    raw = _make_raw(["AF3", "AF4", "O1", "O2"])
    picks, names = resolve_fmt_channel_picks(raw)
    assert names == ["AF3", "AF4"]
    assert picks.size == 2


def test_resolve_fmt_channel_picks_fallback():
    raw = _make_raw(["F3", "O1", "O2"])
    picks, names = resolve_fmt_channel_picks(raw)
    assert names == ["F3"]
    assert picks.size == 1


def test_filter_and_fingerprint_strip_unknown_keys():
    params = {"window_sec": 4.0, "step_sec": 1.0, "unknown_key": 123, "fmt_band": (5.0, 9.0)}
    filtered = filter_frontal_midline_theta_params(params)
    assert "unknown_key" not in filtered
    assert filtered["window_sec"] == 4.0
    assert filtered["fmt_band"] == (5.0, 9.0)

    fp1 = frontal_midline_theta_params_fingerprint({"window_sec": 4.0, "step_sec": 1.0})
    fp2 = frontal_midline_theta_params_fingerprint({"step_sec": 1.0, "window_sec": 4.0})
    assert fp1 == fp2


def test_compute_frontal_midline_theta_series_performance_budget():
    raw = _make_raw(["AF3", "AF4"], duration_sec=600.0, sfreq=256.0)
    t0 = time.monotonic()
    result = compute_frontal_midline_theta_series(raw, window_sec=4.0, step_sec=1.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"FMT compute took {elapsed:.2f}s (budget 3s)"
    assert result["n_windows"] > 0
    assert result["n_valid_windows"] > 0


def test_merge_with_motion_df_slice():
    raw_a = _make_raw(["AF3", "AF4"], duration_sec=30.0)
    raw_b = _make_raw(["AF3", "AF4"], duration_sec=30.0)
    intervals_df = pd.DataFrame(
        {
            "t_start": [1000.0, 1100.0],
            "t_duration": [30.0, 30.0],
        }
    )
    motion_df = pd.DataFrame(
        {
            "t": np.linspace(1000.0, 1130.0, 500),
            "AccX": np.zeros(500),
            "AccY": np.zeros(500),
            "AccZ": np.ones(500) * 0.01,
        }
    )
    result = compute_frontal_midline_theta_merged_for_intervals(
        [raw_a, raw_b],
        intervals_df,
        motion_df=motion_df,
    )
    assert result["times_are_absolute_unix"] is True
    assert result["merged_n_segments"] == 2
    assert len(result["times"]) == len(result["fmt_power"])
    assert result["n_windows"] > 0
