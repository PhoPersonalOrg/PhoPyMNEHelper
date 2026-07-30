"""Tests for motion bad-interval float duration handling (unix t columns)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phopymnehelper.motion_data import BadMotionDataFrame, MotionData


def _synthetic_motion_df_with_high_accel(*, t0: float = 1_700_000_000.0, n: int = 200, fs: float = 32.0) -> pd.DataFrame:
    t = t0 + np.arange(n, dtype=np.float64) / fs
    # Quiet baseline then oscillating spike so consecutive samples stay above change threshold
    acc = np.zeros((n, 3), dtype=np.float64)
    spike = np.arange(80, 120)
    acc[spike, 0] = np.where(spike % 2 == 0, 5.0, -5.0)
    acc[spike, 1] = np.where(spike % 2 == 0, -5.0, 5.0)
    acc[spike, 2] = np.where(spike % 2 == 0, 5.0, -5.0)
    return pd.DataFrame({"t": t, "AccX": acc[:, 0], "AccY": acc[:, 1], "AccZ": acc[:, 2]})


def test_duration_to_seconds_float_and_timedelta():
    assert BadMotionDataFrame._duration_to_seconds(1.5) == pytest.approx(1.5)
    assert BadMotionDataFrame._duration_to_seconds(pd.Timedelta(seconds=2.25)) == pytest.approx(2.25)


def test_motion_bad_intervals_key_suffix_float_duration():
    df = pd.DataFrame({"onset": [100.0, 200.0], "duration": [1.5, 0.25], "description": ["BAD_motion", "BAD_motion"]})
    suffix = BadMotionDataFrame.motion_bad_intervals_key_suffix(df)
    assert isinstance(suffix, str)
    assert len(suffix) == 12


def test_motion_bad_intervals_key_suffix_after_adding_unix_columns():
    df = pd.DataFrame({"onset": [1_700_000_100.0], "duration": [0.5], "description": ["BAD_motion"]})
    with_unix = df.bad_motion_epochs_df.adding_unix_float_columns(inplace=False)
    assert "t_start" in with_unix.columns and "t_duration" in with_unix.columns
    normalized = BadMotionDataFrame.normalize_motion_bad_intervals_df(with_unix)
    assert list(normalized.columns) == ["t_start", "t_duration"]
    assert float(normalized["t_duration"].iloc[0]) == pytest.approx(0.5)
    suffix = BadMotionDataFrame.motion_bad_intervals_key_suffix(normalized)
    assert len(suffix) == 12


def test_find_high_accel_periods_float_t_and_key_suffix():
    motion_df = _synthetic_motion_df_with_high_accel()
    annots, epochs_df = MotionData.find_high_accel_periods(
        a_ds=motion_df,
        total_change_threshold=0.5,
        should_set_bad_period_annotations=False,
        minimum_bad_duration=None,
    )
    assert annots is not None
    assert "onset" in epochs_df.columns and "duration" in epochs_df.columns
    assert len(epochs_df) >= 1
    # duration from unix float t diffs is numeric, not Timedelta
    assert not pd.api.types.is_timedelta64_dtype(epochs_df["duration"].dtype)

    with_unix = epochs_df.bad_motion_epochs_df.adding_unix_float_columns(inplace=False)
    normalized = BadMotionDataFrame.normalize_motion_bad_intervals_df(with_unix)
    suffix = BadMotionDataFrame.motion_bad_intervals_key_suffix(normalized)
    assert len(suffix) == 12
