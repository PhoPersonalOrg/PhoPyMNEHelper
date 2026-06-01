"""Unit tests for theta/delta absolute-time alignment helpers (sleep-intrusion track)."""

import unittest

import numpy as np
import pandas as pd

from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import interval_row_t_start_unix_seconds, raw_relative_times_to_timeline_unix


class TestThetaDeltaTimelineUnix(unittest.TestCase):
    def test_raw_relative_to_timeline(self):
        out = raw_relative_times_to_timeline_unix(np.array([0.5, 1.5]), interval_t_start_unix=1000.0, raw_times_first=0.0)
        np.testing.assert_array_almost_equal(out, np.array([1000.5, 1001.5]))

    def test_raw_relative_offset_first_sample(self):
        out = raw_relative_times_to_timeline_unix(np.array([10.5, 11.5]), interval_t_start_unix=1704067300.0, raw_times_first=10.0)
        np.testing.assert_array_almost_equal(out, np.array([1704067300.5, 1704067301.5]))

    def test_interval_row_t_start_numeric(self):
        iv = pd.Series({"t_start": 1704067200.0, "t_duration": 1.0})
        self.assertAlmostEqual(interval_row_t_start_unix_seconds(iv), 1704067200.0)

    def test_interval_row_t_start_timestamp(self):
        iv = pd.Series({"t_start": pd.Timestamp("2024-01-01T12:00:00Z"), "t_duration": 60.0})
        self.assertAlmostEqual(interval_row_t_start_unix_seconds(iv), pd.Timestamp("2024-01-01T12:00:00Z").timestamp())


if __name__ == "__main__":
    unittest.main()
