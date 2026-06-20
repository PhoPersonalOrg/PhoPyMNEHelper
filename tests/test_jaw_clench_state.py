"""Unit tests for jaw-clench state interval FSM."""

import unittest

import numpy as np
import pandas as pd

from phopymnehelper.analysis.computations.specific.jaw_clench_probability import JAW_CLENCH_PROB_COLUMN
from phopymnehelper.analysis.computations.specific.jaw_clench_state import compute_jaw_clench_state_intervals_from_prob_df, probability_series_to_clench_intervals


def _series_at_hz(values: list, hz: float = 20.0, t0: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    t = t0 + (np.arange(n, dtype=float) / hz)
    return t, np.asarray(values, dtype=float)


class TestJawClenchStateIntervals(unittest.TestCase):
    def test_synthetic_latch_onset_quiet_release(self):
        values = [0.05] * 6 + [0.85] + [0.05] * 20 + [0.50] + [0.05] * 10
        t, prob = _series_at_hz(values, hz=20.0)
        iv = probability_series_to_clench_intervals(t, prob)
        self.assertEqual(len(iv), 1)
        self.assertGreater(float(iv.iloc[0]["t_duration"]), 0.5)
        self.assertGreaterEqual(float(iv.iloc[0]["peak_prob"]), 0.8)
        self.assertGreaterEqual(float(iv.iloc[0]["release_prob"]), 0.45)

    def test_sustain_fade_ignored_single_interval(self):
        values = [0.05] * 4 + [0.90, 0.05] + [0.03] * 40 + [0.48] + [0.05] * 8
        t, prob = _series_at_hz(values, hz=20.0)
        iv = probability_series_to_clench_intervals(t, prob)
        self.assertEqual(len(iv), 1)
        self.assertGreater(float(iv.iloc[0]["t_duration"]), 1.0)

    def test_brief_blip_below_onset_produces_no_interval(self):
        values = [0.05] * 10 + [0.45] + [0.05] * 30
        t, prob = _series_at_hz(values, hz=20.0)
        iv = probability_series_to_clench_intervals(t, prob)
        self.assertEqual(len(iv), 0)

    def test_brief_onset_without_release_cancels_when_short(self):
        values = [0.05] * 6 + [0.85] + [0.05] * 8
        t, prob = _series_at_hz(values, hz=20.0)
        iv = probability_series_to_clench_intervals(t, prob, armed_release_timeout_s=0.15, quiet_min_s=0.10, min_clinch_s=0.40)
        self.assertEqual(len(iv), 0)

    def test_merge_adjacent_intervals(self):
        t1, p1 = _series_at_hz([0.05] * 4 + [0.85] + [0.05] * 8 + [0.50] + [0.05] * 4, hz=20.0, t0=1000.0)
        t2, p2 = _series_at_hz([0.05] * 2 + [0.80] + [0.05] * 8 + [0.55] + [0.05] * 4, hz=20.0, t0=1000.85)
        t = np.concatenate([t1, t2])
        prob = np.concatenate([p1, p2])
        iv = probability_series_to_clench_intervals(t, prob, merge_gap_s=0.50)
        self.assertEqual(len(iv), 1)

    def test_from_prob_df_wrapper(self):
        t, prob = _series_at_hz([0.05] * 6 + [0.85] + [0.05] * 20 + [0.50] + [0.05] * 10)
        df = pd.DataFrame({"t": t, JAW_CLENCH_PROB_COLUMN: prob})
        result = compute_jaw_clench_state_intervals_from_prob_df(df)
        self.assertEqual(int(result["n_intervals"]), 1)
        self.assertIn("intervals_df", result)


if __name__ == "__main__":
    unittest.main()
