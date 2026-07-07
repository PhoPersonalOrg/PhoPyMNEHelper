"""Unit tests for Frame-style EEG mental states computation."""

import unittest

import numpy as np
import pandas as pd

from phopymnehelper.analysis.computations.specific.mental_states import (
    MENTAL_STATE_COLUMNS,
    MentalStatesRollingState,
    _mental_state_values,
    compute_frame_mental_states_from_detailed_df,
    compute_frame_mental_states_series,
)


def _sine_band(freq: float, sfreq: float, n_samples: int, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n_samples, dtype=float) / sfreq
    return amplitude * np.sin(2.0 * np.pi * freq * t)


class TestMentalStatesComputation(unittest.TestCase):
    def test_rolling_norm_first_sample_is_zero(self):
        state = MentalStatesRollingState()
        db = {b: 60.0 for b in ("Delta", "Theta", "Alpha", "Beta", "Gamma")}
        vals = _mental_state_values(db, state)
        for c in MENTAL_STATE_COLUMNS:
            self.assertAlmostEqual(vals[c], 0.0, places=5)

    def test_rolling_norm_bounded_after_fill(self):
        state = MentalStatesRollingState()
        db = {b: 60.0 for b in ("Delta", "Theta", "Alpha", "Beta", "Gamma")}
        for alpha_db in np.linspace(55.0, 75.0, 50):
            db_mut = dict(db)
            db_mut["Alpha"] = alpha_db
            vals = _mental_state_values(db_mut, state)
        for c in MENTAL_STATE_COLUMNS:
            self.assertGreaterEqual(vals[c], 0.0)
            self.assertLessEqual(vals[c], 100.0 + 1e-6)

    def test_series_returns_four_columns(self):
        sfreq = 128.0
        n = int(sfreq * 3)
        alpha_sig = _sine_band(10.0, sfreq, n, amplitude=2.0)
        data = np.stack([alpha_sig, alpha_sig * 0.5], axis=0)
        result = compute_frame_mental_states_series(data, sfreq, window_sec=1.0, step_sec=0.5)
        self.assertGreater(result["n_windows"], 0)
        for c in MENTAL_STATE_COLUMNS:
            self.assertEqual(result[c].shape, result["times"].shape)

    def test_from_detailed_df_columns(self):
        sfreq = 128.0
        n = int(sfreq * 2.5)
        t = np.arange(n, dtype=float) / sfreq + 1000.0
        ch_a = _sine_band(10.0, sfreq, n)
        ch_b = _sine_band(6.0, sfreq, n)
        df = pd.DataFrame({"t": t, "AF3": ch_a, "F3": ch_b})
        out = compute_frame_mental_states_from_detailed_df(df, ["AF3", "F3"], sfreq, window_sec=1.0, step_sec=0.5, incremental=False)
        self.assertGreater(len(out), 0)
        for c in MENTAL_STATE_COLUMNS:
            self.assertIn(c, out.columns)
        self.assertTrue((out["relaxation"] >= 0).all())
        self.assertTrue((out["relaxation"] <= 100).all())

    def test_incremental_skips_processed_windows(self):
        sfreq = 128.0
        n = int(sfreq * 3)
        sig = _sine_band(10.0, sfreq, n)
        data = np.stack([sig, sig], axis=0)
        t_unix = np.arange(n, dtype=float) / sfreq + 5000.0
        state = MentalStatesRollingState()
        r1 = compute_frame_mental_states_series(data, sfreq, t_unix=t_unix, window_sec=1.0, step_sec=0.5, state=state, incremental=False)
        n1 = r1["times"].size
        self.assertGreater(n1, 0)
        r2 = compute_frame_mental_states_series(data, sfreq, t_unix=t_unix, window_sec=1.0, step_sec=0.5, state=state, incremental=True)
        self.assertEqual(r2["times"].size, 0)


if __name__ == "__main__":
    unittest.main()
