"""Unit tests for jaw-clench probability computation."""

import unittest

import mne
import numpy as np
import pandas as pd

from phopymnehelper.analysis.computations.specific.jaw_clench_probability import compute_jaw_clench_probability_from_detailed_df, compute_jaw_clench_probability_from_raw, compute_jaw_clench_probability_merged_for_intervals, compute_jaw_clench_probability_series


def _make_burst_eeg(n_ch: int = 8, n_samp: int = 1280, fs: float = 128.0, burst_amp: float = 200e-6) -> np.ndarray:
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_ch, n_samp)) * 1e-6
    burst_start = n_samp // 2
    burst_end = burst_start + max(1, int(0.12 * fs))
    data[:, burst_start:burst_end] += burst_amp
    return data


class TestJawClenchProbabilitySeries(unittest.TestCase):
    def test_burst_raises_probability(self):
        fs = 128.0
        data = _make_burst_eeg()
        t_unix = 1000.0 + (np.arange(data.shape[1], dtype=float) / fs)
        times, prob = compute_jaw_clench_probability_series(data, fs, t_unix=t_unix)
        self.assertGreater(times.size, 0)
        self.assertTrue(np.all(prob >= 0.0))
        self.assertTrue(np.all(prob <= 1.0))
        self.assertGreater(float(np.nanmax(prob)), 0.2)

    def test_viewport_filter_subsets_times(self):
        fs = 128.0
        data = _make_burst_eeg()
        t_unix = 1000.0 + (np.arange(data.shape[1], dtype=float) / fs)
        times_all, _ = compute_jaw_clench_probability_series(data, fs, t_unix=t_unix)
        times_vp, prob_vp = compute_jaw_clench_probability_series(data, fs, t_unix=t_unix, viewport_t_min=1002.0, viewport_t_max=1004.0)
        self.assertLessEqual(times_vp.size, times_all.size)
        if times_vp.size:
            self.assertGreaterEqual(float(times_vp.min()), 1002.0)
            self.assertLessEqual(float(times_vp.max()), 1004.0)
            self.assertEqual(prob_vp.size, times_vp.size)

    def test_detailed_df_matches_core(self):
        fs = 128.0
        data = _make_burst_eeg(n_ch=4)
        t_unix = 1000.0 + (np.arange(data.shape[1], dtype=float) / fs)
        ch_names = [f"Ch{i}" for i in range(data.shape[0])]
        df = pd.DataFrame(data.T, columns=ch_names)
        df.insert(0, "t", t_unix)
        out_df = compute_jaw_clench_probability_from_detailed_df(df, ch_names, fs)
        times_core, prob_core = compute_jaw_clench_probability_series(data, fs, t_unix=t_unix)
        np.testing.assert_array_almost_equal(out_df["t"].to_numpy(), times_core)
        np.testing.assert_array_almost_equal(out_df["jaw_clench_prob"].to_numpy(), prob_core)

    def test_from_raw_returns_bounded_probabilities(self):
        fs = 128.0
        data = _make_burst_eeg(n_ch=4)
        info = mne.create_info([f"Ch{i}" for i in range(data.shape[0])], sfreq=fs, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        result = compute_jaw_clench_probability_from_raw(raw)
        prob = np.asarray(result["jaw_clench_prob"], dtype=float)
        self.assertGreater(prob.size, 0)
        self.assertTrue(np.all(prob >= 0.0))
        self.assertTrue(np.all(prob <= 1.0))


class TestJawClenchMergedForIntervals(unittest.TestCase):
    def _make_raw(self, n_samp: int = 1280, fs: float = 128.0, ch_prefix: str = "Ch") -> mne.io.BaseRaw:
        data = _make_burst_eeg(n_ch=4, n_samp=n_samp, fs=fs)
        ch_names = [f"{ch_prefix}{i}" for i in range(data.shape[0])]
        info = mne.create_info(ch_names, sfreq=fs, ch_types="eeg")
        return mne.io.RawArray(data, info, verbose="ERROR")

    def test_merged_stitched_times_overlap_intervals(self):
        fs = 128.0
        t0_a, t0_b = 1_700_000_000.0, 1_700_010_000.0
        dur = 10.0
        raw_a = self._make_raw()
        raw_b = self._make_raw()
        intervals_df = pd.DataFrame({"t_start": [t0_a, t0_b], "t_duration": [dur, dur]})
        result = compute_jaw_clench_probability_merged_for_intervals([raw_a, raw_b], intervals_df)
        times = np.asarray(result["times"], dtype=float)
        self.assertGreater(result["n_windows"], 0)
        self.assertTrue(result.get("times_are_absolute_unix"))
        for _, row in intervals_df.iterrows():
            lo = float(row["t_start"])
            hi = lo + float(row["t_duration"])
            seg_mask = (times >= lo - 1e-3) & (times <= hi + 1e-3)
            self.assertGreater(int(seg_mask.sum()), 0, msg=f"expected windows in interval [{lo}, {hi}]")

    def test_reanchor_when_interval_t_start_misaligned(self):
        fs = 128.0
        true_t0 = 1_700_001_100.0
        wrong_t0 = 1_700_000_100.0
        dur = 2000.0
        raw = self._make_raw()
        n_samp = raw.n_times
        t_vec = true_t0 + (np.arange(n_samp, dtype=float) / fs)
        ch_names = list(raw.ch_names)
        parent_df = pd.DataFrame(raw.get_data().T, columns=ch_names)
        parent_df.insert(0, "t", t_vec)
        intervals_df = pd.DataFrame({"t_start": [wrong_t0], "t_duration": [dur]})
        result_wrong = compute_jaw_clench_probability_merged_for_intervals([raw], intervals_df)
        times_wrong = np.asarray(result_wrong["times"], dtype=float)
        self.assertFalse(np.any((times_wrong >= true_t0 - 1e-3) & (times_wrong <= true_t0 + 20.0)))
        result_fixed = compute_jaw_clench_probability_merged_for_intervals([raw], intervals_df, parent_detailed_df=parent_df)
        times_fixed = np.asarray(result_fixed["times"], dtype=float)
        self.assertGreater(result_fixed["n_windows"], 0)
        self.assertTrue(np.any((times_fixed >= true_t0 - 1e-3) & (times_fixed <= true_t0 + 20.0)))

    def test_channel_names_fallback_when_no_eeg_pick_types(self):
        fs = 128.0
        data = _make_burst_eeg(n_ch=4)
        ch_names = [f"EEG{i}" for i in range(data.shape[0])]
        info = mne.create_info(ch_names, sfreq=fs, ch_types="misc")
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        empty = compute_jaw_clench_probability_from_raw(raw)
        self.assertEqual(int(empty["n_windows"]), 0)
        with_names = compute_jaw_clench_probability_from_raw(raw, channel_names=ch_names)
        self.assertGreater(int(with_names["n_windows"]), 0)


if __name__ == "__main__":
    unittest.main()
