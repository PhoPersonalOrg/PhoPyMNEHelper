"""Frame-style EEG mental states from band-power ratios (relaxation, focus, stress, drowsiness).

Port of band-ratio logic from frame-eeg: per-window band-pass variance power, dB session
min-max normalization (50–100), then log-ratio rolling min-max scaled to 0–100%.

Public surfaces:

- :class:`FrameMentalStatesComputation` -- DAG node and compute classmethods
- :func:`compute_frame_mental_states_series` -- numpy core (live + historical)
- :func:`compute_frame_mental_states_from_detailed_df` -- timeline ``detailed_df`` adapter
- :func:`compute_frame_mental_states_from_raw` -- one MNE raw segment
- :func:`compute_frame_mental_states_merged_for_intervals` -- multi-raw stitch (absolute unix)
- :func:`apply_frame_mental_states_to_timeline` -- add/refresh mental-states track
- :func:`compute_frame_mental_states_for_eeg_datasource` -- align raws from a timeline EEG datasource
- :func:`start_frame_mental_states_track_background` -- non-blocking batch compute + Qt main-thread apply
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar, Deque, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd

from phopymnehelper.analysis.computations.gfp_band_power import bandpass_filter_channels
from phopymnehelper.analysis.computations.protocol import ArtifactKind, RunContext
from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import (
    interval_row_t_start_unix_seconds,
    intervals_df_for_theta_delta_track,
    raw_relative_times_to_timeline_unix,
)
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase

logger = logging.getLogger(__name__)

FRAME_MENTAL_STATE_BANDS: Dict[str, Tuple[float, float]] = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (12.0, 30.0),
    "Gamma": (30.0, 50.0),
}

MENTAL_STATE_RELAXATION: str = "relaxation"
MENTAL_STATE_FOCUS: str = "focus"
MENTAL_STATE_STRESS: str = "stress"
MENTAL_STATE_DROWSINESS: str = "drowsiness"

MENTAL_STATE_COLUMNS: Tuple[str, ...] = (
    MENTAL_STATE_RELAXATION,
    MENTAL_STATE_FOCUS,
    MENTAL_STATE_STRESS,
    MENTAL_STATE_DROWSINESS,
)

DEFAULT_WINDOW_SEC: float = 1.0
DEFAULT_STEP_SEC: float = 0.1
DEFAULT_ROLLING_NORM_WINDOW: int = 100
DEFAULT_FILTER_ORDER: int = 4

MENTAL_STATE_LINE_COLORS: Dict[str, str] = {
    MENTAL_STATE_RELAXATION: "#4488ff",
    MENTAL_STATE_FOCUS: "#44cc44",
    MENTAL_STATE_STRESS: "#ff6644",
    MENTAL_STATE_DROWSINESS: "#aa44cc",
}

MENTAL_STATE_LINE_WIDTH: float = 0.6

MENTAL_STATE_DISPLAY_NAMES: Dict[str, str] = {
    MENTAL_STATE_RELAXATION: "relaxation",
    MENTAL_STATE_FOCUS: "focus",
    MENTAL_STATE_STRESS: "stress",
    MENTAL_STATE_DROWSINESS: "drowsiness",
}


@dataclass
class MentalStatesRollingState:
    """Mutable rolling state for band dB min-max and ratio deques."""

    band_db_minmax: Dict[str, List[float]] = field(default_factory=dict)
    alpha_beta_log_ratios: Deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_ROLLING_NORM_WINDOW))
    beta_theta_focus_ratios: Deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_ROLLING_NORM_WINDOW))
    beta_alpha_stress_ratios: Deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_ROLLING_NORM_WINDOW))
    delta_alpha_drowsiness_ratios: Deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_ROLLING_NORM_WINDOW))
    last_center_t: Optional[float] = None
    rolling_norm_window: int = DEFAULT_ROLLING_NORM_WINDOW

    def reset(self) -> None:
        self.band_db_minmax.clear()
        self.alpha_beta_log_ratios.clear()
        self.beta_theta_focus_ratios.clear()
        self.beta_alpha_stress_ratios.clear()
        self.delta_alpha_drowsiness_ratios.clear()
        self.last_center_t = None

    def _ensure_band_db_entry(self, band_name: str) -> List[float]:
        if band_name not in self.band_db_minmax:
            self.band_db_minmax[band_name] = [float(np.inf), float(-np.inf)]
        return self.band_db_minmax[band_name]

    def _db_normalize_band(self, band_name: str, power: float) -> float:
        """Convert power to dB and map to 50–100 using session-expanded min-max."""
        dB_value = 10.0 * np.log10(max(power, 0.0) + 1e-6)
        min_val, max_val = self._ensure_band_db_entry(band_name)
        min_val = min(min_val, dB_value)
        max_val = max(max_val, dB_value)
        self.band_db_minmax[band_name] = [min_val, max_val]
        if max_val > min_val:
            return 50.0 + 50.0 * (dB_value - min_val) / (max_val - min_val)
        return 50.0

    @staticmethod
    def _rolling_norm_100(log_ratio: float, ratio_deque: Deque[float]) -> float:
        ratio_deque.append(float(log_ratio))
        rmin = min(ratio_deque)
        rmax = max(ratio_deque)
        return (log_ratio - rmin) / (rmax - rmin + 1e-6) * 100.0

    def mental_state_values(self, db_band_powers: Mapping[str, float]) -> Dict[str, float]:
        """Compute four mental-state percentages from normalized dB band powers."""
        alpha_power = 10.0 ** (db_band_powers["Alpha"] / 10.0)
        beta_power = 10.0 ** (db_band_powers["Beta"] / 10.0)
        gamma_power = 10.0 ** (db_band_powers["Gamma"] / 10.0)
        delta_power = 10.0 ** (db_band_powers["Delta"] / 10.0)
        theta_power = 10.0 ** (db_band_powers["Theta"] / 10.0)

        alpha_beta_ratio = alpha_power / (beta_power + 1e-6)
        log_alpha_beta = np.log1p(alpha_beta_ratio)
        relaxation = self._rolling_norm_100(log_alpha_beta, self.alpha_beta_log_ratios)

        beta_theta_ratio = beta_power / (theta_power + 1e-6)
        log_beta_theta = np.log1p(beta_theta_ratio)
        focus = self._rolling_norm_100(log_beta_theta, self.beta_theta_focus_ratios)

        combined_beta_gamma = (beta_power / (alpha_power + 1e-6)) * (gamma_power + 1e-6)
        log_stress = np.log1p(combined_beta_gamma)
        stress = self._rolling_norm_100(log_stress, self.beta_alpha_stress_ratios)

        delta_alpha_ratio = delta_power / (alpha_power + 1e-6)
        log_delta_alpha = np.log1p(delta_alpha_ratio)
        drowsiness = self._rolling_norm_100(log_delta_alpha, self.delta_alpha_drowsiness_ratios)

        return {
            MENTAL_STATE_RELAXATION: float(relaxation),
            MENTAL_STATE_FOCUS: float(focus),
            MENTAL_STATE_STRESS: float(stress),
            MENTAL_STATE_DROWSINESS: float(drowsiness),
        }


class FrameMentalStatesComputation(SpecificComputationBase):
    computation_id: ClassVar[str] = "mental_states"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ()
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.stream

    @staticmethod
    def _merged_power_from_filtered_block(block_2d: np.ndarray) -> float:
        """Mean channel variance on already band-pass filtered data."""
        if block_2d.size == 0:
            return float("nan")
        per_ch = np.nanvar(block_2d, axis=1)
        if not np.any(np.isfinite(per_ch)):
            return float("nan")
        return float(np.nanmean(per_ch))

    @classmethod
    def _band_merged_power(
        cls,
        block_2d: np.ndarray,
        fmin: float,
        fmax: float,
        sfreq: float,
        filter_order: int,
        cache: MutableMapping[Tuple[float, float, float], np.ndarray],
    ) -> float:
        """Band-pass each channel, variance per channel, mean across channels."""
        filtered = bandpass_filter_channels(block_2d, fmin, fmax, sfreq, filter_order, cache)
        return cls._merged_power_from_filtered_block(filtered)

    @classmethod
    def _prefilter_mental_state_bands(
        cls,
        data: np.ndarray,
        sfreq: float,
        filter_order: int,
        cache: MutableMapping[Tuple[float, float, float], np.ndarray],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Band-pass filter the full segment once per mental-state band."""
        sf = float(sfreq)
        band_filtered: Dict[str, np.ndarray] = {}
        for band_name, (fmin, fmax) in FRAME_MENTAL_STATE_BANDS.items():
            if fmin >= sf / 2.0:
                return None
            band_filtered[band_name] = bandpass_filter_channels(data, fmin, fmax, sf, filter_order, cache)
        return band_filtered

    @classmethod
    def compute_frame_mental_states_series(
        cls,
        samples_2d: np.ndarray,
        sfreq: float,
        t_unix: Optional[np.ndarray] = None,
        *,
        window_sec: float = DEFAULT_WINDOW_SEC,
        step_sec: float = DEFAULT_STEP_SEC,
        filter_order: int = DEFAULT_FILTER_ORDER,
        state: Optional[MentalStatesRollingState] = None,
        viewport_t_min: Optional[float] = None,
        viewport_t_max: Optional[float] = None,
        incremental: bool = True,
    ) -> Dict[str, Any]:
        """Compute sliding-window mental states from multi-channel EEG samples using Polars and vectorization.

        Parameters
        ----------
        samples_2d
            Shape ``(n_channels, n_samples)``.
        sfreq
            Sample rate in Hz.
        t_unix
            Optional per-sample unix timestamps (length ``n_samples``). Window centers use
            ``t_unix[center_idx]`` when provided; otherwise centers are raw-relative seconds.
        state
            Optional rolling state (mutated in place). When ``incremental`` and ``state.last_center_t``
            is set, windows at or before that center time are skipped.
        """
        import polars as pl

        data = np.asarray(samples_2d, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("samples_2d must be 2D (n_channels, n_samples)")
        n_ch, n_samp = data.shape
        sf = float(sfreq)

        if state is None:
            state = MentalStatesRollingState()
        else:
            for dq in (
                state.alpha_beta_log_ratios,
                state.beta_theta_focus_ratios,
                state.beta_alpha_stress_ratios,
                state.delta_alpha_drowsiness_ratios,
            ):
                if dq.maxlen != state.rolling_norm_window:
                    dq = deque(dq, maxlen=state.rolling_norm_window)

        w = max(1, int(round(window_sec * sf)))
        s = max(1, int(round(step_sec * sf)))

        if sf <= 0.0 or n_ch == 0 or n_samp < w:
            empty = np.array([], dtype=float)
            return dict(
                times=empty,
                relaxation=empty,
                focus=empty,
                stress=empty,
                drowsiness=empty,
                state=state,
                n_windows=0,
            )

        if t_unix is not None:
            t_arr = np.asarray(t_unix, dtype=float)
            if t_arr.size != n_samp:
                raise ValueError(f"t_unix length ({t_arr.size}) must match n_samples ({n_samp})")
        else:
            t_arr = np.arange(n_samp, dtype=float) / sf

        # 1. Calculate window start indices
        starts = np.arange(0, n_samp - w + 1, s)
        centers = starts + w // 2
        times = t_arr[centers]

        # Filter incrementally
        if incremental and state.last_center_t is not None:
            valid_mask = times > state.last_center_t + 1e-9
            starts = starts[valid_mask]
            centers = centers[valid_mask]
            times = times[valid_mask]

        n_windows = len(starts)
        if n_windows == 0:
            empty = np.array([], dtype=float)
            return dict(
                times=empty,
                relaxation=empty,
                focus=empty,
                stress=empty,
                drowsiness=empty,
                state=state,
                n_windows=0,
            )

        start_offset = starts[0]

        # 2. Prefilter
        filter_cache: Dict[Tuple[float, float, float], np.ndarray] = {}
        band_filtered = cls._prefilter_mental_state_bands(data, sf, filter_order, filter_cache)
        if band_filtered is None:
            empty = np.array([], dtype=float)
            return dict(times=empty, relaxation=empty, focus=empty, stress=empty, drowsiness=empty, state=state, n_windows=0)

        # 3. Vectorized variance computation per band
        db_powers_dict = {}
        skip_mask = np.zeros(n_windows, dtype=bool)

        for band_name in FRAME_MENTAL_STATE_BANDS:
            filtered = band_filtered[band_name]

            # Slice filtered to align with starts[0]
            filtered = filtered[:, start_offset:]

            shape = (n_ch, n_windows, w)
            strides = (filtered.strides[0], filtered.strides[1] * s, filtered.strides[1])
            blocks_3d = np.lib.stride_tricks.as_strided(filtered, shape=shape, strides=strides)

            # blocks_3d shape: (n_channels, n_windows, window_samples)
            # Variance along axis=2
            var = np.nanvar(blocks_3d, axis=2)
            has_finite = np.any(np.isfinite(var), axis=0)

            mean_var = np.full(n_windows, np.nan)
            mean_var[has_finite] = np.nanmean(var[:, has_finite], axis=0)

            db_powers_dict[band_name] = mean_var
            skip_mask |= ~np.isfinite(mean_var)

        # Filter out skipped windows
        valid_windows = ~skip_mask
        if not np.any(valid_windows):
            empty = np.array([], dtype=float)
            return dict(times=empty, relaxation=empty, focus=empty, stress=empty, drowsiness=empty, state=state, n_windows=0)

        times = times[valid_windows]
        n_windows = len(times)
        for band_name in FRAME_MENTAL_STATE_BANDS:
            db_powers_dict[band_name] = db_powers_dict[band_name][valid_windows]

        # 4. Polars logic for db normalize and mental state
        initial_mins = {}
        initial_maxs = {}
        for band_name in FRAME_MENTAL_STATE_BANDS:
            if band_name in state.band_db_minmax:
                initial_mins[band_name] = state.band_db_minmax[band_name][0]
                initial_maxs[band_name] = state.band_db_minmax[band_name][1]
            else:
                initial_mins[band_name] = np.inf
                initial_maxs[band_name] = -np.inf

        df = pl.DataFrame(db_powers_dict)
        powers = df.select([pl.col(c).clip(lower_bound=0.0) + 1e-6 for c in df.columns])
        dbs = powers.select([(10.0 * pl.col(c).log10()).alias(c) for c in df.columns])

        prepend_mins = {c: [initial_mins[c]] for c in dbs.columns}
        prepend_maxs = {c: [initial_maxs[c]] for c in dbs.columns}

        dbs_mins_prepended = pl.concat([pl.DataFrame(prepend_mins), dbs])
        dbs_maxs_prepended = pl.concat([pl.DataFrame(prepend_maxs), dbs])

        cum_mins = dbs_mins_prepended.select([pl.col(c).cum_min().alias(c) for c in dbs.columns]).slice(1)
        cum_maxs = dbs_maxs_prepended.select([pl.col(c).cum_max().alias(c) for c in dbs.columns]).slice(1)

        dbs = dbs.with_columns([
            cum_mins[c].alias(f"{c}_min") for c in dbs.columns
        ]).with_columns([
            cum_maxs[c].alias(f"{c}_max") for c in dbs.columns
        ])

        norm_exprs = []
        for c in FRAME_MENTAL_STATE_BANDS:
            norm_col = pl.when(pl.col(f"{c}_max") > pl.col(f"{c}_min")).then(
                50.0 + 50.0 * (pl.col(c) - pl.col(f"{c}_min")) / (pl.col(f"{c}_max") - pl.col(f"{c}_min"))
            ).otherwise(50.0).alias(c)
            norm_exprs.append(norm_col)

        df_norm = dbs.select(norm_exprs)
        powers_from_norm = df_norm.select([(10.0 ** (pl.col(c) / 10.0)).alias(c) for c in df_norm.columns])

        alpha = pl.col("Alpha")
        beta = pl.col("Beta")
        gamma = pl.col("Gamma")
        delta = pl.col("Delta")
        theta = pl.col("Theta")

        log_alpha_beta = (alpha / (beta + 1e-6)).log1p().alias(MENTAL_STATE_RELAXATION)
        log_beta_theta = (beta / (theta + 1e-6)).log1p().alias(MENTAL_STATE_FOCUS)
        log_stress = ((beta / (alpha + 1e-6)) * (gamma + 1e-6)).log1p().alias(MENTAL_STATE_STRESS)
        log_delta_alpha = (delta / (alpha + 1e-6)).log1p().alias(MENTAL_STATE_DROWSINESS)

        df_logs = powers_from_norm.select([log_alpha_beta, log_beta_theta, log_stress, log_delta_alpha])

        rolling_window = state.rolling_norm_window
        prepends = {
            MENTAL_STATE_RELAXATION: list(state.alpha_beta_log_ratios),
            MENTAL_STATE_FOCUS: list(state.beta_theta_focus_ratios),
            MENTAL_STATE_STRESS: list(state.beta_alpha_stress_ratios),
            MENTAL_STATE_DROWSINESS: list(state.delta_alpha_drowsiness_ratios)
        }

        pad_len = len(prepends[MENTAL_STATE_RELAXATION])
        if pad_len > 0:
            prepend_df = pl.DataFrame(prepends)
            df_logs_padded = pl.concat([prepend_df, df_logs])
        else:
            df_logs_padded = df_logs

        df_final_padded = df_logs_padded.select([
            ((pl.col(c) - pl.col(c).rolling_min(window_size=rolling_window, min_samples=1)) /
            (pl.col(c).rolling_max(window_size=rolling_window, min_samples=1) - pl.col(c).rolling_min(window_size=rolling_window, min_samples=1) + 1e-6) * 100.0).alias(c)
            for c in df_logs_padded.columns
        ])

        if pad_len > 0:
            df_final = df_final_padded.slice(pad_len)
        else:
            df_final = df_final_padded

        # Update state
        last_mins = cum_mins.tail(1).to_dict(as_series=False)
        last_maxs = cum_maxs.tail(1).to_dict(as_series=False)
        for c in FRAME_MENTAL_STATE_BANDS:
            state.band_db_minmax[c] = [last_mins[c][0], last_maxs[c][0]]

        state.alpha_beta_log_ratios.extend(df_logs[MENTAL_STATE_RELAXATION].to_list())
        state.beta_theta_focus_ratios.extend(df_logs[MENTAL_STATE_FOCUS].to_list())
        state.beta_alpha_stress_ratios.extend(df_logs[MENTAL_STATE_STRESS].to_list())
        state.delta_alpha_drowsiness_ratios.extend(df_logs[MENTAL_STATE_DROWSINESS].to_list())

        state.last_center_t = float(times[-1])

        relaxation_out = df_final[MENTAL_STATE_RELAXATION].to_numpy()
        focus_out = df_final[MENTAL_STATE_FOCUS].to_numpy()
        stress_out = df_final[MENTAL_STATE_STRESS].to_numpy()
        drowsiness_out = df_final[MENTAL_STATE_DROWSINESS].to_numpy()

        if viewport_t_min is not None and viewport_t_max is not None and times.size:
            lo, hi = float(viewport_t_min), float(viewport_t_max)
            if lo > hi:
                lo, hi = hi, lo
            mask = (times >= lo) & (times <= hi)
            times = times[mask]
            relaxation_out = relaxation_out[mask]
            focus_out = focus_out[mask]
            stress_out = stress_out[mask]
            drowsiness_out = drowsiness_out[mask]

        return dict(
            times=times,
            relaxation=relaxation_out,
            focus=focus_out,
            stress=stress_out,
            drowsiness=drowsiness_out,
            state=state,
            n_windows=int(times.size),
        )

    @classmethod
    def compute_frame_mental_states_from_detailed_df(
        cls,
        df: pd.DataFrame,
        channel_names: Sequence[str],
        sfreq: float,
        *,
        t_col: str = "t",
        viewport_t_min: Optional[float] = None,
        viewport_t_max: Optional[float] = None,
        state: Optional[MentalStatesRollingState] = None,
        incremental: bool = True,
        **compute_kw: Any,
    ) -> pd.DataFrame:
        """Extract EEG channels from a timeline ``detailed_df`` and return mental-state columns."""
        cols = list(MENTAL_STATE_COLUMNS)
        empty = pd.DataFrame({t_col: [], **{c: [] for c in cols}})
        if df is None or len(df) == 0 or t_col not in df.columns:
            return empty
        available = [c for c in channel_names if c in df.columns]
        if not available:
            return empty
        df_sorted = df.sort_values(t_col).reset_index(drop=True)
        t_unix = df_sorted[t_col].to_numpy(dtype=float)
        samples_2d = df_sorted[available].to_numpy(dtype=np.float64).T
        result = cls.compute_frame_mental_states_series(
            samples_2d,
            float(sfreq),
            t_unix=t_unix,
            state=state,
            viewport_t_min=viewport_t_min,
            viewport_t_max=viewport_t_max,
            incremental=incremental,
            **compute_kw,
        )
        times = result["times"]
        if times.size == 0:
            return empty
        out = pd.DataFrame({t_col: times})
        for c in cols:
            out[c] = result[c]
        return out

    @classmethod
    def compute_frame_mental_states_from_raw(
        cls,
        raw_eeg: mne.io.BaseRaw,
        *,
        picks: str = "eeg",
        channel_names: Optional[Sequence[str]] = None,
        window_sec: float = DEFAULT_WINDOW_SEC,
        step_sec: float = DEFAULT_STEP_SEC,
        filter_order: int = DEFAULT_FILTER_ORDER,
        state: Optional[MentalStatesRollingState] = None,
        copy_raw: bool = False,
    ) -> Dict[str, Any]:
        """Compute mental states for one continuous raw segment (raw-relative window centers)."""
        raw = raw_eeg.copy() if copy_raw else raw_eeg
        raw.load_data()
        picks_idx = mne.pick_types(raw.info, eeg=True) if picks == "eeg" else mne.pick_channels(raw.info["ch_names"], include=picks)
        if picks_idx.size == 0 and channel_names:
            available = [c for c in channel_names if c in raw.ch_names]
            if available:
                picks_idx = mne.pick_channels(raw.info["ch_names"], include=available)
        if picks_idx.size == 0:
            empty = np.array([], dtype=float)
            return dict(
                times=empty,
                relaxation=empty,
                focus=empty,
                stress=empty,
                drowsiness=empty,
                state=state or MentalStatesRollingState(),
                n_windows=0,
                params=dict(window_sec=window_sec, step_sec=step_sec),
            )
        data = raw.get_data(picks=picks_idx, reject_by_annotation="omit")
        sfreq = float(raw.info["sfreq"])
        result = cls.compute_frame_mental_states_series(
            data,
            sfreq,
            t_unix=None,
            window_sec=window_sec,
            step_sec=step_sec,
            filter_order=filter_order,
            state=state,
            incremental=False,
        )
        if result["times"].size:
            result = dict(result)
            result["times"] = result["times"] + float(raw.times[0])
        result["params"] = dict(window_sec=window_sec, step_sec=step_sec, filter_order=filter_order, sfreq=sfreq)
        return result

    @classmethod
    def compute_frame_mental_states_merged_for_intervals(
        cls,
        raws: Sequence[mne.io.BaseRaw],
        intervals_df: pd.DataFrame,
        **compute_kw: Any,
    ) -> Dict[str, Any]:
        """Run per-raw compute and stitch into one absolute-time series with shared rolling state."""
        n_iv = len(intervals_df)
        n_raw = len(raws)
        n = min(n_raw, n_iv)
        if n_raw != n_iv:
            logger.warning("mental_states merge: raw count (%s) != interval count (%s); computing %s aligned segment(s).", n_raw, n_iv, n)
        if n == 0:
            raise ValueError("compute_frame_mental_states_merged_for_intervals: no raw/interval pairs (empty inputs).")

        state = MentalStatesRollingState()
        all_t: List[np.ndarray] = []
        series_cols: Dict[str, List[np.ndarray]] = {c: [] for c in MENTAL_STATE_COLUMNS}
        last_params: Optional[Dict[str, Any]] = None

        for i in range(n):
            raw = raws[i]
            iv = intervals_df.iloc[i]
            anchor = interval_row_t_start_unix_seconds(iv)
            raw_t0 = float(np.asarray(raw.times)[0]) if len(raw.times) > 0 else 0.0
            sub = cls.compute_frame_mental_states_from_raw(raw, state=state, **compute_kw)
            state = sub["state"]
            t_rel = np.asarray(sub["times"], dtype=float)
            t_abs = raw_relative_times_to_timeline_unix(t_rel, interval_t_start_unix=anchor, raw_times_first=raw_t0)
            all_t.append(t_abs)
            for c in MENTAL_STATE_COLUMNS:
                series_cols[c].append(np.asarray(sub[c], dtype=float))
            last_params = sub.get("params")

        times_out = np.concatenate(all_t) if len(all_t) > 1 else all_t[0]
        out_series = {c: (np.concatenate(series_cols[c]) if len(series_cols[c]) > 1 else series_cols[c][0]) for c in MENTAL_STATE_COLUMNS}
        n_valid = int(times_out.size)

        return dict(
            times=times_out,
            relaxation=out_series[MENTAL_STATE_RELAXATION],
            focus=out_series[MENTAL_STATE_FOCUS],
            stress=out_series[MENTAL_STATE_STRESS],
            drowsiness=out_series[MENTAL_STATE_DROWSINESS],
            n_windows=n_valid,
            params=last_params or {},
            times_are_absolute_unix=True,
            merged_n_segments=n,
            state=state,
        )

    @classmethod
    def mental_states_colored_title_html(cls) -> str:
        """HTML plot title with each mental-state name in its lane color."""
        parts = [
            f"<span style='color: {MENTAL_STATE_LINE_COLORS[key]}'>{MENTAL_STATE_DISPLAY_NAMES[key]}</span>"
            for key in MENTAL_STATE_COLUMNS
        ]
        return "EEG mental states (" + " / ".join(parts) + ")"

    @classmethod
    def _apply_mental_states_plot_chrome(cls, ms_plot_item: Any, *, y_max: float) -> None:
        """Hide axes and apply the colored multi-lane title."""
        ms_plot_item.setTitle(cls.mental_states_colored_title_html())
        ms_plot_item.setYRange(0, y_max, padding=0.02)
        ms_plot_item.hideAxis("left")
        ms_plot_item.hideAxis("bottom")

    @classmethod
    def _result_to_detailed_df(cls, result: Mapping[str, Any], eeg_ds: Any, track_key: str, *, t0: Optional[float]) -> pd.DataFrame:
        if result.get("times_are_absolute_unix"):
            x_abs = np.asarray(result["times"], dtype=float)
        elif t0 is not None:
            x_abs = float(t0) + np.asarray(result["times"], dtype=float)
        else:
            ru = result.get("t0_unix")
            if ru is not None and np.isfinite(float(ru)):
                t0_eff = float(ru)
            else:
                eu = getattr(eeg_ds, "earliest_unix_timestamp", None)
                if eu is None:
                    raise ValueError(f"{track_key}: cannot resolve t0 (no kwarg, result['t0_unix'], or eeg_ds.earliest_unix_timestamp)")
                t0_eff = float(eu)
            x_abs = t0_eff + np.asarray(result["times"], dtype=float)

        data = {"t": x_abs}
        for c in MENTAL_STATE_COLUMNS:
            data[c] = np.asarray(result[c], dtype=float)
        return pd.DataFrame(data)

    @classmethod
    def _style_mental_states_line(cls) -> Dict[str, Any]:
        return dict(
            plot_pen_colors=[MENTAL_STATE_LINE_COLORS[c] for c in MENTAL_STATE_COLUMNS],
            plot_pen_width=MENTAL_STATE_LINE_WIDTH,
        )

    @classmethod
    def _embed_mental_states_track_on_timeline(cls, timeline, ms_ds, ref_name: str) -> Tuple[Any, Any, Any]:
        from pypho_timeline.core.synchronized_plot_mode import SynchronizedPlotMode
        from pypho_timeline.rendering.datasources.specific.eeg import MentalStatesDetailRenderer

        dock_h = int(80 + MentalStatesDetailRenderer.default_overview_series_height() * 20)
        ms_widget, _root, ms_plot_item, _dock = timeline.add_new_embedded_pyqtgraph_render_plot_widget(
            name=ms_ds.custom_datasource_name,
            dockSize=(500, dock_h),
            dockAddLocationOpts=["bottom"],
            sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA,
        )
        if ref_name in timeline.ui.matplotlib_view_widgets:
            ref_plot = timeline.ui.matplotlib_view_widgets[ref_name].getRootPlotItem()
            x0v, x1v = ref_plot.getViewBox().viewRange()[0]
            ms_plot_item.setXRange(x0v, x1v, padding=0)
        y_max = MentalStatesDetailRenderer.default_overview_series_height()
        cls._apply_mental_states_plot_chrome(ms_plot_item, y_max=y_max)
        timeline.add_track(ms_ds, name=ms_ds.custom_datasource_name, plot_item=ms_plot_item)
        ms_widget.optionsPanel = ms_widget.getOptionsPanel()
        if hasattr(_dock, "updateWidgetsHaveOptionsPanel"):
            _dock.updateWidgetsHaveOptionsPanel()
        return timeline.get_track_tuple(ms_ds.custom_datasource_name)

    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        if ctx.raw is None:
            raise ValueError("FrameMentalStatesComputation requires ctx.raw")
        return self.compute_frame_mental_states_from_raw(ctx.raw, **dict(params))


compute_frame_mental_states_series = FrameMentalStatesComputation.compute_frame_mental_states_series
compute_frame_mental_states_from_detailed_df = FrameMentalStatesComputation.compute_frame_mental_states_from_detailed_df
compute_frame_mental_states_from_raw = FrameMentalStatesComputation.compute_frame_mental_states_from_raw
compute_frame_mental_states_merged_for_intervals = FrameMentalStatesComputation.compute_frame_mental_states_merged_for_intervals
mental_states_colored_title_html = FrameMentalStatesComputation.mental_states_colored_title_html


def compute_frame_mental_states_for_eeg_datasource(eeg_ds: Any, *, eeg_name: Optional[str] = None, **compute_kw: Any) -> Dict[str, Any]:
    """Compute mental states from a timeline EEG datasource (live or offline with raws).

    Returns ``{"live": True}`` for live LSL EEG, else ``{"live": False, "result": ...}``.
    """
    from pypho_timeline.rendering.datasources.specific.lsl import LiveEEGTrackDatasource
    from pypho_timeline.rendering.datasources.track_datasource import RawProvidingTrackDatasource

    if isinstance(eeg_ds, LiveEEGTrackDatasource):
        return {"live": True}

    label = eeg_name or getattr(eeg_ds, "custom_datasource_name", "EEG")
    iv = getattr(eeg_ds, "intervals_df", None)
    if iv is None or len(iv) == 0:
        raise ValueError(f"{label}: no intervals_df (cannot align raws).")
    rd = getattr(eeg_ds, "raw_datasets_dict", None)
    raws_list, n_align = RawProvidingTrackDatasource.aligned_chronological_raws_for_intervals(
        intervals_df=iv,
        raw_datasets_dict=rd,
    )
    if not raws_list or n_align <= 0:
        raise ValueError(f"{label}: no MNE Raw available.")
    result = compute_frame_mental_states_merged_for_intervals(raws_list[:n_align], iv.iloc[:n_align], **compute_kw)
    return {"live": False, "result": result}



def start_frame_mental_states_track_background(timeline: Any, *, eeg_name: str, eeg_ds: Any, executor: Any = None,
            on_complete: Optional[Any] = None, on_error: Optional[Any] = None,
            **compute_kw: Any) -> Tuple[Any, Any]:
    """Run mental-states batch compute off the Qt thread; apply the track on the main thread.

    Parameters
    ----------
    executor
        Optional shared :class:`concurrent.futures.Executor`. When omitted, a single-worker
        :class:`concurrent.futures.ThreadPoolExecutor` is created and returned for reuse/shutdown.
    on_complete
        Optional ``callable(ms_widget, ms_track, ms_ds)`` after a successful apply.
    on_error
        Optional ``callable(exc)`` when compute or apply fails.

    Returns
    -------
    future, executor
        The submitted future and the executor instance (owned when not passed in).
    """
    from concurrent.futures import ThreadPoolExecutor
    from qtpy import QtCore

    from pypho_timeline.rendering.datasources.specific.eeg import mental_states_track_key_for_eeg_datasource

    owned_executor = executor is None
    ex = executor or ThreadPoolExecutor(max_workers=1)
    track_key = mental_states_track_key_for_eeg_datasource(eeg_ds)
    _ctx: Dict[str, Any] = {}

    def _apply_track() -> None:
        payload = _ctx.get("payload", {})
        try:
            if payload.get("live"):
                applied = apply_frame_mental_states_to_timeline(timeline, eeg_name=eeg_name, eeg_ds=eeg_ds)
            else:
                result = payload["result"]
                logger.info("%s: computed n_windows=%s", track_key, result.get("n_windows", 0))
                applied = apply_frame_mental_states_to_timeline(timeline, result, eeg_name=eeg_name, eeg_ds=eeg_ds)
            if on_complete is not None:
                on_complete(*applied)
        except Exception as exc:
            if on_error is not None:
                on_error(exc)
            else:
                logger.warning("%s: mental-states apply failed: %s", track_key, exc)
    ## END def _apply_track() -> None...

    def _on_done(fut: Any) -> None:
        try:
            _ctx["payload"] = fut.result()
        except Exception as exc:
            _ctx["error"] = exc
            if on_error is not None:
                QtCore.QTimer.singleShot(0, lambda: on_error(exc))
            else:
                logger.warning("%s: mental-states compute failed: %s", track_key, exc)
            if owned_executor:
                ex.shutdown(wait=False)
            return

        def _apply_and_maybe_shutdown() -> None:
            _apply_track()
            if owned_executor:
                ex.shutdown(wait=False)

        QtCore.QTimer.singleShot(0, _apply_and_maybe_shutdown)
    ## END def _on_done(fut: Any) -> None...


    fut = ex.submit(compute_frame_mental_states_for_eeg_datasource, eeg_ds, eeg_name=eeg_name, **compute_kw)
    fut.add_done_callback(_on_done)
    if owned_executor:
        logger.debug("%s: started background mental-states compute (owned executor).", track_key)
    else:
        logger.debug("%s: started background mental-states compute.", track_key)
    return fut, ex


def apply_frame_mental_states_to_timeline(timeline, result: Optional[Mapping[str, Any]] = None, *, eeg_name: str, eeg_ds: Any, t0: Optional[float] = None, **kwargs) -> Tuple[Any, Any, Any]:
    """Add or refresh the per-EEG mental-states track on ``timeline``.

    For live LSL EEG, ``result`` may be omitted; a live child datasource recomputes per viewport.
    For historical EEG, pass ``result`` from :func:`compute_frame_mental_states_merged_for_intervals`.
    """
    from pypho_timeline.rendering.datasources.specific.eeg import MentalStatesTrackDatasource, mental_states_track_key_for_eeg_datasource
    from pypho_timeline.rendering.helpers.normalization import ChannelNormalizationMode

    if eeg_ds is None:
        raise ValueError("apply_frame_mental_states_to_timeline requires eeg_ds (got None)")
    if eeg_name is None:
        raise ValueError("apply_frame_mental_states_to_timeline requires eeg_name (got None)")

    track_key = mental_states_track_key_for_eeg_datasource(eeg_ds)
    live_mode = False
    try:
        from pypho_timeline.rendering.datasources.specific.lsl import LiveEEGTrackDatasource, LiveMentalStatesTrackDatasource

        live_mode = isinstance(eeg_ds, LiveEEGTrackDatasource)
    except ImportError:
        LiveMentalStatesTrackDatasource = None  # type: ignore[misc, assignment]

    if track_key in timeline.track_renderers and hasattr(timeline, "track_is_fully_attached") and timeline.track_is_fully_attached(track_key):
        logger.info("%s: refreshing existing track.", track_key)
        ms_widget, ms_track, ms_ds = timeline.get_track_tuple(track_key)
        if live_mode and LiveMentalStatesTrackDatasource is not None:
            ms_ds._source_eeg = eeg_ds  # type: ignore[attr-defined]
            if getattr(eeg_ds, "intervals_df", None) is not None:
                ms_ds.intervals_df = eeg_ds.intervals_df.copy()
        elif result is not None:
            detailed = FrameMentalStatesComputation._result_to_detailed_df(result, eeg_ds, track_key, t0=t0)
            iv = intervals_df_for_theta_delta_track(eeg_ds, result, log_prefix=track_key)
            ms_ds.intervals_df = iv
            ms_ds.detailed_df = detailed
        ms_ds.source_data_changed_signal.emit()
        if ms_widget is not None and hasattr(ms_widget, "getRootPlotItem"):
            from pypho_timeline.rendering.datasources.specific.eeg import MentalStatesDetailRenderer
            FrameMentalStatesComputation._apply_mental_states_plot_chrome(ms_widget.getRootPlotItem(), y_max=MentalStatesDetailRenderer.default_overview_series_height())
        return (ms_widget, ms_track, ms_ds)

    if hasattr(timeline, "teardown_orphaned_track"):
        timeline.teardown_orphaned_track(track_key)

    ref_name = eeg_ds.custom_datasource_name
    if live_mode and LiveMentalStatesTrackDatasource is not None:
        logger.info("%s: creating live mental-states track.", track_key)
        ms_ds = LiveMentalStatesTrackDatasource(source_eeg=eeg_ds, parent=eeg_ds)
    else:
        if result is None:
            raise ValueError(f"{track_key}: historical apply requires result dict")
        n_windows = int(result.get("n_windows", 0) or 0)
        if n_windows <= 0:
            raise ValueError(f"{track_key}: mental-states computation produced no windows.")
        detailed = FrameMentalStatesComputation._result_to_detailed_df(result, eeg_ds, track_key, t0=t0)
        iv = intervals_df_for_theta_delta_track(eeg_ds, result, log_prefix=track_key)
        logger.info("%s: creating mental-states track (n=%s).", track_key, n_windows)
        ms_ds = MentalStatesTrackDatasource(
            intervals_df=iv,
            eeg_df=detailed,
            custom_datasource_name=track_key,
            max_points_per_second=32.0,
            enable_downsampling=True,
            channel_names=list(MENTAL_STATE_COLUMNS),
            normalize=False,
            fallback_normalization_mode=ChannelNormalizationMode.NONE,
            lab_obj_dict=getattr(eeg_ds, "lab_obj_dict", None),
            raw_datasets_dict=getattr(eeg_ds, "raw_datasets_dict", None),
            **FrameMentalStatesComputation._style_mental_states_line(),
        )

    return FrameMentalStatesComputation._embed_mental_states_track_on_timeline(timeline, ms_ds, ref_name)


__all__ = [
    "DEFAULT_FILTER_ORDER",
    "DEFAULT_ROLLING_NORM_WINDOW",
    "DEFAULT_STEP_SEC",
    "DEFAULT_WINDOW_SEC",
    "FRAME_MENTAL_STATE_BANDS",
    "FrameMentalStatesComputation",
    "MENTAL_STATE_COLUMNS",
    "MENTAL_STATE_DISPLAY_NAMES",
    "MENTAL_STATE_DROWSINESS",
    "MENTAL_STATE_FOCUS",
    "MENTAL_STATE_LINE_COLORS",
    "MENTAL_STATE_RELAXATION",
    "MENTAL_STATE_STRESS",
    "MentalStatesRollingState",
    "apply_frame_mental_states_to_timeline",
    "compute_frame_mental_states_for_eeg_datasource",
    "start_frame_mental_states_track_background",
    "mental_states_colored_title_html",
    "compute_frame_mental_states_from_detailed_df",
    "compute_frame_mental_states_from_raw",
    "compute_frame_mental_states_merged_for_intervals",
    "compute_frame_mental_states_series",
]
