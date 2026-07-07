"""Frame-style EEG mental states from band-power ratios (relaxation, focus, stress, drowsiness).

Port of band-ratio logic from frame-eeg: per-window band-pass variance power, dB session
min-max normalization (50–100), then log-ratio rolling min-max scaled to 0–100%.

Public surfaces:

- :func:`compute_frame_mental_states_series` -- numpy core (live + historical)
- :func:`compute_frame_mental_states_from_detailed_df` -- timeline ``detailed_df`` adapter
- :func:`compute_frame_mental_states_from_raw` -- one MNE raw segment
- :func:`compute_frame_mental_states_merged_for_intervals` -- multi-raw stitch (absolute unix)
- :class:`FrameMentalStatesComputation` -- DAG node for ``run_eeg_computations_graph``
- :func:`apply_frame_mental_states_to_timeline` -- add/refresh mental-states track
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

MENTAL_STATE_LINE_WIDTH: float = 1.2


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


def _ensure_band_db_entry(state: MentalStatesRollingState, band_name: str) -> List[float]:
    if band_name not in state.band_db_minmax:
        state.band_db_minmax[band_name] = [float(np.inf), float(-np.inf)]
    return state.band_db_minmax[band_name]


def _band_merged_power(block_2d: np.ndarray, fmin: float, fmax: float, sfreq: float, filter_order: int, cache: MutableMapping[Tuple[float, float, float], np.ndarray]) -> float:
    """Band-pass each channel, variance per channel, mean across channels."""
    filtered = bandpass_filter_channels(block_2d, fmin, fmax, sfreq, filter_order, cache)
    if filtered.size == 0:
        return float("nan")
    per_ch = np.nanvar(filtered, axis=1)
    if not np.any(np.isfinite(per_ch)):
        return float("nan")
    return float(np.nanmean(per_ch))


def _db_normalize_band(band_name: str, power: float, state: MentalStatesRollingState) -> float:
    """Convert power to dB and map to 50–100 using session-expanded min-max."""
    dB_value = 10.0 * np.log10(max(power, 0.0) + 1e-6)
    min_val, max_val = _ensure_band_db_entry(state, band_name)
    min_val = min(min_val, dB_value)
    max_val = max(max_val, dB_value)
    state.band_db_minmax[band_name] = [min_val, max_val]
    if max_val > min_val:
        return 50.0 + 50.0 * (dB_value - min_val) / (max_val - min_val)
    return 50.0


def _rolling_norm_100(log_ratio: float, ratio_deque: Deque[float]) -> float:
    ratio_deque.append(float(log_ratio))
    rmin = min(ratio_deque)
    rmax = max(ratio_deque)
    return (log_ratio - rmin) / (rmax - rmin + 1e-6) * 100.0


def _mental_state_values(db_band_powers: Mapping[str, float], state: MentalStatesRollingState) -> Dict[str, float]:
    """Compute four mental-state percentages from normalized dB band powers."""
    alpha_power = 10.0 ** (db_band_powers["Alpha"] / 10.0)
    beta_power = 10.0 ** (db_band_powers["Beta"] / 10.0)
    gamma_power = 10.0 ** (db_band_powers["Gamma"] / 10.0)
    delta_power = 10.0 ** (db_band_powers["Delta"] / 10.0)
    theta_power = 10.0 ** (db_band_powers["Theta"] / 10.0)

    alpha_beta_ratio = alpha_power / (beta_power + 1e-6)
    log_alpha_beta = np.log1p(alpha_beta_ratio)
    relaxation = _rolling_norm_100(log_alpha_beta, state.alpha_beta_log_ratios)

    beta_theta_ratio = beta_power / (theta_power + 1e-6)
    log_beta_theta = np.log1p(beta_theta_ratio)
    focus = _rolling_norm_100(log_beta_theta, state.beta_theta_focus_ratios)

    combined_beta_gamma = (beta_power / (alpha_power + 1e-6)) * (gamma_power + 1e-6)
    log_stress = np.log1p(combined_beta_gamma)
    stress = _rolling_norm_100(log_stress, state.beta_alpha_stress_ratios)

    delta_alpha_ratio = delta_power / (alpha_power + 1e-6)
    log_delta_alpha = np.log1p(delta_alpha_ratio)
    drowsiness = _rolling_norm_100(log_delta_alpha, state.delta_alpha_drowsiness_ratios)

    return {
        MENTAL_STATE_RELAXATION: float(relaxation),
        MENTAL_STATE_FOCUS: float(focus),
        MENTAL_STATE_STRESS: float(stress),
        MENTAL_STATE_DROWSINESS: float(drowsiness),
    }


def compute_frame_mental_states_series(samples_2d: np.ndarray, sfreq: float, t_unix: Optional[np.ndarray] = None, *, window_sec: float = DEFAULT_WINDOW_SEC, step_sec: float = DEFAULT_STEP_SEC, filter_order: int = DEFAULT_FILTER_ORDER, state: Optional[MentalStatesRollingState] = None, viewport_t_min: Optional[float] = None, viewport_t_max: Optional[float] = None, incremental: bool = True) -> Dict[str, Any]:
    """Compute sliding-window mental states from multi-channel EEG samples.

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
    data = np.asarray(samples_2d, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("samples_2d must be 2D (n_channels, n_samples)")
    n_ch, n_samp = data.shape
    sf = float(sfreq)
    if sf <= 0.0 or n_ch == 0 or n_samp == 0:
        empty = np.array([], dtype=float)
        return dict(
            times=empty,
            relaxation=empty,
            focus=empty,
            stress=empty,
            drowsiness=empty,
            state=state or MentalStatesRollingState(),
            n_windows=0,
        )

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
    if n_samp < w:
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

    filter_cache: Dict[Tuple[float, float, float], np.ndarray] = {}
    times_out: List[float] = []
    relaxation_out: List[float] = []
    focus_out: List[float] = []
    stress_out: List[float] = []
    drowsiness_out: List[float] = []

    win_start = 0
    while win_start + w <= n_samp:
        center_idx = win_start + w // 2
        t_center = float(t_arr[center_idx])

        if incremental and state.last_center_t is not None and t_center <= state.last_center_t + 1e-9:
            win_start += s
            continue

        block = data[:, win_start : win_start + w]
        db_powers: Dict[str, float] = {}
        skip_window = False
        for band_name, (fmin, fmax) in FRAME_MENTAL_STATE_BANDS.items():
            if fmin >= sf / 2.0:
                skip_window = True
                break
            merged = _band_merged_power(block, fmin, fmax, sf, filter_order, filter_cache)
            if not np.isfinite(merged):
                skip_window = True
                break
            db_powers[band_name] = _db_normalize_band(band_name, merged, state)

        if not skip_window and len(db_powers) == len(FRAME_MENTAL_STATE_BANDS):
            ms = _mental_state_values(db_powers, state)
            times_out.append(t_center)
            relaxation_out.append(ms[MENTAL_STATE_RELAXATION])
            focus_out.append(ms[MENTAL_STATE_FOCUS])
            stress_out.append(ms[MENTAL_STATE_STRESS])
            drowsiness_out.append(ms[MENTAL_STATE_DROWSINESS])
            state.last_center_t = t_center

        win_start += s

    times_arr = np.asarray(times_out, dtype=float)
    if viewport_t_min is not None and viewport_t_max is not None and times_arr.size:
        lo, hi = float(viewport_t_min), float(viewport_t_max)
        if lo > hi:
            lo, hi = hi, lo
        mask = (times_arr >= lo) & (times_arr <= hi)
        times_arr = times_arr[mask]
        relaxation_out = np.asarray(relaxation_out, dtype=float)[mask]
        focus_out = np.asarray(focus_out, dtype=float)[mask]
        stress_out = np.asarray(stress_out, dtype=float)[mask]
        drowsiness_out = np.asarray(drowsiness_out, dtype=float)[mask]
    else:
        relaxation_out = np.asarray(relaxation_out, dtype=float)
        focus_out = np.asarray(focus_out, dtype=float)
        stress_out = np.asarray(stress_out, dtype=float)
        drowsiness_out = np.asarray(drowsiness_out, dtype=float)

    return dict(
        times=times_arr,
        relaxation=relaxation_out,
        focus=focus_out,
        stress=stress_out,
        drowsiness=drowsiness_out,
        state=state,
        n_windows=int(times_arr.size),
    )


def compute_frame_mental_states_from_detailed_df(df: pd.DataFrame, channel_names: Sequence[str], sfreq: float, *, t_col: str = "t", viewport_t_min: Optional[float] = None, viewport_t_max: Optional[float] = None, state: Optional[MentalStatesRollingState] = None, incremental: bool = True, **compute_kw: Any) -> pd.DataFrame:
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
    result = compute_frame_mental_states_series(
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


def compute_frame_mental_states_from_raw(raw_eeg: mne.io.BaseRaw, *, picks: str = "eeg", channel_names: Optional[Sequence[str]] = None, window_sec: float = DEFAULT_WINDOW_SEC, step_sec: float = DEFAULT_STEP_SEC, filter_order: int = DEFAULT_FILTER_ORDER, state: Optional[MentalStatesRollingState] = None, copy_raw: bool = False) -> Dict[str, Any]:
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
    result = compute_frame_mental_states_series(
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


def compute_frame_mental_states_merged_for_intervals(raws: Sequence[mne.io.BaseRaw], intervals_df: pd.DataFrame, **compute_kw: Any) -> Dict[str, Any]:
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
        sub = compute_frame_mental_states_from_raw(raw, state=state, **compute_kw)
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


def _result_to_detailed_df(result: Mapping[str, Any], eeg_ds: Any, track_key: str, *, t0: Optional[float]) -> pd.DataFrame:
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


def _style_mental_states_line() -> Dict[str, Any]:
    return dict(
        plot_pen_colors=[MENTAL_STATE_LINE_COLORS[c] for c in MENTAL_STATE_COLUMNS],
        plot_pen_width=MENTAL_STATE_LINE_WIDTH,
    )


def _embed_mental_states_track_on_timeline(timeline, ms_ds, ref_name: str) -> Tuple[Any, Any, Any]:
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
    ms_plot_item.setTitle("EEG mental states (relaxation / focus / stress / drowsiness, %)")
    ms_plot_item.setLabel("bottom", "Time (unix s)")
    ms_plot_item.setLabel("left", "%")
    ms_plot_item.setYRange(0, y_max, padding=0.02)
    ms_plot_item.showAxis("left")
    timeline.add_track(ms_ds, name=ms_ds.custom_datasource_name, plot_item=ms_plot_item)
    ms_widget.optionsPanel = ms_widget.getOptionsPanel()
    if hasattr(_dock, "updateWidgetsHaveOptionsPanel"):
        _dock.updateWidgetsHaveOptionsPanel()
    return timeline.get_track_tuple(ms_ds.custom_datasource_name)


class FrameMentalStatesComputation(SpecificComputationBase):
    computation_id: ClassVar[str] = "mental_states"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ()
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.stream

    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        if ctx.raw is None:
            raise ValueError("FrameMentalStatesComputation requires ctx.raw")
        return compute_frame_mental_states_from_raw(ctx.raw, **dict(params))


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
            detailed = _result_to_detailed_df(result, eeg_ds, track_key, t0=t0)
            iv = intervals_df_for_theta_delta_track(eeg_ds, result, log_prefix=track_key)
            ms_ds.intervals_df = iv
            ms_ds.detailed_df = detailed
        ms_ds.source_data_changed_signal.emit()
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
        detailed = _result_to_detailed_df(result, eeg_ds, track_key, t0=t0)
        iv = intervals_df_for_theta_delta_track(eeg_ds, result, log_prefix=track_key)
        logger.info("%s: creating mental-states track (n=%s).", track_key, n_windows)
        ms_ds = MentalStatesTrackDatasource(
            intervals_df=iv,
            eeg_df=detailed,
            custom_datasource_name=track_key,
            max_points_per_second=64.0,
            enable_downsampling=True,
            channel_names=list(MENTAL_STATE_COLUMNS),
            normalize=False,
            fallback_normalization_mode=ChannelNormalizationMode.NONE,
            lab_obj_dict=getattr(eeg_ds, "lab_obj_dict", None),
            raw_datasets_dict=getattr(eeg_ds, "raw_datasets_dict", None),
            **_style_mental_states_line(),
        )

    return _embed_mental_states_track_on_timeline(timeline, ms_ds, ref_name)


__all__ = [
    "DEFAULT_FILTER_ORDER",
    "DEFAULT_ROLLING_NORM_WINDOW",
    "DEFAULT_STEP_SEC",
    "DEFAULT_WINDOW_SEC",
    "FRAME_MENTAL_STATE_BANDS",
    "FrameMentalStatesComputation",
    "MENTAL_STATE_COLUMNS",
    "MENTAL_STATE_DROWSINESS",
    "MENTAL_STATE_FOCUS",
    "MENTAL_STATE_LINE_COLORS",
    "MENTAL_STATE_RELAXATION",
    "MENTAL_STATE_STRESS",
    "MentalStatesRollingState",
    "apply_frame_mental_states_to_timeline",
    "compute_frame_mental_states_from_detailed_df",
    "compute_frame_mental_states_from_raw",
    "compute_frame_mental_states_merged_for_intervals",
    "compute_frame_mental_states_series",
]
