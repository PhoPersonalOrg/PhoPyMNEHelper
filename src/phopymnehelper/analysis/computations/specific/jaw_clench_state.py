"""Jaw-clench state intervals from jaw-clench probability (latch + release-after-quiet FSM)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from phopymnehelper.analysis.computations.specific.jaw_clench_probability import JAW_CLENCH_PROB_COLUMN, compute_jaw_clench_probability_merged_for_intervals, intervals_df_for_jaw_clench_track

logger = logging.getLogger(__name__)

JAW_CLENCH_STATE_DEFAULT_ONSET_THRESH: float = 0.55
JAW_CLENCH_STATE_DEFAULT_QUIET_THRESH: float = 0.20
JAW_CLENCH_STATE_DEFAULT_QUIET_MIN_S: float = 0.30
JAW_CLENCH_STATE_DEFAULT_RELEASE_THRESH: float = 0.35
JAW_CLENCH_STATE_DEFAULT_MIN_CLINCH_S: float = 0.10
JAW_CLENCH_STATE_DEFAULT_MAX_CLINCH_S: float = 120.0
JAW_CLENCH_STATE_DEFAULT_MERGE_GAP_S: float = 0.10
JAW_CLENCH_STATE_DEFAULT_PAD_S: float = 0.05
JAW_CLENCH_STATE_DEFAULT_ARMED_RELEASE_TIMEOUT_S: float = 30.0


def jaw_clench_state_track_key_for_eeg_datasource(eeg_ds: Any) -> str:
    """Return ``custom_datasource_name`` for a jaw-clench state interval child track."""
    base = getattr(eeg_ds, "custom_datasource_name", None) or "EEG"
    return f"{base}_jaw_clench_state"


def _merge_clinch_interval_rows(rows: List[Dict[str, float]], merge_gap_s: float) -> List[Dict[str, float]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: float(r["t_start"]))
    merged: List[Dict[str, float]] = [dict(sorted_rows[0])]
    for row in sorted_rows[1:]:
        prev = merged[-1]
        prev_end = float(prev["t_start"]) + float(prev["t_duration"])
        gap = float(row["t_start"]) - prev_end
        if gap <= float(merge_gap_s):
            new_end = max(prev_end, float(row["t_start"]) + float(row["t_duration"]))
            prev["t_duration"] = new_end - float(prev["t_start"])
            prev["peak_prob"] = max(float(prev.get("peak_prob", 0.0)), float(row.get("peak_prob", 0.0)))
            prev["release_prob"] = max(float(prev.get("release_prob", 0.0)), float(row.get("release_prob", 0.0)))
        else:
            merged.append(dict(row))
    return merged


def probability_series_to_clench_intervals(t: np.ndarray, prob: np.ndarray, *, onset_thresh: float = JAW_CLENCH_STATE_DEFAULT_ONSET_THRESH, quiet_thresh: float = JAW_CLENCH_STATE_DEFAULT_QUIET_THRESH, quiet_min_s: float = JAW_CLENCH_STATE_DEFAULT_QUIET_MIN_S, release_thresh: float = JAW_CLENCH_STATE_DEFAULT_RELEASE_THRESH, min_clinch_s: float = JAW_CLENCH_STATE_DEFAULT_MIN_CLINCH_S, max_clinch_s: float = JAW_CLENCH_STATE_DEFAULT_MAX_CLINCH_S, merge_gap_s: float = JAW_CLENCH_STATE_DEFAULT_MERGE_GAP_S, pad_s: float = JAW_CLENCH_STATE_DEFAULT_PAD_S, armed_release_timeout_s: float = JAW_CLENCH_STATE_DEFAULT_ARMED_RELEASE_TIMEOUT_S, require_rising_onset: bool = True) -> pd.DataFrame:
    """Convert jaw-clench probability samples into merged clinch interval rows.

    Uses a latch-and-release-after-quiet FSM: onset spikes enter CLINCHED, low ``prob`` during
    sustained bite is ignored, and release is detected when ``prob`` rises above ``release_thresh``
    after a quiet plateau below ``quiet_thresh``.
    """
    t_arr = np.asarray(t, dtype=float)
    p_arr = np.asarray(prob, dtype=float)
    if t_arr.size == 0 or p_arr.size == 0:
        return pd.DataFrame({"t_start": pd.Series(dtype=float), "t_duration": pd.Series(dtype=float), "t_end": pd.Series(dtype=float), "peak_prob": pd.Series(dtype=float), "release_prob": pd.Series(dtype=float)})
    if t_arr.size != p_arr.size:
        raise ValueError(f"t length ({t_arr.size}) must match prob length ({p_arr.size})")
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    p_arr = p_arr[order]
    valid = np.isfinite(t_arr) & np.isfinite(p_arr)
    t_arr = t_arr[valid]
    p_arr = p_arr[valid]
    if t_arr.size == 0:
        return pd.DataFrame({"t_start": pd.Series(dtype=float), "t_duration": pd.Series(dtype=float), "t_end": pd.Series(dtype=float), "peak_prob": pd.Series(dtype=float), "release_prob": pd.Series(dtype=float)})

    rows: List[Dict[str, float]] = []
    clinched = False
    armed = False
    t_onset = 0.0
    t_armed = 0.0
    quiet_run_start: Optional[float] = None
    peak_prob = 0.0
    release_prob = 0.0
    prev_p = float(p_arr[0])

    def _close_interval(t_end: float, *, release_p: float) -> None:
        nonlocal clinched, armed, quiet_run_start, peak_prob, release_prob
        t_start = float(t_onset) - float(pad_s)
        t_stop = float(t_end) + float(pad_s)
        duration = t_stop - t_start
        if duration >= float(min_clinch_s):
            rows.append(dict(t_start=t_start, t_duration=duration, t_end=t_stop, peak_prob=float(peak_prob), release_prob=float(release_p)))
        clinched = False
        armed = False
        quiet_run_start = None
        peak_prob = 0.0
        release_prob = 0.0

    def _cancel_clinch() -> None:
        nonlocal clinched, armed, quiet_run_start, peak_prob, release_prob
        clinched = False
        armed = False
        quiet_run_start = None
        peak_prob = 0.0
        release_prob = 0.0

    for i in range(t_arr.size):
        ti = float(t_arr[i])
        pi = float(p_arr[i])
        rising = pi > prev_p
        if not clinched:
            onset_hit = pi >= float(onset_thresh) and (rising or not require_rising_onset or i == 0)
            if onset_hit:
                clinched = True
                armed = False
                t_onset = ti
                t_armed = 0.0
                quiet_run_start = None
                peak_prob = pi
                release_prob = 0.0
        else:
            peak_prob = max(peak_prob, pi)
            if pi < float(quiet_thresh):
                if quiet_run_start is None:
                    quiet_run_start = ti
                elif (ti - quiet_run_start) >= float(quiet_min_s):
                    if not armed:
                        armed = True
                        t_armed = ti
            else:
                quiet_run_start = None
            if armed:
                if pi >= float(release_thresh):
                    release_prob = max(release_prob, pi)
                    _close_interval(ti, release_p=pi)
                elif (ti - t_armed) >= float(armed_release_timeout_s):
                    if (ti - t_onset) >= float(min_clinch_s):
                        _close_interval(ti, release_p=release_prob)
                    else:
                        _cancel_clinch()
            elif (ti - t_onset) > float(max_clinch_s):
                _close_interval(ti, release_p=release_prob)
        prev_p = pi

    if clinched:
        t_last = float(t_arr[-1])
        if armed:
            if (t_last - t_onset) >= float(min_clinch_s):
                _close_interval(t_last, release_p=release_prob)
            else:
                _cancel_clinch()
        else:
            _cancel_clinch()

    merged_rows = _merge_clinch_interval_rows(rows, merge_gap_s)
    if not merged_rows:
        return pd.DataFrame({"t_start": pd.Series(dtype=float), "t_duration": pd.Series(dtype=float), "t_end": pd.Series(dtype=float), "peak_prob": pd.Series(dtype=float), "release_prob": pd.Series(dtype=float)})
    return pd.DataFrame(merged_rows)


def compute_jaw_clench_state_intervals_from_prob_df(detailed_df: pd.DataFrame, *, t_col: str = "t", prob_col: str = JAW_CLENCH_PROB_COLUMN, **fsm_kw: Any) -> Dict[str, Any]:
    """Run the clinch FSM on a probability ``detailed_df`` (``t`` + ``jaw_clench_prob``)."""
    if detailed_df is None or len(detailed_df) == 0 or t_col not in detailed_df.columns or prob_col not in detailed_df.columns:
        empty = pd.DataFrame({"t_start": pd.Series(dtype=float), "t_duration": pd.Series(dtype=float), "t_end": pd.Series(dtype=float)})
        return dict(intervals_df=empty, params=dict(fsm_kw), n_intervals=0)
    df = detailed_df.sort_values(t_col).reset_index(drop=True)
    intervals_df = probability_series_to_clench_intervals(df[t_col].to_numpy(dtype=float), df[prob_col].to_numpy(dtype=float), **fsm_kw)
    params = dict(onset_thresh=fsm_kw.get("onset_thresh", JAW_CLENCH_STATE_DEFAULT_ONSET_THRESH), quiet_thresh=fsm_kw.get("quiet_thresh", JAW_CLENCH_STATE_DEFAULT_QUIET_THRESH), quiet_min_s=fsm_kw.get("quiet_min_s", JAW_CLENCH_STATE_DEFAULT_QUIET_MIN_S), release_thresh=fsm_kw.get("release_thresh", JAW_CLENCH_STATE_DEFAULT_RELEASE_THRESH), min_clinch_s=fsm_kw.get("min_clinch_s", JAW_CLENCH_STATE_DEFAULT_MIN_CLINCH_S), max_clinch_s=fsm_kw.get("max_clinch_s", JAW_CLENCH_STATE_DEFAULT_MAX_CLINCH_S), merge_gap_s=fsm_kw.get("merge_gap_s", JAW_CLENCH_STATE_DEFAULT_MERGE_GAP_S), pad_s=fsm_kw.get("pad_s", JAW_CLENCH_STATE_DEFAULT_PAD_S))
    return dict(intervals_df=intervals_df, params=params, n_intervals=int(len(intervals_df)))


def compute_jaw_clench_state_intervals_from_raw(raws: Sequence[Any], intervals_df: pd.DataFrame, *, eeg_ds: Any = None, parent_detailed_df: Optional[pd.DataFrame] = None, channel_names: Optional[Sequence[str]] = None, prob_kw: Optional[Mapping[str, Any]] = None, fsm_kw: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Compute probability from raws, then derive clinch state intervals."""
    prob_result = compute_jaw_clench_probability_merged_for_intervals(raws, intervals_df, eeg_ds=eeg_ds, parent_detailed_df=parent_detailed_df, channel_names=channel_names, **dict(prob_kw or {}))
    n_windows = int(prob_result.get("n_windows", 0) or 0)
    if n_windows <= 0:
        empty = pd.DataFrame({"t_start": pd.Series(dtype=float), "t_duration": pd.Series(dtype=float), "t_end": pd.Series(dtype=float)})
        return dict(intervals_df=empty, prob_result=prob_result, params=dict(fsm_kw or {}), n_intervals=0)
    if prob_result.get("times_are_absolute_unix"):
        prob_df = pd.DataFrame({"t": np.asarray(prob_result["times"], dtype=float), JAW_CLENCH_PROB_COLUMN: np.asarray(prob_result[JAW_CLENCH_PROB_COLUMN], dtype=float)})
    else:
        raise ValueError("compute_jaw_clench_state_intervals_from_raw requires times_are_absolute_unix from probability merge")
    state_result = compute_jaw_clench_state_intervals_from_prob_df(prob_df, **dict(fsm_kw or {}))
    state_result["prob_result"] = prob_result
    return state_result


def _style_jaw_clench_state_intervals(intervals_df: pd.DataFrame) -> pd.DataFrame:
    """Apply red overview styling to clinch interval rows."""
    import pyqtgraph as pg

    out = intervals_df.copy()
    color = pg.mkColor("#e45756")
    color.setAlphaF(0.75)
    pen = pg.mkPen(color, width=1)
    brush = pg.mkBrush(color)
    out["series_vertical_offset"] = 0.0
    out["series_height"] = 1.0
    out["pen"] = [pen] * len(out)
    out["brush"] = [brush] * len(out)
    return out


def apply_jaw_clench_state_to_timeline(timeline, result: Optional[Mapping[str, Any]] = None, *, eeg_name: str, eeg_ds: Any, prob_detailed_df: Optional[pd.DataFrame] = None) -> Tuple[Any, Any, Any]:
    """Add or refresh the jaw-clench state interval track on ``timeline``."""
    from pypho_timeline.core.synchronized_plot_mode import SynchronizedPlotMode
    from pypho_timeline.rendering.datasources.specific.eeg import JawClenchStateTrackDatasource, jaw_clench_state_track_key_for_eeg_datasource

    if eeg_ds is None:
        raise ValueError("apply_jaw_clench_state_to_timeline requires eeg_ds (got None)")
    if eeg_name is None:
        raise ValueError("apply_jaw_clench_state_to_timeline requires eeg_name (got None)")

    track_key = jaw_clench_state_track_key_for_eeg_datasource(eeg_ds)
    live_mode = False
    try:
        from pypho_timeline.rendering.datasources.specific.lsl import LiveEEGTrackDatasource, LiveJawClenchStateTrackDatasource
        live_mode = isinstance(eeg_ds, LiveEEGTrackDatasource)
    except ImportError:
        LiveJawClenchStateTrackDatasource = None  # type: ignore[misc, assignment]

    if live_mode and LiveJawClenchStateTrackDatasource is not None:
        if track_key in timeline.track_renderers and hasattr(timeline, "track_is_fully_attached") and timeline.track_is_fully_attached(track_key):
            logger.info("%s: refreshing existing live track.", track_key)
            jaw_widget, jaw_track, jaw_ds = timeline.get_track_tuple(track_key)
            jaw_ds._source_eeg = eeg_ds  # type: ignore[attr-defined]
            if getattr(eeg_ds, "intervals_df", None) is not None:
                jaw_ds.intervals_df = eeg_ds.intervals_df.copy()
            jaw_ds.source_data_changed_signal.emit()
            return (jaw_widget, jaw_track, jaw_ds)
        if hasattr(timeline, "teardown_orphaned_track"):
            timeline.teardown_orphaned_track(track_key)
        logger.info("%s: creating live jaw-clench state track.", track_key)
        jaw_ds = LiveJawClenchStateTrackDatasource(source_eeg=eeg_ds, parent=eeg_ds, timeline=timeline)
        jaw_widget, _root, jaw_plot_item, _jaw_dock = timeline.add_new_embedded_pyqtgraph_render_plot_widget(name=jaw_ds.custom_datasource_name, dockSize=(500, 40), dockAddLocationOpts=["bottom"], sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA)
        ref_name = eeg_ds.custom_datasource_name
        if ref_name in timeline.ui.matplotlib_view_widgets:
            ref_plot = timeline.ui.matplotlib_view_widgets[ref_name].getRootPlotItem()
            x0v, x1v = ref_plot.getViewBox().viewRange()[0]
            jaw_plot_item.setXRange(x0v, x1v, padding=0)
        jaw_plot_item.setTitle("Jaw clench state (EEG-derived intervals)")
        jaw_plot_item.setLabel("bottom", "Time (unix s)")
        jaw_plot_item.setLabel("left", "Clinch")
        jaw_plot_item.setYRange(0, 1, padding=0.0)
        jaw_plot_item.hideAxis("left")
        timeline.add_track(jaw_ds, name=jaw_ds.custom_datasource_name, plot_item=jaw_plot_item)
        jaw_widget.optionsPanel = jaw_widget.getOptionsPanel()
        if hasattr(_jaw_dock, "updateWidgetsHaveOptionsPanel"):
            _jaw_dock.updateWidgetsHaveOptionsPanel()
        return timeline.get_track_tuple(jaw_ds.custom_datasource_name)

    if result is None and prob_detailed_df is not None:
        result = compute_jaw_clench_state_intervals_from_prob_df(prob_detailed_df)
    if result is None:
        raise ValueError(f"{track_key}: historical apply requires result dict or prob_detailed_df")

    intervals_df = result.get("intervals_df")
    if intervals_df is None:
        raise ValueError(f"{track_key}: result missing intervals_df")
    intervals_df = _style_jaw_clench_state_intervals(intervals_df)
    n_intervals = int(result.get("n_intervals", len(intervals_df)) or 0)
    logger.info("%s: applying jaw-clench state track (n_intervals=%s).", track_key, n_intervals)

    if track_key in timeline.track_renderers and hasattr(timeline, "track_is_fully_attached") and timeline.track_is_fully_attached(track_key):
        logger.info("%s: refreshing existing track.", track_key)
        jaw_widget, jaw_track, jaw_ds = timeline.get_track_tuple(track_key)
        prob_result = result.get("prob_result")
        if prob_result is not None and eeg_ds is not None:
            jaw_ds.intervals_df = intervals_df.copy()
        else:
            jaw_ds.intervals_df = intervals_df.copy()
        jaw_ds.source_data_changed_signal.emit()
        return (jaw_widget, jaw_track, jaw_ds)

    if hasattr(timeline, "teardown_orphaned_track"):
        timeline.teardown_orphaned_track(track_key)

    prob_result = result.get("prob_result")
    parent_iv = intervals_df_for_jaw_clench_track(eeg_ds, prob_result, log_prefix=track_key) if prob_result is not None else getattr(eeg_ds, "intervals_df", None)
    if parent_iv is None or len(parent_iv) == 0:
        parent_iv = intervals_df.copy()
    jaw_ds = JawClenchStateTrackDatasource(intervals_df=intervals_df, parent_intervals_df=parent_iv, custom_datasource_name=track_key, enable_downsampling=False, lab_obj_dict=getattr(eeg_ds, "lab_obj_dict", None), raw_datasets_dict=getattr(eeg_ds, "raw_datasets_dict", None))
    ref_name = eeg_ds.custom_datasource_name
    jaw_widget, _root, jaw_plot_item, _jaw_dock = timeline.add_new_embedded_pyqtgraph_render_plot_widget(name=jaw_ds.custom_datasource_name, dockSize=(500, 40), dockAddLocationOpts=["bottom"], sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA)
    if ref_name in timeline.ui.matplotlib_view_widgets:
        ref_plot = timeline.ui.matplotlib_view_widgets[ref_name].getRootPlotItem()
        x0v, x1v = ref_plot.getViewBox().viewRange()[0]
        jaw_plot_item.setXRange(x0v, x1v, padding=0)
    jaw_plot_item.setTitle("Jaw clench state (EEG-derived intervals)")
    jaw_plot_item.setLabel("bottom", "Time (unix s)")
    jaw_plot_item.setLabel("left", "Clinch")
    jaw_plot_item.setYRange(0, 1, padding=0.0)
    jaw_plot_item.hideAxis("left")
    timeline.add_track(jaw_ds, name=jaw_ds.custom_datasource_name, plot_item=jaw_plot_item)
    jaw_widget.optionsPanel = jaw_widget.getOptionsPanel()
    if hasattr(_jaw_dock, "updateWidgetsHaveOptionsPanel"):
        _jaw_dock.updateWidgetsHaveOptionsPanel()
    return timeline.get_track_tuple(jaw_ds.custom_datasource_name)


__all__ = [
    "JAW_CLENCH_STATE_DEFAULT_ARMED_RELEASE_TIMEOUT_S",
    "JAW_CLENCH_STATE_DEFAULT_MAX_CLINCH_S",
    "JAW_CLENCH_STATE_DEFAULT_MERGE_GAP_S",
    "JAW_CLENCH_STATE_DEFAULT_MIN_CLINCH_S",
    "JAW_CLENCH_STATE_DEFAULT_ONSET_THRESH",
    "JAW_CLENCH_STATE_DEFAULT_PAD_S",
    "JAW_CLENCH_STATE_DEFAULT_QUIET_MIN_S",
    "JAW_CLENCH_STATE_DEFAULT_QUIET_THRESH",
    "JAW_CLENCH_STATE_DEFAULT_RELEASE_THRESH",
    "apply_jaw_clench_state_to_timeline",
    "compute_jaw_clench_state_intervals_from_prob_df",
    "compute_jaw_clench_state_intervals_from_raw",
    "jaw_clench_state_track_key_for_eeg_datasource",
    "probability_series_to_clench_intervals",
]
