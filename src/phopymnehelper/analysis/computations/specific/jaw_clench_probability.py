"""Jaw-clench probability from EEG (sliding-window PTP + multi-channel sync heuristic).

Public surfaces:

- :func:`compute_jaw_clench_probability_series` -- numpy core (live + historical)
- :func:`compute_jaw_clench_probability_from_detailed_df` -- timeline ``detailed_df`` adapter
- :func:`compute_jaw_clench_probability_from_raw` -- one MNE raw segment
- :func:`compute_jaw_clench_probability_merged_for_intervals` -- multi-raw stitch (absolute unix)
- :class:`JawClenchProbabilityComputation` -- DAG node for ``run_eeg_computations_graph``
- :func:`apply_jaw_clench_probability_to_timeline` -- add/refresh the probability track
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from phopymnehelper.analysis.computations.protocol import ArtifactKind, RunContext
from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import interval_row_t_start_unix_seconds, raw_relative_times_to_timeline_unix
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase

logger = logging.getLogger(__name__)

JAW_CLENCH_PROB_COLUMN: str = "jaw_clench_prob"
_FETCH_DETAIL_T_END_INCLUSIVE_EPS: float = 1e-3


def _interval_row_unix_bounds(iv: pd.Series) -> Tuple[float, float]:
    """Return ``(t_start_unix, t_end_unix)`` for one overview interval row."""
    lo = interval_row_t_start_unix_seconds(iv)
    t_end = iv.get("t_end", None)
    if t_end is not None:
        hi = interval_row_t_start_unix_seconds(pd.Series({"t_start": t_end}))
    else:
        dur = iv.get("t_duration", 0.0)
        if hasattr(dur, "total_seconds"):
            dur = float(dur.total_seconds())
        else:
            dur = float(dur)
        hi = lo + dur
    return lo, hi


def _slice_detailed_df_for_interval(detailed_df: Optional[pd.DataFrame], iv: pd.Series) -> pd.DataFrame:
    """Slice parent ``detailed_df`` by interval ``t_start``/``t_end`` (same contract as timeline fetch)."""
    if detailed_df is None or len(detailed_df) == 0 or "t" not in detailed_df.columns:
        return pd.DataFrame()
    lo, hi = _interval_row_unix_bounds(iv)
    t = pd.to_numeric(detailed_df["t"], errors="coerce")
    mask = (t >= lo) & (t <= hi + _FETCH_DETAIL_T_END_INCLUSIVE_EPS)
    return detailed_df.loc[mask].copy()


def _series_overlaps_interval(t_arr: np.ndarray, lo: float, hi: float) -> bool:
    if t_arr.size == 0:
        return False
    t_lo = float(np.nanmin(t_arr))
    t_hi = float(np.nanmax(t_arr))
    return t_hi >= lo - _FETCH_DETAIL_T_END_INCLUSIVE_EPS and t_lo <= hi + _FETCH_DETAIL_T_END_INCLUSIVE_EPS


def _log_jaw_clench_interval_overlap(track_key: str, x_abs: np.ndarray, eeg_ds: Any) -> None:
    """Log whether the stitched jaw-clench series overlaps each parent interval row."""
    if x_abs.size == 0 or eeg_ds is None:
        return
    iv = getattr(eeg_ds, "intervals_df", None)
    if iv is None or len(iv) == 0:
        return
    try:
        t_series_lo, t_series_hi = float(np.nanmin(x_abs)), float(np.nanmax(x_abs))
        for j in range(len(iv)):
            row = iv.iloc[j]
            lo, hi = _interval_row_unix_bounds(row)
            overlap = t_series_hi >= lo - _FETCH_DETAIL_T_END_INCLUSIVE_EPS and t_series_lo <= hi + _FETCH_DETAIL_T_END_INCLUSIVE_EPS
            logger.info("%s: overlap interval[%s] t_start=%.3f t_end=%.3f vs series[%.3f, %.3f]: %s", track_key, j, lo, hi, t_series_lo, t_series_hi, overlap)
            if not overlap:
                logger.warning("%s: no samples in jaw-clench series overlap interval row %s (detail rects may be grey-only for that segment).", track_key, j)
    except Exception as ex:
        logger.debug("%s: interval vs series overlap check skipped: %s", track_key, ex)


def compute_jaw_clench_probability_series(samples_2d: np.ndarray, sfreq: float, t_unix: Optional[np.ndarray] = None, *, window_s: float = 0.25, step_s: float = 0.05, z_thresh: float = 3.0, min_sync_channels: int = 3, max_considered_channels: int = 6, viewport_t_min: Optional[float] = None, viewport_t_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sliding-window jaw-clench probability in ``[0, 1]``.

    Parameters
    ----------
    samples_2d
        Shape ``(n_channels, n_samples)`` in Volts.
    sfreq
        Sample rate in Hz.
    t_unix
        Optional per-sample unix timestamps (length ``n_samples``). Window centers use
        ``t_unix[center_idx]`` when provided; otherwise centers are raw-relative seconds.
    viewport_t_min, viewport_t_max
        When set, only return window centers whose time falls in this inclusive range
        (used for live viewport slicing while z-score stats use the full ``samples_2d``).
    """
    data = np.asarray(samples_2d, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("samples_2d must be 2D (n_channels, n_samples)")
    n_ch, n_samp = data.shape
    sf = float(sfreq)
    if sf <= 0.0 or n_ch == 0 or n_samp == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    w = max(1, int(round(window_s * sf)))
    s = max(1, int(round(step_s * sf)))
    if n_samp < w:
        return np.array([], dtype=float), np.array([], dtype=float)
    sw = sliding_window_view(data, w, axis=1)
    win_idxs = np.arange(0, sw.shape[1], s, dtype=int)
    if win_idxs.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    sw = sw[:, win_idxs, :]
    ptp = sw.max(axis=-1) - sw.min(axis=-1)
    ch_med = np.median(ptp, axis=1, keepdims=True)
    ch_mad = np.median(np.abs(ptp - ch_med), axis=1, keepdims=True)
    denom = np.maximum(ch_mad * 1.4826, 1e-12)
    z = (ptp - ch_med) / denom
    over = z > float(z_thresh)
    k = over.sum(axis=0)
    k_cap = np.minimum(k, min(max_considered_channels, n_ch))
    max_z = np.where(over, z, -np.inf).max(axis=0)
    max_z[np.isneginf(max_z)] = 0.0
    excess = np.clip(max_z - float(z_thresh), 0.0, None)
    denom_k = float(min(max_considered_channels, n_ch))
    p = np.clip((k_cap / denom_k) * np.tanh(excess / 3.0), 0.0, 1.0)
    centers = (win_idxs + w // 2).astype(int)
    if t_unix is not None:
        t_arr = np.asarray(t_unix, dtype=float)
        if t_arr.size != n_samp:
            raise ValueError(f"t_unix length ({t_arr.size}) must match n_samples ({n_samp})")
        t_centers = t_arr[np.clip(centers, 0, n_samp - 1)]
    else:
        t_centers = centers.astype(float) / sf
    if viewport_t_min is not None and viewport_t_max is not None:
        lo, hi = float(viewport_t_min), float(viewport_t_max)
        if lo > hi:
            lo, hi = hi, lo
        mask = (t_centers >= lo) & (t_centers <= hi)
        return t_centers[mask], p[mask]
    return t_centers, p


def compute_jaw_clench_probability_from_detailed_df(df: pd.DataFrame, channel_names: Sequence[str], sfreq: float, *, t_col: str = "t", viewport_t_min: Optional[float] = None, viewport_t_max: Optional[float] = None, **compute_kw: Any) -> pd.DataFrame:
    """Extract EEG channels from a timeline ``detailed_df`` and return ``t`` + ``jaw_clench_prob``."""
    if df is None or len(df) == 0 or t_col not in df.columns:
        return pd.DataFrame({t_col: [], JAW_CLENCH_PROB_COLUMN: []})
    available = [c for c in channel_names if c in df.columns]
    if not available:
        return pd.DataFrame({t_col: [], JAW_CLENCH_PROB_COLUMN: []})
    df_sorted = df.sort_values(t_col).reset_index(drop=True)
    t_unix = df_sorted[t_col].to_numpy(dtype=float)
    samples_2d = df_sorted[available].to_numpy(dtype=np.float64).T
    times, prob = compute_jaw_clench_probability_series(samples_2d, float(sfreq), t_unix=t_unix, viewport_t_min=viewport_t_min, viewport_t_max=viewport_t_max, **compute_kw)
    if times.size == 0:
        return pd.DataFrame({t_col: [], JAW_CLENCH_PROB_COLUMN: []})
    return pd.DataFrame({t_col: times, JAW_CLENCH_PROB_COLUMN: prob})


def compute_jaw_clench_probability_from_raw(raw_eeg: mne.io.BaseRaw, *, picks: str = "eeg", channel_names: Optional[Sequence[str]] = None, window_s: float = 0.25, step_s: float = 0.05, z_thresh: float = 3.0, min_sync_channels: int = 3, max_considered_channels: int = 6, copy_raw: bool = False) -> Dict[str, Any]:
    """Compute jaw-clench probability for one continuous raw segment."""
    raw = raw_eeg.copy() if copy_raw else raw_eeg
    raw.load_data()
    picks_idx = mne.pick_types(raw.info, eeg=True) if picks == "eeg" else mne.pick_channels(raw.info["ch_names"], include=picks)
    if picks_idx.size == 0 and channel_names:
        available = [c for c in channel_names if c in raw.ch_names]
        if available:
            picks_idx = mne.pick_channels(raw.info["ch_names"], include=available)
    if picks_idx.size == 0:
        raw_dur = float(raw.times[-1] - raw.times[0]) if len(raw.times) > 1 else 0.0
        logger.warning("compute_jaw_clench_probability_from_raw: no EEG picks (raw duration=%.3fs, ch_names=%s, channel_names=%s).", raw_dur, raw.ch_names[:8], list(channel_names)[:8] if channel_names else None)
        return dict(times=np.array([], dtype=float), jaw_clench_prob=np.array([], dtype=float), params=dict(window_s=window_s, step_s=step_s, z_thresh=z_thresh), n_windows=0)
    data = raw.get_data(picks=picks_idx, reject_by_annotation="omit")
    sfreq = float(raw.info["sfreq"])
    times_rel, prob = compute_jaw_clench_probability_series(data, sfreq, t_unix=None, window_s=window_s, step_s=step_s, z_thresh=z_thresh, min_sync_channels=min_sync_channels, max_considered_channels=max_considered_channels)
    if times_rel.size:
        times_rel = times_rel + float(raw.times[0])
    params = dict(window_s=window_s, step_s=step_s, z_thresh=z_thresh, min_sync_channels=min_sync_channels, max_considered_channels=max_considered_channels, sfreq=sfreq)
    return dict(times=times_rel, jaw_clench_prob=prob, params=params, n_windows=int(times_rel.size))


def intervals_df_for_jaw_clench_track(eeg_ds: Any, result: Mapping[str, Any], *, log_prefix: str) -> pd.DataFrame:
    """Subset parent EEG ``intervals_df`` when merge produced fewer stitched segments than parent rows."""
    iv = getattr(eeg_ds, "intervals_df", None)
    if iv is None or len(iv) == 0:
        raise ValueError(f"{log_prefix}: eeg_ds.intervals_df is missing or empty")
    n_full = len(iv)
    n_merge = result.get("merged_n_segments")
    if isinstance(n_merge, int) and n_merge > 0 and n_merge < n_full:
        logger.info("%s: jaw-clench track uses %s interval row(s) (merged_n_segments=%s; parent EEG has %s).", log_prefix, n_merge, n_merge, n_full)
        return iv.iloc[:n_merge].copy()
    return iv.copy()


def compute_jaw_clench_probability_merged_for_intervals(raws: Sequence[mne.io.BaseRaw], intervals_df: pd.DataFrame, *, eeg_ds: Any = None, parent_detailed_df: Optional[pd.DataFrame] = None, channel_names: Optional[Sequence[str]] = None, **compute_kw: Any) -> Dict[str, Any]:
    """Run :func:`compute_jaw_clench_probability_from_raw` per raw and stitch absolute unix times.

    When ``eeg_ds`` or ``parent_detailed_df`` is provided, each segment is validated against the
    parent EEG ``detailed_df`` slice for that interval; mis-anchored times are re-aligned to
    ``parent_slice['t'].min()`` so offline interval fetch can slice non-empty rows.
    """
    if parent_detailed_df is None and eeg_ds is not None:
        parent_detailed_df = getattr(eeg_ds, "detailed_df", None)
    if channel_names is None and eeg_ds is not None:
        channel_names = getattr(eeg_ds, "channel_names", None)
    raw_kw = dict(compute_kw)
    if channel_names is not None:
        raw_kw.setdefault("channel_names", channel_names)
    n_iv = len(intervals_df)
    n_raw = len(raws)
    n = min(n_raw, n_iv)
    if n_raw != n_iv:
        logger.warning("jaw_clench merge: raw count (%s) != interval count (%s); computing %s aligned segment(s).", n_raw, n_iv, n)
    if n == 0:
        raise ValueError("compute_jaw_clench_probability_merged_for_intervals: no raw/interval pairs (empty inputs).")
    all_t: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    last_params: Optional[Dict[str, Any]] = None
    for i in range(n):
        raw = raws[i]
        iv = intervals_df.iloc[i]
        anchor = interval_row_t_start_unix_seconds(iv)
        raw_t0 = float(np.asarray(raw.times)[0]) if len(raw.times) > 0 else 0.0
        sub = compute_jaw_clench_probability_from_raw(raw, **raw_kw)
        t_rel = np.asarray(sub["times"], dtype=float)
        p = np.asarray(sub[JAW_CLENCH_PROB_COLUMN], dtype=float)
        t_abs = raw_relative_times_to_timeline_unix(t_rel, interval_t_start_unix=anchor, raw_times_first=raw_t0)
        if parent_detailed_df is not None and t_rel.size > 0:
            parent_slice = _slice_detailed_df_for_interval(parent_detailed_df, iv)
            if len(parent_slice) > 0:
                p_t = pd.to_numeric(parent_slice["t"], errors="coerce")
                parent_t0 = float(p_t.min())
                parent_t1 = float(p_t.max())
                if np.isfinite(parent_t0) and np.isfinite(parent_t1) and not _series_overlaps_interval(t_abs, parent_t0, parent_t1):
                    logger.warning("jaw_clench merge: re-anchoring segment %s to parent detailed_df t0=%.3f (series was [%.3f, %.3f], parent [%.3f, %.3f]).", i, parent_t0, float(np.nanmin(t_abs)) if t_abs.size else float("nan"), float(np.nanmax(t_abs)) if t_abs.size else float("nan"), parent_t0, parent_t1)
                    t_abs = raw_relative_times_to_timeline_unix(t_rel, interval_t_start_unix=parent_t0, raw_times_first=raw_t0)
        all_t.append(t_abs)
        all_p.append(p)
        last_params = sub.get("params")
    times_out = np.concatenate(all_t) if len(all_t) > 1 else (all_t[0] if all_t else np.array([], dtype=float))
    prob_out = np.concatenate(all_p) if len(all_p) > 1 else (all_p[0] if all_p else np.array([], dtype=float))
    return dict(times=times_out, jaw_clench_prob=prob_out, params=last_params or {}, n_windows=len(times_out), times_are_absolute_unix=True, merged_n_segments=n)


class JawClenchProbabilityComputation(SpecificComputationBase):
    computation_id: ClassVar[str] = "jaw_clench_probability"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ()
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.stream


    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        if ctx.raw is None:
            raise ValueError("JawClenchProbabilityComputation requires ctx.raw")
        return compute_jaw_clench_probability_from_raw(ctx.raw, **dict(params))


def apply_jaw_clench_probability_to_timeline(timeline, result: Optional[Mapping[str, Any]] = None, *, eeg_name: str, eeg_ds: Any, t0: Optional[float] = None) -> Tuple[Any, Any, Any]:
    """Add or refresh the jaw-clench probability track on ``timeline``.

    For live LSL EEG (:class:`~pypho_timeline.rendering.datasources.specific.lsl.LiveEEGTrackDatasource`),
    ``result`` may be omitted; a live child datasource recomputes per viewport.

    For historical EEG, pass ``result`` from :func:`compute_jaw_clench_probability_merged_for_intervals`
    or :func:`compute_jaw_clench_probability_from_raw`.
    """
    from pypho_timeline.core.synchronized_plot_mode import SynchronizedPlotMode
    from pypho_timeline.rendering.datasources.specific.eeg import JawClenchProbabilityTrackDatasource, jaw_clench_track_key_for_eeg_datasource
    from pypho_timeline.rendering.helpers.normalization import ChannelNormalizationMode

    if eeg_ds is None:
        raise ValueError("apply_jaw_clench_probability_to_timeline requires eeg_ds (got None)")
    if eeg_name is None:
        raise ValueError("apply_jaw_clench_probability_to_timeline requires eeg_name (got None)")

    track_key = jaw_clench_track_key_for_eeg_datasource(eeg_ds)
    live_mode = False
    try:
        from pypho_timeline.rendering.datasources.specific.lsl import LiveEEGTrackDatasource, LiveJawClenchProbabilityTrackDatasource
        live_mode = isinstance(eeg_ds, LiveEEGTrackDatasource)
    except ImportError:
        LiveJawClenchProbabilityTrackDatasource = None  # type: ignore[misc, assignment]

    if track_key in timeline.track_renderers and hasattr(timeline, 'track_is_fully_attached') and timeline.track_is_fully_attached(track_key):
        logger.info("%s: refreshing existing track.", track_key)
        jaw_widget, jaw_track, jaw_ds = timeline.get_track_tuple(track_key)
        if live_mode and LiveJawClenchProbabilityTrackDatasource is not None:
            jaw_ds._source_eeg = eeg_ds  # type: ignore[attr-defined]
            if getattr(eeg_ds, "intervals_df", None) is not None:
                jaw_ds.intervals_df = eeg_ds.intervals_df.copy()
        elif result is not None:
            n_windows = int(result.get("n_windows", 0) or 0)
            if n_windows <= 0:
                raise ValueError(f"{track_key}: jaw-clench computation produced no probability windows (check EEG channel picks and interval alignment).")
            detailed = _result_to_detailed_df(result, eeg_ds, track_key, t0=t0)
            jaw_ds.intervals_df = intervals_df_for_jaw_clench_track(eeg_ds, result, log_prefix=track_key)
            jaw_ds.detailed_df = detailed
        jaw_ds.source_data_changed_signal.emit()
        return (jaw_widget, jaw_track, jaw_ds)

    if hasattr(timeline, 'teardown_orphaned_track'):
        timeline.teardown_orphaned_track(track_key)

    ref_name = eeg_ds.custom_datasource_name
    if live_mode and LiveJawClenchProbabilityTrackDatasource is not None:
        logger.info("%s: creating live jaw-clench probability track.", track_key)
        jaw_ds = LiveJawClenchProbabilityTrackDatasource(source_eeg=eeg_ds, parent=eeg_ds)
    else:
        if result is None:
            raise ValueError(f"{track_key}: historical apply requires result dict")
        n_windows = int(result.get("n_windows", 0) or 0)
        if n_windows <= 0:
            raise ValueError(f"{track_key}: jaw-clench computation produced no probability windows (check EEG channel picks and interval alignment).")
        detailed = _result_to_detailed_df(result, eeg_ds, track_key, t0=t0)
        iv_jaw = intervals_df_for_jaw_clench_track(eeg_ds, result, log_prefix=track_key)
        jaw_ds = JawClenchProbabilityTrackDatasource(intervals_df=iv_jaw, eeg_df=detailed, custom_datasource_name=track_key, max_points_per_second=64.0, enable_downsampling=True, channel_names=[JAW_CLENCH_PROB_COLUMN], normalize=False, fallback_normalization_mode=ChannelNormalizationMode.NONE, plot_pen_colors=["#e45756"], plot_pen_width=1.5, lab_obj_dict=getattr(eeg_ds, "lab_obj_dict", None), raw_datasets_dict=getattr(eeg_ds, "raw_datasets_dict", None))

    jaw_widget, _root, jaw_plot_item, _jaw_dock = timeline.add_new_embedded_pyqtgraph_render_plot_widget(name=jaw_ds.custom_datasource_name, dockSize=(500, 60), dockAddLocationOpts=["bottom"], sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA)
    if ref_name in timeline.ui.matplotlib_view_widgets:
        ref_plot = timeline.ui.matplotlib_view_widgets[ref_name].getRootPlotItem()
        x0v, x1v = ref_plot.getViewBox().viewRange()[0]
        jaw_plot_item.setXRange(x0v, x1v, padding=0)
    jaw_plot_item.setTitle("Jaw clench probability (EEG-derived EMG proxy)")
    jaw_plot_item.setLabel("bottom", "Time (unix s)")
    jaw_plot_item.setLabel("left", "P(jaw clench)")
    jaw_plot_item.setYRange(0, 1, padding=0.0)
    jaw_plot_item.showAxis("left")
    timeline.add_track(jaw_ds, name=jaw_ds.custom_datasource_name, plot_item=jaw_plot_item)
    jaw_widget.optionsPanel = jaw_widget.getOptionsPanel()
    if hasattr(_jaw_dock, "updateWidgetsHaveOptionsPanel"):
        _jaw_dock.updateWidgetsHaveOptionsPanel()
    return timeline.get_track_tuple(jaw_ds.custom_datasource_name)


def _result_to_detailed_df(result: Mapping[str, Any], eeg_ds: Any, track_key: str, *, t0: Optional[float]) -> pd.DataFrame:
    if result.get("times_are_absolute_unix"):
        x_abs = np.asarray(result["times"], dtype=float)
        logger.info("%s: using absolute unix times from merge (n=%s, t in [%.3f, %.3f]).", track_key, x_abs.size, float(np.nanmin(x_abs)) if x_abs.size else float("nan"), float(np.nanmax(x_abs)) if x_abs.size else float("nan"))
    elif t0 is not None:
        x_abs = float(t0) + np.asarray(result["times"], dtype=float)
    else:
        ru = result.get("t0_unix")
        if ru is not None and np.isfinite(float(ru)):
            x_abs = float(ru) + np.asarray(result["times"], dtype=float)
        else:
            eu = getattr(eeg_ds, "earliest_unix_timestamp", None)
            if eu is None:
                raise ValueError(f"{track_key}: cannot resolve t0 for jaw-clench series")
            x_abs = float(eu) + np.asarray(result["times"], dtype=float)
    y = np.asarray(result[JAW_CLENCH_PROB_COLUMN], dtype=float)
    _log_jaw_clench_interval_overlap(track_key, x_abs, eeg_ds)
    return pd.DataFrame({"t": x_abs, JAW_CLENCH_PROB_COLUMN: y})


__all__ = [
    "JAW_CLENCH_PROB_COLUMN",
    "JawClenchProbabilityComputation",
    "apply_jaw_clench_probability_to_timeline",
    "compute_jaw_clench_probability_from_detailed_df",
    "compute_jaw_clench_probability_from_raw",
    "compute_jaw_clench_probability_merged_for_intervals",
    "compute_jaw_clench_probability_series",
    "intervals_df_for_jaw_clench_track",
]
