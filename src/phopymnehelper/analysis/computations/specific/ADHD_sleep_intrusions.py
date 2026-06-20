"""ADHD / sleep-intrusion theta/delta band-power ratio from EEG with optional motion masking.

Sliding-window PSD over good EEG channels. Motion windows are excluded by intersecting against
``BAD_motion`` annotations derived from ``motion_df`` (when provided). Optional autoreject masking
is supported via :func:`phopymnehelper.analysis.computations.specific.bad_epochs.fit_autoreject_bad_sample_mask`.

Public surfaces:

- :func:`compute_theta_delta_sleep_intrusion_series` -- one raw → flat dict (``times`` are raw-relative seconds)
- :func:`compute_theta_delta_sleep_intrusion_merged_for_intervals` -- one row per aligned raw/interval → stitched dict (``times`` absolute unix when passed to timeline apply)
- :class:`ThetaDeltaSleepIntrusionComputation` -- DAG node wrapper for ``run_eeg_computations_graph``
- :func:`apply_adhd_sleep_intrusion_to_timeline` -- adds/refreshes the ``{eeg_ds_name}_theta_delta`` track

Typical usage::

    from phopymnehelper.analysis.computations.eeg_registry import run_eeg_computations_graph, session_fingerprint_for_raw_or_path
    from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import apply_adhd_sleep_intrusion_to_timeline

    result = run_eeg_computations_graph(eeg_raw, session=session_fingerprint_for_raw_or_path(eeg_raw), goals=("theta_delta_sleep_intrusion",))["theta_delta_sleep_intrusion"]
    apply_adhd_sleep_intrusion_to_timeline(timeline, result, eeg_name=eeg_name, eeg_ds=eeg_ds)
"""

from __future__ import annotations

from datetime import datetime
import logging
import warnings
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from scipy import signal

from phopymnehelper.analysis.computations.protocol import ArtifactKind, RunContext
from phopymnehelper.analysis.computations.specific.bad_epochs import fit_autoreject_bad_sample_mask
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase

logger = logging.getLogger(__name__)


ANALYSIS_TRACK_NAME: str = "ANALYSIS_theta_delta"  # legacy single-stream name; use theta_delta_track_key_for_eeg_datasource per EEG track
MOTION_BAD_DESC: str = "BAD_motion"


def interval_row_t_start_unix_seconds(iv: pd.Series) -> float:
    """Normalize a single overview interval row's ``t_start`` to unix seconds (float).

    Expect ``t_start`` as timezone-aware/arithmetic ``datetime``/``Timestamp``, or numeric
    **unix seconds**. Avoid mixing timelines (e.g. raw-relative floats) without
    ``raw_relative_times_to_timeline_unix`` first — mis-anchored merges yield empty slices in
    interval-masked fetch.
    """
    ts = iv["t_start"]
    if isinstance(ts, (datetime, pd.Timestamp)):
        dt = pd.Timestamp(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return float(dt.timestamp())
    out = float(pd.to_numeric(ts, errors="coerce"))
    if not np.isfinite(out):
        raise ValueError(f"Non-finite t_start in interval row: {ts!r}")
    return out


def raw_relative_times_to_timeline_unix(times_rel: np.ndarray, *, interval_t_start_unix: float, raw_times_first: float) -> np.ndarray:
    """Map MNE raw-relative window-center times to absolute unix seconds on the timeline.

    ``interval_t_start_unix`` is the unix time of the first sample of that segment (matches
    ``intervals_df`` ``t_start``). ``raw_times_first`` is ``float(raw.times[0])`` (usually 0).
    """
    return float(interval_t_start_unix) + (np.asarray(times_rel, dtype=float) - float(raw_times_first))


def intervals_df_for_theta_delta_track(eeg_ds: Any, result: Mapping[str, Any], *, log_prefix: str) -> pd.DataFrame:
    """Subset parent EEG ``intervals_df`` when merge produced fewer stitched segments than parent rows.

    When ``aligned_chronological_raws_for_intervals`` returns fewer raws than ``intervals_df`` rows,
    :func:`compute_theta_delta_sleep_intrusion_merged_for_intervals` only fills times for ``n`` paired
    segments but the naive track would copy the full ``intervals_df``, leaving trailing intervals with
    no ``detailed_df`` samples — each fetch masks to zero rows.

    Uses ``result['merged_n_segments']`` from the merged compute path only; otherwise returns a full copy.
    """
    iv = getattr(eeg_ds, "intervals_df", None)
    if iv is None or len(iv) == 0:
        raise ValueError(f"{log_prefix}: eeg_ds.intervals_df is missing or empty")
    n_full = len(iv)
    n_merge = result.get("merged_n_segments")
    if isinstance(n_merge, int) and n_merge > 0 and n_merge < n_full:
        logger.info("%s: theta/delta track uses %s interval row(s) (merged_n_segments=%s; parent EEG has %s).", log_prefix, n_merge, n_merge, n_full)
        return iv.iloc[:n_merge].copy()
    return iv.copy()


def compute_theta_delta_sleep_intrusion_merged_for_intervals(raws: Sequence[mne.io.BaseRaw], intervals_df: pd.DataFrame, *, motion_df: Optional[pd.DataFrame] = None, **compute_kw: Any) -> Dict[str, Any]:
    """Run :func:`compute_theta_delta_sleep_intrusion_series` per raw, stitch into one absolute-time series.

    Assumes ``raws[i]`` corresponds to ``intervals_df.iloc[i]`` (same contract as
    ``RawProvidingTrackDatasource.aligned_chronological_raws_for_intervals``). Returned
    ``times`` are **absolute unix seconds**; set ``times_are_absolute_unix`` so
    :func:`apply_adhd_sleep_intrusion_to_timeline` does not add ``t0`` again.

    Orphan interval rows or raws beyond ``min(n_raws, n_intervals)`` are ignored; a warning is logged.
    """
    n_iv = len(intervals_df)
    n_raw = len(raws)
    n = min(n_raw, n_iv)
    if n_raw != n_iv:
        logger.warning("theta_delta merge: raw count (%s) != interval count (%s); computing %s aligned segment(s).", n_raw, n_iv, n)
    if n == 0:
        raise ValueError("compute_theta_delta_sleep_intrusion_merged_for_intervals: no raw/interval pairs (empty inputs).")

    all_t: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    last_params: Optional[Dict[str, Any]] = None
    motion_df_out_last: Optional[pd.DataFrame] = None

    for i in range(n):
        raw = raws[i]
        iv = intervals_df.iloc[i]
        anchor = interval_row_t_start_unix_seconds(iv)
        raw_t0 = float(np.asarray(raw.times)[0]) if len(raw.times) > 0 else 0.0
        sub = compute_theta_delta_sleep_intrusion_series(raw, motion_df=motion_df, meas_date=raw.info.get("meas_date"), **compute_kw)
        t_rel = np.asarray(sub["times"], dtype=float)
        y = np.asarray(sub["theta_delta_ratio"], dtype=float)
        t_abs = raw_relative_times_to_timeline_unix(t_rel, interval_t_start_unix=anchor, raw_times_first=raw_t0)
        all_t.append(t_abs)
        all_y.append(y)
        last_params = sub.get("params")
        if sub.get("motion_high_accel_df") is not None:
            motion_df_out_last = sub["motion_high_accel_df"]

        t_end_iv = anchor + float(iv["t_duration"])
        if t_abs.size:
            t_min, t_max = float(np.nanmin(t_abs)), float(np.nanmax(t_abs))
            logger.info("theta_delta segment %s/%s: interval t_start=%.3f t_end~=%.3f | series t [%.3f, %.3f] (n=%s)", i + 1, n, anchor, t_end_iv, t_min, t_max, t_abs.size)
            if t_max < anchor - 1.0 or t_min > t_end_iv + 1.0:
                logger.warning("theta_delta segment %s: series unix range may not overlap interval bounds (check t_start vs raw).", i + 1)

    times_out = np.concatenate(all_t) if len(all_t) > 1 else all_t[0]
    ratio_out = np.concatenate(all_y) if len(all_y) > 1 else all_y[0]
    n_valid = int(np.sum(np.isfinite(ratio_out)))
    session_mean = float(np.nanmean(ratio_out)) if n_valid > 0 else float("nan")

    return dict(times=times_out, theta_delta_ratio=ratio_out, session_mean_theta_delta=session_mean, n_windows=len(times_out), n_valid_windows=n_valid, t0_unix=None, motion_high_accel_df=motion_df_out_last, params=last_params or {}, times_are_absolute_unix=True, merged_n_segments=n)


def _good_picks(raw: mne.io.BaseRaw) -> np.ndarray:
    """Return indices of good channels for spectral analysis with sane fallbacks.

    First tries EEG-typed channels; if none, falls back to all non-bad data channels by name.
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if picks.size > 0:
        return picks
    bads = set(raw.info.get("bads") or [])
    return np.asarray([i for i, ch in enumerate(raw.ch_names) if ch not in bads], dtype=int)


def _psd(sig_2d: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from mne.time_frequency import psd_array_multitaper
        out = psd_array_multitaper(sig_2d, sfreq, adaptive=True, verbose="ERROR")
        return np.asarray(out[0]), np.asarray(out[1])
    except Exception:
        n = sig_2d.shape[1]
        nperseg = min(int(sfreq * 2), n) if n >= 8 else n
        if nperseg < 4:
            raise
        freqs, psd = signal.welch(sig_2d, fs=sfreq, nperseg=nperseg, axis=-1, detrend="linear")
        return psd, freqs


def _bandpower(psd_1d: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> float:
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return float("nan")
    return float(np.trapz(psd_1d[m], freqs[m]))


def compute_theta_delta_sleep_intrusion_series(raw_eeg: mne.io.BaseRaw, *, motion_df: Optional[pd.DataFrame] = None, total_accel_threshold: float = 0.5, minimum_motion_bad_duration: float = 0.05, meas_date: Any = None, l_freq: float = 1.0, h_freq: Optional[float] = 40.0, window_sec: float = 4.0, step_sec: float = 1.0, delta_band: Tuple[float, float] = (1.0, 4.0), theta_band: Tuple[float, float] = (4.0, 8.0), use_autoreject: bool = False, autoreject_epoch_sec: float = 3.0, autoreject_kwargs: Optional[Mapping[str, Any]] = None, channel_agg: str = "mean", copy_raw: bool = True) -> Dict[str, Any]:
    """Compute the sliding-window theta/delta band-power ratio.

    Returns a flat dict with ``times``, ``theta_delta_ratio``, ``session_mean_theta_delta``,
    ``n_windows``, ``n_valid_windows``, ``motion_high_accel_df``, ``params``.
    """
    if channel_agg not in ("mean", "median"):
        raise ValueError("channel_agg must be 'mean' or 'median'")

    raw = raw_eeg.copy() if copy_raw else raw_eeg
    raw.load_data()

    nyq = 0.5 * float(raw.info["sfreq"])
    eff_h_freq = h_freq
    if eff_h_freq is not None and eff_h_freq >= nyq:
        eff_h_freq = max(float(l_freq) + 1.0, nyq - 1.0)
        warnings.warn(f"h_freq {h_freq} >= Nyquist ({nyq}); using {eff_h_freq}", RuntimeWarning, stacklevel=2)

    motion_df_out: Optional[pd.DataFrame] = None
    if motion_df is not None:
        from phopymnehelper.motion_data import MotionData
        md = meas_date if meas_date is not None else raw.info.get("meas_date")
        motion_annots, motion_df_out = MotionData.find_high_accel_periods(a_ds=motion_df, total_accel_threshold=total_accel_threshold, should_set_bad_period_annotations=False, minimum_bad_duration=minimum_motion_bad_duration, meas_date=md)
        cur = raw.annotations
        raw.set_annotations(motion_annots if (cur is None or len(cur) == 0) else cur + motion_annots)

    raw.filter(l_freq=l_freq, h_freq=eff_h_freq, verbose=False)

    picks = _good_picks(raw)
    if picks.size == 0:
        raise ValueError(f"No usable channels in raw (n_channels={len(raw.ch_names)}, bads={raw.info.get('bads')})")

    ar_mask: Optional[np.ndarray] = fit_autoreject_bad_sample_mask(raw, autoreject_epoch_sec=autoreject_epoch_sec, autoreject_kwargs=autoreject_kwargs) if use_autoreject else None

    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    tmax = float(raw.times[-1])

    annots = raw.annotations
    motion_intervals: list = []
    if annots is not None and len(annots) > 0:
        for k in range(len(annots)):
            desc = str(annots.description[k])
            if MOTION_BAD_DESC not in desc:
                continue
            t0_a = float(annots.onset[k])
            motion_intervals.append((t0_a, t0_a + float(annots.duration[k])))

    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be positive")

    times_list: list = []
    ratio_list: list = []
    eps = 1e-10
    min_samples = max(8, int(0.5 * window_sec * sfreq))
    t0 = 0.0
    while t0 + window_sec <= tmax + 1e-9:
        t1 = t0 + window_sec
        tc = 0.5 * (t0 + t1)
        times_list.append(tc)

        hits_motion = any((t0 < a1 and t1 > a0) for a0, a1 in motion_intervals)
        if hits_motion:
            ratio_list.append(float("nan"))
            t0 += step_sec
            continue

        if ar_mask is not None:
            i0 = max(0, int(np.floor(t0 * sfreq)))
            i1 = min(n_times, int(np.ceil(t1 * sfreq)))
            if i1 > i0 and bool(np.any(ar_mask[i0:i1])):
                ratio_list.append(float("nan"))
                t0 += step_sec
                continue

        s0 = max(0, int(np.floor(t0 * sfreq)))
        s1 = min(n_times, int(np.ceil(t1 * sfreq)))
        if s1 - s0 < min_samples:
            ratio_list.append(float("nan"))
            t0 += step_sec
            continue

        block = raw.get_data(picks=picks, start=s0, stop=s1)
        agg = np.mean(block, axis=0, keepdims=True) if channel_agg == "mean" else np.median(block, axis=0, keepdims=True)
        try:
            psd, freqs = _psd(agg, sfreq)
        except Exception:
            ratio_list.append(float("nan"))
            t0 += step_sec
            continue

        p_delta = _bandpower(psd[0], freqs, delta_band)
        p_theta = _bandpower(psd[0], freqs, theta_band)
        if not np.isfinite(p_delta) or not np.isfinite(p_theta):
            ratio_list.append(float("nan"))
            t0 += step_sec
            continue

        ratio_list.append(float(p_theta / (p_delta + eps)))
        t0 += step_sec

    times_arr = np.asarray(times_list, dtype=float)
    ratio_arr = np.asarray(ratio_list, dtype=float)
    n_valid = int(np.sum(np.isfinite(ratio_arr)))
    session_mean = float(np.nanmean(ratio_arr)) if n_valid > 0 else float("nan")

    md = raw.info.get("meas_date")
    t0_unix: Optional[float] = None
    if md is not None:
        try:
            t0_unix = float(md.timestamp()) if hasattr(md, "timestamp") else float(md)
        except (TypeError, ValueError, OSError):
            t0_unix = None

    params = dict(total_accel_threshold=total_accel_threshold, minimum_motion_bad_duration=minimum_motion_bad_duration, l_freq=l_freq, h_freq=eff_h_freq, h_freq_requested=h_freq, window_sec=window_sec, step_sec=step_sec, delta_band=tuple(delta_band), theta_band=tuple(theta_band), use_autoreject=use_autoreject, autoreject_epoch_sec=autoreject_epoch_sec, channel_agg=channel_agg, n_picks=int(picks.size))

    return dict(times=times_arr, theta_delta_ratio=ratio_arr, session_mean_theta_delta=session_mean, n_windows=len(times_arr), n_valid_windows=n_valid, t0_unix=t0_unix, motion_high_accel_df=motion_df_out, params=params)


class ThetaDeltaSleepIntrusionComputation(SpecificComputationBase):
    computation_id: ClassVar[str] = "theta_delta_sleep_intrusion"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ()
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.stream


    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        if ctx.raw is None:
            raise ValueError("ThetaDeltaSleepIntrusionComputation requires ctx.raw")
        return compute_theta_delta_sleep_intrusion_series(ctx.raw, **dict(params))


def apply_adhd_sleep_intrusion_to_timeline(timeline, result: Mapping[str, Any], *, eeg_name: str, eeg_ds: Any, t0: Optional[float] = None) -> Tuple[Any, Any, Any]:
    """Add or refresh the per-EEG theta/delta track on ``timeline`` from a compute result.

    Track name is ``{eeg_ds.custom_datasource_name}_theta_delta`` (see
    ``pypho_timeline.rendering.datasources.specific.eeg.theta_delta_track_key_for_eeg_datasource``).

    When ``result.get('times_are_absolute_unix')`` is truthy, ``result['times']`` are already unix
    seconds on the timeline axis and ``t0`` / ``t0_unix`` / ``earliest_unix_timestamp`` are not applied.

    Otherwise ``t0`` resolution order: explicit kwarg > ``result['t0_unix']`` (raw ``meas_date`` from
    the compute) > ``eeg_ds.earliest_unix_timestamp``.

    Interval overview rows (``intervals_df`` on the datasource) carry only rectangle bounds for the track;
    the plotted theta/delta series is stored in ``detailed_df`` (column ``t`` + ``theta_delta``), not
    as columns on ``intervals_df``.
    """
    from pypho_timeline.core.synchronized_plot_mode import SynchronizedPlotMode
    from pypho_timeline.rendering.datasources.specific.eeg import ThetaDeltaSleepIntrusionTrackDatasource, theta_delta_track_key_for_eeg_datasource

    if eeg_ds is None:
        raise ValueError("apply_adhd_sleep_intrusion_to_timeline requires eeg_ds (got None)")
    if eeg_name is None:
        raise ValueError("apply_adhd_sleep_intrusion_to_timeline requires eeg_name (got None)")

    track_key = theta_delta_track_key_for_eeg_datasource(eeg_ds)
    if result.get("times_are_absolute_unix"):
        x_abs = np.asarray(result["times"], dtype=float)
        logger.info("%s: using absolute unix times from merge (n=%s, t in [%.3f, %.3f]).", track_key, x_abs.size, float(np.nanmin(x_abs)) if x_abs.size else float("nan"), float(np.nanmax(x_abs)) if x_abs.size else float("nan"))
    elif t0 is not None:
        t0_eff: float = float(t0)
        logger.info("%s: t0 from argument: %s", track_key, t0_eff)
        x_abs = t0_eff + np.asarray(result["times"], dtype=float)
    else:
        ru = result.get("t0_unix")
        if ru is not None and np.isfinite(float(ru)):
            t0_eff = float(ru)
            logger.info("%s: t0 from result['t0_unix']: %s", track_key, t0_eff)
        else:
            eu = getattr(eeg_ds, "earliest_unix_timestamp", None)
            if eu is None:
                raise ValueError(f"{track_key}: cannot resolve t0 (no kwarg, result['t0_unix'], or eeg_ds.earliest_unix_timestamp)")
            t0_eff = float(eu)
            logger.warning("%s: no explicit t0 / valid result['t0_unix']; falling back to eeg_ds.earliest_unix_timestamp=%s", track_key, t0_eff)

        x_abs = t0_eff + np.asarray(result["times"], dtype=float)
    y = np.asarray(result["theta_delta_ratio"], dtype=float)

    if x_abs.size and len(getattr(eeg_ds, "intervals_df", [])):
        try:
            t_series_lo, t_series_hi = float(np.nanmin(x_abs)), float(np.nanmax(x_abs))
            for j in range(len(eeg_ds.intervals_df)):
                iv = eeg_ds.intervals_df.iloc[j]
                lo = interval_row_t_start_unix_seconds(iv)
                hi = lo + float(iv["t_duration"])
                overlap = t_series_hi >= lo - 1e-3 and t_series_lo <= hi + 1e-3
                logger.info("%s: overlap interval[%s] t_start=%.3f t_end=%.3f vs series[%s,%s]: %s", track_key, j, lo, hi, t_series_lo, t_series_hi, overlap)
                if not overlap:
                    logger.warning("%s: no samples in theta/delta series overlap interval row %s (detail rects may be grey-only for that segment).", track_key, j)
        except Exception as ex:
            logger.debug("%s: interval vs series overlap check skipped: %s", track_key, ex)

    detailed = pd.DataFrame({"t": x_abs, "theta_delta": y})
    iv_theta = intervals_df_for_theta_delta_track(eeg_ds, result, log_prefix=track_key)

    if track_key in timeline.track_renderers and hasattr(timeline, 'track_is_fully_attached') and timeline.track_is_fully_attached(track_key):
        logger.info("%s: refreshing existing track.", track_key)
        td_ratio_widget, td_ratio_track, td_ratio_ds = timeline.get_track_tuple(track_key)
        td_ratio_ds.intervals_df = iv_theta.copy()
        td_ratio_ds.detailed_df = detailed
        td_ratio_ds.source_data_changed_signal.emit()
        return (td_ratio_widget, td_ratio_track, td_ratio_ds)

    if hasattr(timeline, 'teardown_orphaned_track'):
        timeline.teardown_orphaned_track(track_key)

    logger.info("%s: creating new track (n=%s, x_range=[%s, %s]).", track_key, len(detailed), x_abs.min() if x_abs.size else float("nan"), x_abs.max() if x_abs.size else float("nan"))
    td_ratio_ds = ThetaDeltaSleepIntrusionTrackDatasource(intervals_df=iv_theta, eeg_df=detailed, custom_datasource_name=track_key, max_points_per_second=64.0, enable_downsampling=True, channel_names=["theta_delta"], normalize=True, normalize_over_full_data=True, plot_pen_colors=["#9467bd"], plot_pen_width=1.2, lab_obj_dict=getattr(eeg_ds, "lab_obj_dict", None), raw_datasets_dict=getattr(eeg_ds, "raw_datasets_dict", None))
    td_ratio_widget, _root, ratio_plot_item, _ratio_dock = timeline.add_new_embedded_pyqtgraph_render_plot_widget(name=td_ratio_ds.custom_datasource_name, dockSize=(500, 60), dockAddLocationOpts=["bottom"], sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA)

    ## Set range to same range as EEG track -- this should be good
    ref_name = eeg_ds.custom_datasource_name
    if ref_name in timeline.ui.matplotlib_view_widgets:
        ref_plot = timeline.ui.matplotlib_view_widgets[ref_name].getRootPlotItem()
        x0v, x1v = ref_plot.getViewBox().viewRange()[0]
        ratio_plot_item.setXRange(x0v, x1v, padding=0)

    ratio_plot_item.setTitle("ADHD sleep intrusion series (theta / delta, NaN = motion/QC excluded)")
    ratio_plot_item.setLabel("bottom", "Time (unix s)")
    ratio_plot_item.setLabel("left", "theta / delta (norm.)")
    # ratio_plot_item.setYRange(0, 1, padding=0.0)
    ratio_plot_item.showAxis("left")
    timeline.add_track(td_ratio_ds, name=td_ratio_ds.custom_datasource_name, plot_item=ratio_plot_item)
    td_ratio_widget.optionsPanel = td_ratio_widget.getOptionsPanel()

    ## get the returned values:
    td_ratio_widget, td_ratio_track, td_ratio_ds = timeline.get_track_tuple(td_ratio_ds.custom_datasource_name)
    # detailed_td_ratio_df: pd.DataFrame = td_ratio_ds.detailed_df
    return (td_ratio_widget, td_ratio_track, td_ratio_ds)

__all__ = [
    "ANALYSIS_TRACK_NAME",
    "ThetaDeltaSleepIntrusionComputation",
    "apply_adhd_sleep_intrusion_to_timeline",
    "compute_theta_delta_sleep_intrusion_merged_for_intervals",
    "compute_theta_delta_sleep_intrusion_series",
    "interval_row_t_start_unix_seconds",
    "intervals_df_for_theta_delta_track",
    "raw_relative_times_to_timeline_unix",
]
