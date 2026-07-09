"""Frontal-midline theta (FMT) power (5–9 Hz) from EEG with optional motion masking.

Sliding-window PSD over frontal-midline channels (Fz preferred; Emotiv AF3/AF4 proxy).
Motion windows are excluded by intersecting against ``BAD_motion`` annotations derived from
``motion_df`` (when provided). Optional autoreject masking is supported via
:func:`phopymnehelper.analysis.computations.specific.bad_epochs.fit_autoreject_bad_sample_mask`.

Primary spectral marker of prolonged wakefulness (Marzano et al., SLEEP 2013).

Public surfaces:

- :func:`compute_frontal_midline_theta_series` -- one raw → flat dict (``times`` are raw-relative seconds)
- :func:`compute_frontal_midline_theta_merged_for_intervals` -- one row per aligned raw/interval → stitched dict
- :class:`FrontalMidlineThetaComputation` -- DAG node wrapper for ``run_eeg_computations_graph``
- :func:`apply_frontal_midline_theta_to_timeline` -- adds/refreshes the ``{eeg_ds_name}_frontal_midline_theta`` track

Typical usage::

    from phopymnehelper.analysis.computations.eeg_registry import run_eeg_computations_graph, session_fingerprint_for_raw_or_path
    from phopymnehelper.analysis.computations.specific.frontal_midline_theta import apply_frontal_midline_theta_to_timeline

    result = run_eeg_computations_graph(eeg_raw, session=session_fingerprint_for_raw_or_path(eeg_raw), goals=("frontal_midline_theta",))["frontal_midline_theta"]
    apply_frontal_midline_theta_to_timeline(timeline, result, eeg_name=eeg_name, eeg_ds=eeg_ds)
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from scipy import signal

from phopymnehelper.analysis.computations.protocol import ArtifactKind, RunContext
from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import (
    MOTION_BAD_DESC,
    interval_row_t_start_unix_seconds,
    intervals_df_for_theta_delta_track,
    raw_relative_times_to_timeline_unix,
)
from phopymnehelper.analysis.computations.specific.bad_epochs import fit_autoreject_bad_sample_mask
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase

logger = logging.getLogger(__name__)

DEFAULT_FMT_BAND: Tuple[float, float] = (5.0, 9.0)
DEFAULT_FMT_L_FREQ: float = 4.0
DEFAULT_FMT_H_FREQ: float = 12.0

# Priority: exact midline Fz, then Emotiv AF3/AF4 proxy, then nearby frontal fallbacks.
_FZ_NAMES: Tuple[str, ...] = ("FZ", "Fz", "fz")
_AF_PAIR: Tuple[str, ...] = ("AF3", "AF4")
_FRONTAL_FALLBACK: Tuple[str, ...] = ("FP1", "FP2", "F3", "F4", "Fp1", "Fp2")

FRONTAL_MIDLINE_THETA_PARAM_KEYS: FrozenSet[str] = frozenset(
    {
        "total_accel_threshold",
        "minimum_motion_bad_duration",
        "meas_date",
        "l_freq",
        "h_freq",
        "window_sec",
        "step_sec",
        "fmt_band",
        "use_autoreject",
        "autoreject_epoch_sec",
        "autoreject_kwargs",
        "channel_agg",
        "copy_raw",
        "motion_df",
        "t0",
    }
)


def filter_frontal_midline_theta_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: params[k] for k in FRONTAL_MIDLINE_THETA_PARAM_KEYS if k in params}


def frontal_midline_theta_params_fingerprint(params: Mapping[str, Any]) -> str:
    f = filter_frontal_midline_theta_params(params)
    # motion_df is not JSON-stable; fingerprint presence only
    if "motion_df" in f:
        f = dict(f)
        f["motion_df"] = f["motion_df"] is not None
    return json.dumps({k: f[k] for k in sorted(f.keys())}, sort_keys=True, default=str)


def resolve_fmt_channel_picks(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, List[str]]:
    """Resolve frontal-midline channel indices and names for FMT power.

    Priority:
    1. FZ / Fz if present
    2. Mean of available AF3 + AF4 (Emotiv midline proxy)
    3. First available from FP1, FP2, F3, F4
    """
    name_to_idx = {ch: i for i, ch in enumerate(raw.ch_names)}
    bads = set(raw.info.get("bads") or [])

    for fz in _FZ_NAMES:
        if fz in name_to_idx and fz not in bads:
            return np.asarray([name_to_idx[fz]], dtype=int), [fz]
    ## END for fz in _FZ_NAMES....

    af_present = [ch for ch in _AF_PAIR if ch in name_to_idx and ch not in bads]
    if len(af_present) >= 1:
        return np.asarray([name_to_idx[ch] for ch in af_present], dtype=int), list(af_present)

    for fb in _FRONTAL_FALLBACK:
        if fb in name_to_idx and fb not in bads:
            return np.asarray([name_to_idx[fb]], dtype=int), [fb]
    ## END for fb in _FRONTAL_FALLBACK....

    # Last resort: any good EEG channel
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if picks.size > 0:
        names = [raw.ch_names[i] for i in picks[:1]]
        return picks[:1], names
    raise ValueError(f"No usable frontal-midline channels in raw (ch_names={list(raw.ch_names)}, bads={list(bads)})")


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


def compute_frontal_midline_theta_merged_for_intervals(
    raws: Sequence[mne.io.BaseRaw],
    intervals_df: pd.DataFrame,
    *,
    motion_df: Optional[pd.DataFrame] = None,
    **compute_kw: Any,
) -> Dict[str, Any]:
    """Run :func:`compute_frontal_midline_theta_series` per raw, stitch into one absolute-time series.

    Assumes ``raws[i]`` corresponds to ``intervals_df.iloc[i]``. Returned ``times`` are
    **absolute unix seconds**; set ``times_are_absolute_unix`` so
    :func:`apply_frontal_midline_theta_to_timeline` does not add ``t0`` again.
    """
    n_iv = len(intervals_df)
    n_raw = len(raws)
    n = min(n_raw, n_iv)
    if n_raw != n_iv:
        logger.warning("fmt merge: raw count (%s) != interval count (%s); computing %s aligned segment(s).", n_raw, n_iv, n)
    if n == 0:
        raise ValueError("compute_frontal_midline_theta_merged_for_intervals: no raw/interval pairs (empty inputs).")

    all_t: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    last_params: Optional[Dict[str, Any]] = None
    motion_df_out_last: Optional[pd.DataFrame] = None

    for i in range(n):
        raw = raws[i]
        iv = intervals_df.iloc[i]
        anchor = interval_row_t_start_unix_seconds(iv)
        raw_t0 = float(np.asarray(raw.times)[0]) if len(raw.times) > 0 else 0.0
        sub = compute_frontal_midline_theta_series(raw, motion_df=motion_df, meas_date=raw.info.get("meas_date"), **compute_kw)
        t_rel = np.asarray(sub["times"], dtype=float)
        y = np.asarray(sub["fmt_power"], dtype=float)
        t_abs = raw_relative_times_to_timeline_unix(t_rel, interval_t_start_unix=anchor, raw_times_first=raw_t0)
        all_t.append(t_abs)
        all_y.append(y)
        last_params = sub.get("params")
        if sub.get("motion_high_accel_df") is not None:
            motion_df_out_last = sub["motion_high_accel_df"]

        t_end_iv = anchor + float(iv["t_duration"])
        if t_abs.size:
            t_min, t_max = float(np.nanmin(t_abs)), float(np.nanmax(t_abs))
            logger.info(
                "fmt segment %s/%s: interval t_start=%.3f t_end~=%.3f | series t [%.3f, %.3f] (n=%s)",
                i + 1,
                n,
                anchor,
                t_end_iv,
                t_min,
                t_max,
                t_abs.size,
            )
            if t_max < anchor - 1.0 or t_min > t_end_iv + 1.0:
                logger.warning("fmt segment %s: series unix range may not overlap interval bounds (check t_start vs raw).", i + 1)
        ## END for i in range(n)....

    times_out = np.concatenate(all_t) if len(all_t) > 1 else all_t[0]
    power_out = np.concatenate(all_y) if len(all_y) > 1 else all_y[0]
    n_valid = int(np.sum(np.isfinite(power_out)))
    session_mean = float(np.nanmean(power_out)) if n_valid > 0 else float("nan")

    return dict(
        times=times_out,
        fmt_power=power_out,
        session_mean_fmt_power=session_mean,
        n_windows=len(times_out),
        n_valid_windows=n_valid,
        t0_unix=None,
        motion_high_accel_df=motion_df_out_last,
        params=last_params or {},
        times_are_absolute_unix=True,
        merged_n_segments=n,
    )


def compute_frontal_midline_theta_series(
    raw_eeg: mne.io.BaseRaw,
    *,
    motion_df: Optional[pd.DataFrame] = None,
    total_accel_threshold: float = 0.5,
    minimum_motion_bad_duration: float = 0.05,
    meas_date: Any = None,
    l_freq: float = DEFAULT_FMT_L_FREQ,
    h_freq: Optional[float] = DEFAULT_FMT_H_FREQ,
    window_sec: float = 4.0,
    step_sec: float = 1.0,
    fmt_band: Tuple[float, float] = DEFAULT_FMT_BAND,
    use_autoreject: bool = False,
    autoreject_epoch_sec: float = 3.0,
    autoreject_kwargs: Optional[Mapping[str, Any]] = None,
    channel_agg: str = "mean",
    copy_raw: bool = True,
) -> Dict[str, Any]:
    """Compute sliding-window frontal-midline theta band power (default 5–9 Hz).

    Returns a flat dict with ``times``, ``fmt_power``, ``session_mean_fmt_power``,
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
        motion_annots, motion_df_out = MotionData.find_high_accel_periods(
            a_ds=motion_df,
            total_accel_threshold=total_accel_threshold,
            should_set_bad_period_annotations=False,
            minimum_bad_duration=minimum_motion_bad_duration,
            meas_date=md,
        )
        cur = raw.annotations
        raw.set_annotations(motion_annots if (cur is None or len(cur) == 0) else cur + motion_annots)

    picks, fmt_channels = resolve_fmt_channel_picks(raw)
    # Bandpass only the selected channels for efficiency
    raw.filter(l_freq=l_freq, h_freq=eff_h_freq, picks=picks, verbose=False)

    ar_mask: Optional[np.ndarray] = (
        fit_autoreject_bad_sample_mask(raw, autoreject_epoch_sec=autoreject_epoch_sec, autoreject_kwargs=autoreject_kwargs)
        if use_autoreject
        else None
    )

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
        ## END for k in range(len(annots))....

    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be positive")

    times_list: list = []
    power_list: list = []
    min_samples = max(8, int(0.5 * window_sec * sfreq))
    t0 = 0.0
    while t0 + window_sec <= tmax + 1e-9:
        t1 = t0 + window_sec
        tc = 0.5 * (t0 + t1)
        times_list.append(tc)

        hits_motion = any((t0 < a1 and t1 > a0) for a0, a1 in motion_intervals)
        if hits_motion:
            power_list.append(float("nan"))
            t0 += step_sec
            continue

        if ar_mask is not None:
            i0 = max(0, int(np.floor(t0 * sfreq)))
            i1 = min(n_times, int(np.ceil(t1 * sfreq)))
            if i1 > i0 and bool(np.any(ar_mask[i0:i1])):
                power_list.append(float("nan"))
                t0 += step_sec
                continue

        s0 = max(0, int(np.floor(t0 * sfreq)))
        s1 = min(n_times, int(np.ceil(t1 * sfreq)))
        if s1 - s0 < min_samples:
            power_list.append(float("nan"))
            t0 += step_sec
            continue

        block = raw.get_data(picks=picks, start=s0, stop=s1)
        agg = np.mean(block, axis=0, keepdims=True) if channel_agg == "mean" else np.median(block, axis=0, keepdims=True)
        try:
            psd, freqs = _psd(agg, sfreq)
        except Exception:
            power_list.append(float("nan"))
            t0 += step_sec
            continue

        p_fmt = _bandpower(psd[0], freqs, fmt_band)
        power_list.append(p_fmt if np.isfinite(p_fmt) else float("nan"))
        t0 += step_sec

    times_arr = np.asarray(times_list, dtype=float)
    power_arr = np.asarray(power_list, dtype=float)
    n_valid = int(np.sum(np.isfinite(power_arr)))
    session_mean = float(np.nanmean(power_arr)) if n_valid > 0 else float("nan")

    md = raw.info.get("meas_date")
    t0_unix: Optional[float] = None
    if md is not None:
        try:
            t0_unix = float(md.timestamp()) if hasattr(md, "timestamp") else float(md)
        except (TypeError, ValueError, OSError):
            t0_unix = None

    params = dict(
        total_accel_threshold=total_accel_threshold,
        minimum_motion_bad_duration=minimum_motion_bad_duration,
        l_freq=l_freq,
        h_freq=eff_h_freq,
        h_freq_requested=h_freq,
        window_sec=window_sec,
        step_sec=step_sec,
        fmt_band=tuple(fmt_band),
        use_autoreject=use_autoreject,
        autoreject_epoch_sec=autoreject_epoch_sec,
        channel_agg=channel_agg,
        fmt_channels=list(fmt_channels),
        n_picks=int(picks.size),
    )

    return dict(
        times=times_arr,
        fmt_power=power_arr,
        session_mean_fmt_power=session_mean,
        n_windows=len(times_arr),
        n_valid_windows=n_valid,
        t0_unix=t0_unix,
        motion_high_accel_df=motion_df_out,
        params=params,
    )


class FrontalMidlineThetaComputation(SpecificComputationBase):
    computation_id: ClassVar[str] = "frontal_midline_theta"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ()
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.stream
    params_fingerprint_fn: ClassVar[Optional[Callable[[Mapping[str, Any]], str]]] = frontal_midline_theta_params_fingerprint

    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        if ctx.raw is None:
            raise ValueError("FrontalMidlineThetaComputation requires ctx.raw")
        kw = filter_frontal_midline_theta_params(params)
        kw.pop("t0", None)
        kw.pop("motion_df", None)
        # motion_df may still be passed via params for direct callers; prefer explicit if present
        motion_df = params.get("motion_df")
        return compute_frontal_midline_theta_series(ctx.raw, motion_df=motion_df, **{k: v for k, v in kw.items() if k != "motion_df"})


def apply_frontal_midline_theta_to_timeline(
    timeline,
    result: Mapping[str, Any],
    *,
    eeg_name: str,
    eeg_ds: Any,
    t0: Optional[float] = None,
    show_left_axis: bool = True,
    show_bottom_axis: bool = False,
    show_title_label: bool = False,
) -> Tuple[Any, Any, Any]:
    """Add or refresh the per-EEG frontal-midline theta track on ``timeline`` from a compute result.

    Track name is ``{eeg_ds.custom_datasource_name}_frontal_midline_theta``.
    """
    from pypho_timeline.core.synchronized_plot_mode import SynchronizedPlotMode
    from pypho_timeline.rendering.datasources.specific.eeg import (
        FrontalMidlineThetaTrackDatasource,
        frontal_midline_theta_track_key_for_eeg_datasource,
    )

    if eeg_ds is None:
        raise ValueError("apply_frontal_midline_theta_to_timeline requires eeg_ds (got None)")
    if eeg_name is None:
        raise ValueError("apply_frontal_midline_theta_to_timeline requires eeg_name (got None)")

    track_key = frontal_midline_theta_track_key_for_eeg_datasource(eeg_ds)
    if result.get("times_are_absolute_unix"):
        x_abs = np.asarray(result["times"], dtype=float)
        logger.info(
            "%s: using absolute unix times from merge (n=%s, t in [%.3f, %.3f]).",
            track_key,
            x_abs.size,
            float(np.nanmin(x_abs)) if x_abs.size else float("nan"),
            float(np.nanmax(x_abs)) if x_abs.size else float("nan"),
        )
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

    y = np.asarray(result["fmt_power"], dtype=float)

    if x_abs.size and len(getattr(eeg_ds, "intervals_df", [])):
        try:
            t_series_lo, t_series_hi = float(np.nanmin(x_abs)), float(np.nanmax(x_abs))
            for j in range(len(eeg_ds.intervals_df)):
                iv = eeg_ds.intervals_df.iloc[j]
                lo = interval_row_t_start_unix_seconds(iv)
                hi = lo + float(iv["t_duration"])
                overlap = t_series_hi >= lo - 1e-3 and t_series_lo <= hi + 1e-3
                logger.info(
                    "%s: overlap interval[%s] t_start=%.3f t_end=%.3f vs series[%s,%s]: %s",
                    track_key,
                    j,
                    lo,
                    hi,
                    t_series_lo,
                    t_series_hi,
                    overlap,
                )
                if not overlap:
                    logger.warning(
                        "%s: no samples in FMT series overlap interval row %s (detail rects may be grey-only for that segment).",
                        track_key,
                        j,
                    )
                ## END for j in range(len(eeg_ds.intervals_df))....
        except Exception as ex:
            logger.debug("%s: interval vs series overlap check skipped: %s", track_key, ex)

    detailed = pd.DataFrame({"t": x_abs, "fmt_power": y})
    iv_fmt = intervals_df_for_theta_delta_track(eeg_ds, result, log_prefix=track_key)

    if track_key in timeline.track_renderers and hasattr(timeline, "track_is_fully_attached") and timeline.track_is_fully_attached(track_key):
        logger.info("%s: refreshing existing track.", track_key)
        fmt_widget, fmt_track, fmt_ds = timeline.get_track_tuple(track_key)
        fmt_ds.intervals_df = iv_fmt.copy()
        fmt_ds.detailed_df = detailed
        fmt_ds.source_data_changed_signal.emit()
        return (fmt_widget, fmt_track, fmt_ds)

    if hasattr(timeline, "teardown_orphaned_track"):
        timeline.teardown_orphaned_track(track_key)

    logger.info(
        "%s: creating new track (n=%s, x_range=[%s, %s]).",
        track_key,
        len(detailed),
        x_abs.min() if x_abs.size else float("nan"),
        x_abs.max() if x_abs.size else float("nan"),
    )
    fmt_ds = FrontalMidlineThetaTrackDatasource(
        intervals_df=iv_fmt,
        eeg_df=detailed,
        custom_datasource_name=track_key,
        max_points_per_second=64.0,
        enable_downsampling=True,
        channel_names=["fmt_power"],
        normalize=True,
        normalize_over_full_data=True,
        plot_pen_colors=["#d62728"],
        plot_pen_width=1.2,
        lab_obj_dict=getattr(eeg_ds, "lab_obj_dict", None),
        raw_datasets_dict=getattr(eeg_ds, "raw_datasets_dict", None),
    )
    fmt_widget, _root, fmt_plot_item, _fmt_dock = timeline.add_new_embedded_pyqtgraph_render_plot_widget(
        name=fmt_ds.custom_datasource_name,
        dockSize=(500, 60),
        dockAddLocationOpts=["bottom"],
        sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA,
    )

    ref_name = eeg_ds.custom_datasource_name
    if ref_name in timeline.ui.matplotlib_view_widgets:
        ref_plot = timeline.ui.matplotlib_view_widgets[ref_name].getRootPlotItem()
        x0v, x1v = ref_plot.getViewBox().viewRange()[0]
        fmt_plot_item.setXRange(x0v, x1v, padding=0)

    if show_bottom_axis:
        fmt_plot_item.setLabel("bottom", "Time (unix s)")
        fmt_plot_item.showAxis("bottom")
    else:
        fmt_plot_item.hideAxis("bottom")

    if show_title_label:
        fmt_plot_item.setTitle("Frontal-midline theta power (5–9 Hz, NaN = motion/QC excluded)")
    else:
        fmt_plot_item.setTitle(None)

    if show_left_axis:
        fmt_plot_item.setLabel("left", "FMT power (5–9 Hz, norm.)")
        fmt_plot_item.showAxis("left")
    else:
        fmt_plot_item.hideAxis("left")

    timeline.add_track(fmt_ds, name=fmt_ds.custom_datasource_name, plot_item=fmt_plot_item)
    fmt_widget.optionsPanel = fmt_widget.getOptionsPanel()

    fmt_widget, fmt_track, fmt_ds = timeline.get_track_tuple(fmt_ds.custom_datasource_name)
    return (fmt_widget, fmt_track, fmt_ds)


__all__ = [
    "DEFAULT_FMT_BAND",
    "DEFAULT_FMT_H_FREQ",
    "DEFAULT_FMT_L_FREQ",
    "FRONTAL_MIDLINE_THETA_PARAM_KEYS",
    "FrontalMidlineThetaComputation",
    "apply_frontal_midline_theta_to_timeline",
    "compute_frontal_midline_theta_merged_for_intervals",
    "compute_frontal_midline_theta_series",
    "filter_frontal_midline_theta_params",
    "frontal_midline_theta_params_fingerprint",
    "resolve_fmt_channel_picks",
]
