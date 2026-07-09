"""Frontal-midline theta (FMT) power (5–9 Hz) from EEG with optional motion masking.

Vectorized STFT band-power over frontal-midline channels (Fz preferred; Emotiv AF3/AF4 proxy).
Motion windows are excluded via a sample-level mask derived from ``BAD_motion`` annotations
(optional ``motion_df``). Optional autoreject masking is supported via
:func:`phopymnehelper.analysis.computations.specific.bad_epochs.fit_autoreject_bad_sample_mask`.

Primary spectral marker of prolonged wakefulness (Marzano et al., SLEEP 2013).

Public surfaces:

- :func:`compute_frontal_midline_theta_series` -- one raw → flat dict (``times`` are raw-relative seconds)
- :func:`compute_frontal_midline_theta_merged_for_intervals` -- one row per aligned raw/interval → stitched dict
- :class:`FrontalMidlineThetaComputation` -- DAG node wrapper for ``run_eeg_computations_graph``
- :func:`apply_frontal_midline_theta_to_timeline` -- adds/refreshes the ``{eeg_ds_name}_frontal_midline_theta`` track
"""

from __future__ import annotations

import json
import logging
import time
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
    if "motion_df" in f:
        f = dict(f)
        f["motion_df"] = f["motion_df"] is not None
    return json.dumps({k: f[k] for k in sorted(f.keys())}, sort_keys=True, default=str)


def resolve_fmt_channel_picks(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, List[str]]:
    """Resolve frontal-midline channel indices and names for FMT power."""
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

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if picks.size > 0:
        names = [raw.ch_names[i] for i in picks[:1]]
        return picks[:1], names
    raise ValueError(f"No usable frontal-midline channels in raw (ch_names={list(raw.ch_names)}, bads={list(bads)})")


def _merge_time_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [sorted_iv[0]]
    for a0, a1 in sorted_iv[1:]:
        last_a0, last_a1 = merged[-1]
        if a0 <= last_a1:
            merged[-1] = (last_a0, max(last_a1, a1))
        else:
            merged.append((a0, a1))
    ## END for a0, a1 in sorted_iv[1:]....

    return merged


def _motion_intervals_from_annotations(raw: mne.io.BaseRaw) -> List[Tuple[float, float]]:
    annots = raw.annotations
    motion_intervals: List[Tuple[float, float]] = []
    if annots is None or len(annots) == 0:
        return motion_intervals
    for k in range(len(annots)):
        desc = str(annots.description[k])
        if MOTION_BAD_DESC not in desc:
            continue
        t0_a = float(annots.onset[k])
        motion_intervals.append((t0_a, t0_a + float(annots.duration[k])))
    ## END for k in range(len(annots))....

    return _merge_time_intervals(motion_intervals)


def _build_sample_bad_mask(
    n_times: int,
    sfreq: float,
    motion_intervals: Sequence[Tuple[float, float]],
    ar_mask: Optional[np.ndarray],
) -> np.ndarray:
    sample_mask = np.zeros(n_times, dtype=bool)
    if ar_mask is not None:
        n_ar = min(n_times, int(ar_mask.size))
        sample_mask[:n_ar] |= ar_mask[:n_ar]
    for a0, a1 in motion_intervals:
        i0 = max(0, int(np.floor(a0 * sfreq)))
        i1 = min(n_times, int(np.ceil(a1 * sfreq)))
        if i1 > i0:
            sample_mask[i0:i1] = True
    ## END for a0, a1 in motion_intervals....

    return sample_mask


def _mask_stft_bins_from_sample_mask(
    t_bins: np.ndarray,
    power: np.ndarray,
    *,
    sfreq: float,
    window_sec: float,
    n_times: int,
    sample_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Set STFT bin power to NaN when any sample in the bin's window is bad."""
    if t_bins.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    nperseg = max(8, int(window_sec * sfreq))
    power_out = np.asarray(power, dtype=float).copy()
    for j in range(power_out.size):
        s0 = max(0, int(np.floor(float(t_bins[j]) * sfreq)))
        s1 = min(n_times, s0 + nperseg)
        if s1 > s0 and bool(sample_mask[s0:s1].any()):
            power_out[j] = float("nan")
    ## END for j in range(power_out.size)....

    times = t_bins + window_sec / 2.0
    return times, power_out


def _fmt_series_stats_pl(times: np.ndarray, power: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Aggregate FMT series stats via Polars; return arrays plus n_valid and session mean."""
    import polars as pl

    df = pl.DataFrame({"t": times, "fmt_power": power})
    finite = df.filter(pl.col("fmt_power").is_finite())
    n_valid = int(finite.height)
    if n_valid == 0:
        return times, power, 0, float("nan")
    session_mean = float(finite.select(pl.col("fmt_power").mean()).item())
    return times, power, n_valid, session_mean


def _vectorized_fmt_power_series(
    sig: np.ndarray,
    *,
    sfreq: float,
    window_sec: float,
    step_sec: float,
    fmt_band: Tuple[float, float],
    sample_mask: np.ndarray,
    n_times: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-pass STFT band integration for FMT power (5–9 Hz default)."""
    nperseg = max(8, int(window_sec * sfreq))
    if sig.size < nperseg:
        empty = np.array([], dtype=float)
        return empty, empty

    noverlap = max(0, nperseg - int(step_sec * sfreq))
    freqs, t_bins, sxx = signal.spectrogram(sig, fs=sfreq, nperseg=nperseg, noverlap=noverlap, detrend="linear")
    lo, hi = fmt_band
    band_mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(band_mask):
        power = np.full(t_bins.size, np.nan, dtype=float)
    else:
        power = np.trapz(sxx[band_mask, :], freqs[band_mask], axis=0)

    times = t_bins + window_sec / 2.0
    return _mask_stft_bins_from_sample_mask(
        t_bins,
        power,
        sfreq=sfreq,
        window_sec=window_sec,
        n_times=n_times,
        sample_mask=sample_mask,
    )


def _slice_motion_df_for_interval(
    motion_pl: Any,
    *,
    t_lo: float,
    t_hi: float,
) -> Optional[pd.DataFrame]:
    """Slice a Polars motion table to an interval's unix time range; return pandas for MotionData API."""
    import polars as pl

    if motion_pl is None:
        return None
    sliced = motion_pl.filter(pl.col("t").is_between(float(t_lo), float(t_hi), closed="both"))
    if sliced.is_empty():
        return None
    return sliced.to_pandas()


def compute_frontal_midline_theta_merged_for_intervals(
    raws: Sequence[mne.io.BaseRaw],
    intervals_df: pd.DataFrame,
    *,
    motion_df: Optional[pd.DataFrame] = None,
    **compute_kw: Any,
) -> Dict[str, Any]:
    """Run :func:`compute_frontal_midline_theta_series` per raw, stitch into one absolute-time series."""
    import polars as pl

    n_iv = len(intervals_df)
    n_raw = len(raws)
    n = min(n_raw, n_iv)
    if n_raw != n_iv:
        logger.warning("fmt merge: raw count (%s) != interval count (%s); computing %s aligned segment(s).", n_raw, n_iv, n)
    if n == 0:
        raise ValueError("compute_frontal_midline_theta_merged_for_intervals: no raw/interval pairs (empty inputs).")

    motion_pl = pl.from_pandas(motion_df) if motion_df is not None else None
    merge_kw = dict(compute_kw)
    merge_kw.setdefault("copy_raw", False)

    all_t: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    last_params: Optional[Dict[str, Any]] = None
    motion_df_out_last: Optional[pd.DataFrame] = None

    for i in range(n):
        seg_t0 = time.perf_counter()
        raw = raws[i]
        iv = intervals_df.iloc[i]
        anchor = interval_row_t_start_unix_seconds(iv)
        t_end_iv = anchor + float(iv["t_duration"])
        raw_t0 = float(np.asarray(raw.times)[0]) if len(raw.times) > 0 else 0.0

        segment_motion = _slice_motion_df_for_interval(motion_pl, t_lo=anchor, t_hi=t_end_iv)

        sub = compute_frontal_midline_theta_series(
            raw,
            motion_df=segment_motion,
            meas_date=raw.info.get("meas_date"),
            **merge_kw,
        )
        t_rel = np.asarray(sub["times"], dtype=float)
        y = np.asarray(sub["fmt_power"], dtype=float)
        t_abs = raw_relative_times_to_timeline_unix(t_rel, interval_t_start_unix=anchor, raw_times_first=raw_t0)
        all_t.append(t_abs)
        all_y.append(y)
        last_params = sub.get("params")
        if sub.get("motion_high_accel_df") is not None:
            motion_df_out_last = sub["motion_high_accel_df"]

        elapsed = time.perf_counter() - seg_t0
        if t_abs.size:
            t_min, t_max = float(np.nanmin(t_abs)), float(np.nanmax(t_abs))
            logger.info(
                "fmt segment %s/%s done in %.2fs (n_windows=%s): interval t_start=%.3f t_end~=%.3f | series t [%.3f, %.3f]",
                i + 1,
                n,
                elapsed,
                t_abs.size,
                anchor,
                t_end_iv,
                t_min,
                t_max,
            )
            if t_max < anchor - 1.0 or t_min > t_end_iv + 1.0:
                logger.warning("fmt segment %s: series unix range may not overlap interval bounds (check t_start vs raw).", i + 1)
        else:
            logger.info("fmt segment %s/%s done in %.2fs (n_windows=0).", i + 1, n, elapsed)
        ## END for i in range(n)....

    times_out = np.concatenate(all_t) if len(all_t) > 1 else all_t[0]
    power_out = np.concatenate(all_y) if len(all_y) > 1 else all_y[0]
    _, _, n_valid, session_mean = _fmt_series_stats_pl(times_out, power_out)

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
    """Compute sliding-window frontal-midline theta band power (default 5–9 Hz) via vectorized STFT."""
    if channel_agg not in ("mean", "median"):
        raise ValueError("channel_agg must be 'mean' or 'median'")
    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be positive")

    t_compute0 = time.perf_counter()
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
    raw.filter(l_freq=l_freq, h_freq=eff_h_freq, picks=picks, verbose=False)

    ar_mask: Optional[np.ndarray] = (
        fit_autoreject_bad_sample_mask(raw, autoreject_epoch_sec=autoreject_epoch_sec, autoreject_kwargs=autoreject_kwargs)
        if use_autoreject
        else None
    )

    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    motion_intervals = _motion_intervals_from_annotations(raw)
    sample_mask = _build_sample_bad_mask(n_times, sfreq, motion_intervals, ar_mask)

    block = raw.get_data(picks=picks)
    if channel_agg == "mean":
        sig = np.mean(block, axis=0)
    else:
        sig = np.median(block, axis=0)

    times_arr, power_arr = _vectorized_fmt_power_series(
        sig,
        sfreq=sfreq,
        window_sec=window_sec,
        step_sec=step_sec,
        fmt_band=fmt_band,
        sample_mask=sample_mask,
        n_times=n_times,
    )
    times_arr, power_arr, n_valid, session_mean = _fmt_series_stats_pl(times_arr, power_arr)

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
        compute_engine="stft_spectrogram",
    )

    logger.debug(
        "compute_frontal_midline_theta_series: %.2fs, n_windows=%s, n_valid=%s, channels=%s",
        time.perf_counter() - t_compute0,
        len(times_arr),
        n_valid,
        fmt_channels,
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
    """Add or refresh the per-EEG frontal-midline theta track on ``timeline`` from a compute result."""
    import polars as pl

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

    detailed = pl.DataFrame({"t": x_abs, "fmt_power": y}).to_pandas()
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
