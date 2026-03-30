"""ADHD / sleep-intrusion style theta–delta ratio from EEG with motion masking and optional autoreject.

Motion high-acceleration intervals are merged as MNE annotations (``BAD_motion``) and excluded from
sliding-window spectral analysis. Per-session bad channels use ``EEGComputations.time_independent_bad_channels``
(pyprep). Optional autoreject marks additional epoch spans as invalid for the ratio time series.

Passing ``motion_df`` imports ``MotionData`` (same dependency chain as ``phopymnehelper.motion_data``, including
``pypho_timeline`` where applicable). With ``motion_df=None`` the analysis runs without that import.

For DAG / :class:`~phopymnehelper.analysis.computations.protocol.ComputationNode` use,
:class:`ThetaDeltaSleepIntrusionComputation` (``to_computation_node()`` / ``run_fn()``); direct scripts and
notebooks can keep calling :func:`compute_theta_delta_sleep_intrusion_series`.

Example (synthetic EEG, no motion dataframe):

    import numpy as np
    import mne
    from phopymnehelper.analysis.computations.specific import compute_theta_delta_sleep_intrusion_series

    sfreq, n_ch, n_times = 100.0, 4, 2000
    data = np.random.randn(n_ch, n_times) * 1e-6
    info = mne.create_info([f"EEG{i}" for i in range(n_ch)], sfreq, "eeg")
    raw = mne.io.RawArray(data, info)
    out = compute_theta_delta_sleep_intrusion_series(raw, motion_df=None, window_sec=2.0, step_sec=1.0)
    assert "theta_delta_ratio" in out and len(out["times"]) == len(out["theta_delta_ratio"])
"""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from scipy import signal

from phopymnehelper.EEG_data import EEGComputations
from phopymnehelper.analysis.computations.protocol import ArtifactKind, RunContext
from phopymnehelper.analysis.computations.specific.bad_epochs import fit_autoreject_bad_sample_mask
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase


THETA_DELTA_SLEEP_INTRUSION_PARAM_KEYS: FrozenSet[str] = frozenset(
    {
        "motion_df",
        "total_accel_threshold",
        "minimum_motion_bad_duration",
        "meas_date",
        "l_freq",
        "h_freq",
        "window_sec",
        "step_sec",
        "delta_band",
        "theta_band",
        "use_autoreject",
        "autoreject_epoch_sec",
        "autoreject_kwargs",
        "bad_channel_kwargs",
        "channel_agg",
        "copy_raw",
        "motion_description_substr",
    }
)


def filter_theta_delta_sleep_intrusion_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: params[k] for k in THETA_DELTA_SLEEP_INTRUSION_PARAM_KEYS if k in params}


def theta_delta_sleep_intrusion_params_fingerprint(params: Mapping[str, Any]) -> str:
    f = filter_theta_delta_sleep_intrusion_params(params)
    payload: Dict[str, Any] = {}
    for k, v in sorted(f.items()):
        if k == "motion_df":
            if v is None:
                payload[k] = None
            else:
                try:
                    from pandas.util import hash_pandas_object
                    ho = hash_pandas_object(v)
                    arr = np.ascontiguousarray(np.asarray(ho, dtype=np.uint64))
                    hb = hashlib.sha256(arr.tobytes()).hexdigest()[:24]
                except Exception:
                    hb = f"rows{len(v)}cols{len(v.columns)}"
                payload[k] = {"n": len(v), "cols": list(v.columns), "h": hb}
        else:
            payload[k] = v
    return json.dumps(payload, sort_keys=True, default=str)


def _merge_annotations(raw: mne.io.BaseRaw, extra: mne.Annotations) -> None:
    cur = raw.annotations
    if cur is None or len(cur) == 0:
        raw.set_annotations(extra)
    else:
        raw.set_annotations(cur + extra)


def _annotations_intervals_seconds(annots: Optional[mne.Annotations]) -> List[Tuple[float, float, str]]:
    if annots is None or len(annots) == 0:
        return []
    out: List[Tuple[float, float, str]] = []
    for k in range(len(annots)):
        d = float(annots.duration[k])
        out.append((float(annots.onset[k]), float(annots.onset[k]) + d, str(annots.description[k])))
    return out


def _window_overlaps_intervals(t0: float, t1: float, intervals: List[Tuple[float, float, str]], description_filter: Optional[str]) -> bool:
    for a0, a1, desc in intervals:
        if description_filter is not None and description_filter not in desc:
            continue
        if t0 < a1 and t1 > a0:
            return True
    return False


def _sliding_window_edges(tmax: float, window_sec: float, step_sec: float) -> List[Tuple[float, float, float]]:
    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be positive")
    out: List[Tuple[float, float, float]] = []
    t0 = 0.0
    while t0 + window_sec <= tmax + 1e-9:
        t1 = t0 + window_sec
        center = 0.5 * (t0 + t1)
        out.append((t0, t1, center))
        t0 += step_sec
    return out


def _bandpower_from_psd(psd_1d: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> float:
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return float("nan")
    return float(np.trapz(psd_1d[m], freqs[m]))


def _psd_multitaper_or_welch(sig_2d: np.ndarray, sfreq: float, adaptive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from mne.time_frequency import psd_array_multitaper
        psd, freqs = psd_array_multitaper(sig_2d, sfreq, adaptive=adaptive, verbose="ERROR")
        return psd, freqs
    except Exception:
        n = sig_2d.shape[1]
        nperseg = min(int(sfreq * 2), n) if n >= 8 else n
        if nperseg < 4:
            raise
        freqs, psd = signal.welch(sig_2d, fs=sfreq, nperseg=nperseg, axis=-1, detrend="linear")
        return psd, freqs


def _window_hits_sample_mask(t0: float, t1: float, sfreq: float, n_times: int, sample_mask: np.ndarray) -> bool:
    i0 = max(0, int(np.floor(t0 * sfreq)))
    i1 = min(n_times, int(np.ceil(t1 * sfreq)))
    if i1 <= i0:
        return False
    return bool(np.any(sample_mask[i0:i1]))


def compute_theta_delta_sleep_intrusion_series(raw_eeg: mne.io.BaseRaw, motion_df: Optional[pd.DataFrame] = None, *, total_accel_threshold: float = 0.5, minimum_motion_bad_duration: float = 0.05, meas_date: Any = None, l_freq: float = 1.0, h_freq: Optional[float] = 40.0, window_sec: float = 4.0, step_sec: float = 1.0, delta_band: Tuple[float, float] = (1.0, 4.0), theta_band: Tuple[float, float] = (4.0, 8.0), use_autoreject: bool = False, autoreject_epoch_sec: float = 3.0, autoreject_kwargs: Optional[Mapping[str, Any]] = None, bad_channel_kwargs: Optional[Mapping[str, Any]] = None, channel_agg: str = "mean", copy_raw: bool = True, motion_description_substr: str = "BAD_motion") -> Dict[str, Any]:
    """Compute sliding-window theta/d delta power ratio with motion and QC exclusions.

    ``motion_df`` must use column ``t`` (seconds, aligned with ``raw_eeg.times``). When ``motion_df`` is
    passed, ``meas_date`` defaults to ``raw_eeg.info['meas_date']`` for annotation alignment.

    Parameters
    ----------
    raw_eeg :
        Continuous EEG (``mne.io.Raw`` or compatible).
    motion_df :
        Optional motion detail dataframe (see ``MotionData.find_high_accel_periods``).
    total_accel_threshold, minimum_motion_bad_duration :
        Passed to ``MotionData.find_high_accel_periods`` (as ``total_accel_threshold`` / ``minimum_bad_duration``).
    l_freq, h_freq :
        Bandpass before spectral analysis (MNI highpass / lowpass).
    window_sec, step_sec :
        Sliding analysis window length and step (seconds).
    delta_band, theta_band :
        Frequency limits (Hz) for band power; delta low edge should be ``>= l_freq`` if possible.
    use_autoreject :
        If True, fit autoreject on fixed-length epochs (with ``reject_by_annotation='omit'``) and NaN-out
        windows overlapping autoreject-bad epoch spans. Does not run ICA.
    autoreject_epoch_sec :
        Fixed epoch length for autoreject only.
    autoreject_kwargs :
        Extra keyword arguments for ``autoreject.AutoReject`` (merged over defaults).
    bad_channel_kwargs :
        Passed through to ``EEGComputations.time_independent_bad_channels``.
    channel_agg :
        ``\"mean\"`` or ``\"median\"`` across good EEG channels before PSD.
    copy_raw :
        If True, analyze a copy of ``raw_eeg`` (recommended).
    motion_description_substr :
        Annotations containing this substring are treated as motion bad for window rejection.

    Returns
    -------
    dict
        ``times``, ``theta_delta_ratio``, ``session_mean_theta_delta``, ``n_windows``, ``n_valid_windows``,
        ``bad_channel_result``, ``motion_high_accel_df``, ``params``.
    """
    if channel_agg not in ("mean", "median"):
        raise ValueError("channel_agg must be 'mean' or 'median'")
    if delta_band[0] < l_freq:
        warnings.warn(f"delta_band lower edge {delta_band[0]} is below highpass l_freq={l_freq}; interpret band powers with care", RuntimeWarning, stacklevel=2)

    raw = raw_eeg.copy() if copy_raw else raw_eeg
    raw.load_data()
    nyq = 0.5 * float(raw.info["sfreq"])
    eff_h_freq = h_freq
    if eff_h_freq is not None and eff_h_freq >= nyq:
        eff_h_freq = max(float(l_freq) + 1.0, nyq - 1.0)
        warnings.warn(f"h_freq {h_freq} >= Nyquist ({nyq}); using {eff_h_freq}", RuntimeWarning, stacklevel=2)

    motion_df_out: Optional[pd.DataFrame] = None
    motion_annots: Optional[mne.Annotations] = None
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
        _merge_annotations(raw, motion_annots)

    raw.filter(l_freq=l_freq, h_freq=eff_h_freq, verbose=False)
    bad_channel_result = EEGComputations.time_independent_bad_channels(raw, **dict(bad_channel_kwargs or {}))
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if picks.size == 0:
        warnings.warn("No good EEG picks after bad-channel detection; returning empty series", RuntimeWarning, stacklevel=2)
        empty = np.array([], dtype=float)
        params = dict(total_accel_threshold=total_accel_threshold, minimum_motion_bad_duration=minimum_motion_bad_duration, l_freq=l_freq, h_freq=eff_h_freq, h_freq_requested=h_freq, window_sec=window_sec, step_sec=step_sec, delta_band=delta_band, theta_band=theta_band, use_autoreject=use_autoreject, autoreject_epoch_sec=autoreject_epoch_sec, channel_agg=channel_agg, motion_description_substr=motion_description_substr)
        return dict(times=empty, theta_delta_ratio=empty, session_mean_theta_delta=float("nan"), n_windows=0, n_valid_windows=0, bad_channel_result=bad_channel_result, motion_high_accel_df=motion_df_out, params=params)

    ar_mask: Optional[np.ndarray] = None
    if use_autoreject:
        ar_mask = fit_autoreject_bad_sample_mask(raw, autoreject_epoch_sec=autoreject_epoch_sec, autoreject_kwargs=autoreject_kwargs)

    annot_intervals = _annotations_intervals_seconds(raw.annotations)
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    tmax = float(raw.times[-1])
    edges = _sliding_window_edges(tmax, window_sec, step_sec)

    times_list: List[float] = []
    ratio_list: List[float] = []
    eps = 1e-10

    for t0, t1, tc in edges:
        if _window_overlaps_intervals(t0, t1, annot_intervals, motion_description_substr):
            times_list.append(tc)
            ratio_list.append(float("nan"))
            continue
        if ar_mask is not None and _window_hits_sample_mask(t0, t1, sfreq, n_times, ar_mask):
            times_list.append(tc)
            ratio_list.append(float("nan"))
            continue
        s0 = max(0, int(np.floor(t0 * sfreq)))
        s1 = min(n_times, int(np.ceil(t1 * sfreq)))
        if s1 - s0 < max(8, int(0.5 * window_sec * sfreq)):
            times_list.append(tc)
            ratio_list.append(float("nan"))
            continue
        block = raw.get_data(picks=picks, start=s0, stop=s1)
        if channel_agg == "mean":
            sig1 = np.mean(block, axis=0, keepdims=True)
        else:
            sig1 = np.median(block, axis=0, keepdims=True)
        try:
            psd, freqs = _psd_multitaper_or_welch(sig1, sfreq)
        except Exception:
            times_list.append(tc)
            ratio_list.append(float("nan"))
            continue
        p_psd = psd[0]
        p_delta = _bandpower_from_psd(p_psd, freqs, delta_band)
        p_theta = _bandpower_from_psd(p_psd, freqs, theta_band)
        if not np.isfinite(p_delta) or not np.isfinite(p_theta):
            times_list.append(tc)
            ratio_list.append(float("nan"))
            continue
        times_list.append(tc)
        ratio_list.append(float(p_theta / (p_delta + eps)))
    ## END for t0, t1, tc in edges...

    times_arr = np.asarray(times_list, dtype=float)
    ratio_arr = np.asarray(ratio_list, dtype=float)
    valid = np.isfinite(ratio_arr)
    n_valid = int(np.sum(valid))
    session_mean = float(np.nanmean(ratio_arr)) if n_valid > 0 else float("nan")

    params = dict(total_accel_threshold=total_accel_threshold, minimum_motion_bad_duration=minimum_motion_bad_duration, l_freq=l_freq, h_freq=eff_h_freq, h_freq_requested=h_freq, window_sec=window_sec, step_sec=step_sec, delta_band=tuple(delta_band), theta_band=tuple(theta_band), use_autoreject=use_autoreject, autoreject_epoch_sec=autoreject_epoch_sec, channel_agg=channel_agg, motion_description_substr=motion_description_substr)

    return dict(times=times_arr, theta_delta_ratio=ratio_arr, session_mean_theta_delta=session_mean, n_windows=len(times_arr), n_valid_windows=n_valid, bad_channel_result=bad_channel_result, motion_high_accel_df=motion_df_out, params=params)


class ThetaDeltaSleepIntrusionComputation(SpecificComputationBase):
    computation_id: ClassVar[str] = "theta_delta_sleep_intrusion"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ()
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.stream
    params_fingerprint_fn: ClassVar[Optional[Callable[[Mapping[str, Any]], str]]] = theta_delta_sleep_intrusion_params_fingerprint


    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        if ctx.raw is None:
            raise ValueError("ThetaDeltaSleepIntrusionComputation requires ctx.raw")
        kw = filter_theta_delta_sleep_intrusion_params(params)
        motion_df = kw.pop("motion_df", None)
        return compute_theta_delta_sleep_intrusion_series(ctx.raw, motion_df=motion_df, **kw)


# Updates the Qt timeline after `adhd_ctx["out"]` exists: yellow theta/delta overlay on the EEG track + new docked ANALYSIS_theta_delta track (once).


def apply_adhd_sleep_intrusion_to_timeline(timeline, adhd_ctx):
    """ Usage: apply_adhd_sleep_intrusion_to_timeline(timeline, adhd_ctx) """
    from datetime import datetime

    import pyqtgraph as pg
    from pypho_timeline.core.synchronized_plot_mode import SynchronizedPlotMode
    from pypho_timeline.rendering.datasources.specific.eeg import EEGTrackDatasource
    from pypho_timeline.utils.datetime_helpers import datetime_to_unix_timestamp, float_to_datetime

    out = adhd_ctx["out"]
    if out is None:
        print("Run the compute cell first.")
        return
    t0 = float(adhd_ctx["t0"])
    x_abs = t0 + np.asarray(out["times"], dtype=float)
    y = np.asarray(out["theta_delta_ratio"], dtype=float)
    finite = np.isfinite(y)
    ymax = float(np.nanpercentile(y[finite], 98)) if np.any(finite) else 1.0
    if ymax <= 0:
        ymax = 1.0
    y_overlay = np.clip(y / ymax, 0.0, 1.0)
    ew, _, _ = timeline.get_track_tuple(adhd_ctx["eeg_name"])
    if ew is None:
        print("Missing EEG widget for", adhd_ctx["eeg_name"])
        return
    eeg_pi = ew.getRootPlotItem()
    if (not hasattr(timeline, "_adhd_theta_delta_overlay")) or (timeline._adhd_theta_delta_overlay is None):
        timeline._adhd_theta_delta_overlay = pg.PlotDataItem(x=x_abs, y=y_overlay, pen=pg.mkPen("#ffcc00", width=2))
        eeg_pi.addItem(timeline._adhd_theta_delta_overlay)
    else:
        timeline._adhd_theta_delta_overlay.setData(x=x_abs, y=y_overlay)
    analysis_name = "ANALYSIS_theta_delta"
    if analysis_name in timeline.track_renderers:
        print(analysis_name, "already present; overlay updated only.")
        return
    detailed = pd.DataFrame({"t": x_abs, "theta_delta": y})
    ratio_ds = EEGTrackDatasource(intervals_df=adhd_ctx["eeg_ds"].intervals_df.copy(), eeg_df=detailed, custom_datasource_name=analysis_name, max_points_per_second=64.0, enable_downsampling=True, channel_names=["theta_delta"])
    timeline.TrackRenderingMixin_on_buildUI()
    track_widget, _root, plot_item, _dock = timeline.add_new_embedded_pyqtgraph_render_plot_widget(name=analysis_name, dockSize=(500, 80), dockAddLocationOpts=["bottom"], sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA)
    if isinstance(timeline.total_data_start_time, (datetime, pd.Timestamp)):
        plot_item.setXRange(datetime_to_unix_timestamp(timeline.total_data_start_time), datetime_to_unix_timestamp(timeline.total_data_end_time), padding=0)
    elif timeline.reference_datetime is not None:
        plot_item.setXRange(datetime_to_unix_timestamp(float_to_datetime(timeline.total_data_start_time, timeline.reference_datetime)), datetime_to_unix_timestamp(float_to_datetime(timeline.total_data_end_time, timeline.reference_datetime)), padding=0)
    else:
        plot_item.setXRange(float(timeline.total_data_start_time), float(timeline.total_data_end_time), padding=0)
    plot_item.setLabel("bottom", "Time")
    plot_item.setYRange(0, 1, padding=0.05)
    plot_item.hideAxis("left")
    timeline.add_track(ratio_ds, name=analysis_name, plot_item=plot_item)
    track_widget.optionsPanel = track_widget.getOptionsPanel()



__all__ = [
    "THETA_DELTA_SLEEP_INTRUSION_PARAM_KEYS",
    "ThetaDeltaSleepIntrusionComputation",
    "compute_theta_delta_sleep_intrusion_series",
    "filter_theta_delta_sleep_intrusion_params",
    "theta_delta_sleep_intrusion_params_fingerprint",
]
