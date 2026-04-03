"""Pyprep bad-channel QC plus optional autoreject bad-epoch timing (no motion pipeline).

Results are suitable for caching, DAG nodes (:class:`BadEpochsQCComputation`), timeline
overlays via :func:`apply_bad_epochs_overlays_to_timeline`, and an interval strip track
via :func:`ensure_bad_epochs_interval_track` (same ``time_offset`` as the overlays).
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Tuple
import phopymnehelper.type_aliases as types

import mne
import numpy as np
import pandas as pd

from phopymnehelper.EEG_data import EEGComputations
from phopymnehelper.analysis.computations.protocol import ArtifactKind, RunContext
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase
from phopymnehelper.helpers.dataframe_accessor_helpers import MaskedValidDataFrameAccessor
from phopymnehelper.motion_data import MotionData, MotionDataFrame, BadMotionDataFrame


BAD_EPOCH_INTERVALS_TRACK_DEFAULT_NAME: str = "bad epoch intervals"


BAD_EPOCHS_QC_PARAM_KEYS: FrozenSet[str] = frozenset(
    {
        "l_freq",
        "h_freq",
        "use_autoreject",
        "autoreject_epoch_sec",
        "autoreject_kwargs",
        "bad_channel_kwargs",
        "copy_raw",
    }
)


def filter_bad_epochs_qc_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: params[k] for k in BAD_EPOCHS_QC_PARAM_KEYS if k in params}


def bad_epochs_qc_params_fingerprint(params: Mapping[str, Any]) -> str:
    f = filter_bad_epochs_qc_params(params)
    return json.dumps({k: f[k] for k in sorted(f.keys())}, sort_keys=True, default=str)


def autoreject_bad_sample_mask(raw: mne.io.BaseRaw, epochs: mne.Epochs, reject_log: Any) -> np.ndarray:
    mask = np.zeros(int(raw.n_times), dtype=bool)
    bad = np.asarray(reject_log.bad_epochs, dtype=bool).ravel()
    first = int(raw.first_samp)
    n_ep = len(epochs.events)
    if bad.shape[0] != n_ep:
        warnings.warn("autoreject bad_epochs length mismatch; skipping autoreject time mask", RuntimeWarning, stacklevel=2)
        return mask
    ep_len = len(epochs.times)
    for i in np.where(bad)[0]:
        abs_start = int(epochs.events[i, 0])
        abs_end = abs_start + ep_len
        i0 = max(0, abs_start - first)
        i1 = min(int(raw.n_times), abs_end - first)
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def fit_autoreject_bad_sample_mask(raw: mne.io.BaseRaw, *, autoreject_epoch_sec: float = 3.0, autoreject_kwargs: Optional[Mapping[str, Any]] = None) -> Optional[np.ndarray]:
    try:
        import autoreject
    except ImportError:
        warnings.warn("autoreject not installed; skipping autoreject masking", RuntimeWarning, stacklevel=2)
        return None
    epochs = mne.make_fixed_length_epochs(raw, duration=autoreject_epoch_sec, preload=True, reject_by_annotation="omit")
    if len(epochs) < 2:
        warnings.warn("Too few epochs after reject_by_annotation; skipping autoreject", RuntimeWarning, stacklevel=2)
        return None
    ar_kw = dict(n_interpolate=[1, 2, 3], random_state=1337, n_jobs=1, verbose=False)
    ar_kw.update(dict(autoreject_kwargs or {}))
    ar = autoreject.AutoReject(**ar_kw)
    n_fit = min(20, len(epochs))
    try:
        ar.fit(epochs[:n_fit])
        _epochs_clean, reject_log = ar.transform(epochs, return_log=True)
        del _epochs_clean
        return autoreject_bad_sample_mask(raw, epochs, reject_log)
    except Exception as e:
        warnings.warn(f"autoreject failed ({type(e).__name__}: {e}); continuing without autoreject mask", RuntimeWarning, stacklevel=2)
        return None


def _bad_sample_mask_to_intervals_rel(raw: mne.io.BaseRaw, mask: np.ndarray) -> List[Tuple[float, float]]:
    n = int(mask.size)
    if n == 0 or not np.any(mask):
        return []
    t = raw.times
    sfreq = float(raw.info["sfreq"])
    out: List[Tuple[float, float]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        t0 = float(t[i])
        t1 = float(t[j - 1]) + 1.0 / sfreq
        out.append((t0, t1))
        i = j
    return out





class BadEpochsQCComputation(SpecificComputationBase):
    """ 
    from phopymnehelper.analysis.computations.specific.bad_epochs import BadEpochsQCComputation

    """

    computation_id: ClassVar[str] = "bad_epochs"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ()
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.summary
    params_fingerprint_fn: ClassVar[Optional[Callable[[Mapping[str, Any]], str]]] = bad_epochs_qc_params_fingerprint


    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        if ctx.raw is None:
            raise ValueError("BadEpochsQCComputation requires ctx.raw")
        kw = filter_bad_epochs_qc_params(params)
        t0 = kw.get('t0', None) 
        # if t0 is None:
        #     t0 = ctx.earliest_unix_timestamp

        return self.compute_bad_epochs_qc(ctx.raw, t0=t0, **kw)


    @classmethod
    def compute_bad_epochs_qc(cls, raw_eeg: mne.io.BaseRaw, *, l_freq: float = 1.0, h_freq: Optional[float] = 40.0, use_autoreject: bool = True, autoreject_epoch_sec: float = 3.0, autoreject_kwargs: Optional[Mapping[str, Any]] = None, bad_channel_kwargs: Optional[Mapping[str, Any]] = None, copy_raw: bool = True,
                                t0: Optional[float]=None,
                            ) -> Dict[str, Any]:
        raw = raw_eeg.copy() if copy_raw else raw_eeg
        raw.load_data()
        nyq = 0.5 * float(raw.info["sfreq"])
        eff_h_freq = h_freq
        if eff_h_freq is not None and eff_h_freq >= nyq:
            eff_h_freq = max(float(l_freq) + 1.0, nyq - 1.0)
            warnings.warn(f"h_freq {h_freq} >= Nyquist ({nyq}); using {eff_h_freq}", RuntimeWarning, stacklevel=2)
        raw.filter(l_freq=l_freq, h_freq=eff_h_freq, verbose=False)
        bad_channel_result = EEGComputations.time_independent_bad_channels(raw, **dict(bad_channel_kwargs or {}))
        ar_mask: Optional[np.ndarray] = None
        if use_autoreject:
            ar_mask = fit_autoreject_bad_sample_mask(raw, autoreject_epoch_sec=autoreject_epoch_sec, autoreject_kwargs=autoreject_kwargs)
        bad_epoch_intervals_rel: List[Tuple[float, float]] = [] if ar_mask is None else _bad_sample_mask_to_intervals_rel(raw, ar_mask)
        params = dict(l_freq=l_freq, h_freq=eff_h_freq, h_freq_requested=h_freq, use_autoreject=use_autoreject, autoreject_epoch_sec=autoreject_epoch_sec)
        out: Dict[str, Any] = dict(bad_channel_result=bad_channel_result, bad_epoch_intervals_rel=bad_epoch_intervals_rel, params=params)
        if bad_epoch_intervals_rel is not None:
            
            # # eeg_df: pd.DataFrame = eeg_ds.detailed_df.sort_values("t").reset_index(drop=True)
            # # _t = raw.to_data_frame().sort_values("t").reset_index(drop=True)["t"]
            # _t: float = raw.first_time
            # # # first = int(raw.first_samp)
            # # _t = raw.first_time
            # print(f"type(_t): {type(_t)}, t: {_t}")
            # # if pd.api.types.is_datetime64_any_dtype(_t):
            # #     t0 = float(pd.to_datetime(_t, utc=True, errors="coerce").astype(np.int64).iloc[0] / 1e9)
            # # else:
            # #     t0 = float(pd.to_numeric(_t, errors="coerce").iloc[0])
            # t0 = float(_t)
            # print(f'\tt0: {t0}')
            # ## OUTPUTS: t0
            bad_epoch_intervals_df: pd.DataFrame = pd.DataFrame(bad_epoch_intervals_rel, columns=['start_t_rel', 'end_t_rel'])
            t_col_names: str = ['start_t', 'end_t']
            t_rel_col_names = [f'{a_t_col}_rel' for a_t_col in t_col_names]
            if t0 is not None:
                for a_t_col, a_t_rel_col in zip(t_col_names, t_rel_col_names):
                    bad_epoch_intervals_df[a_t_col] = bad_epoch_intervals_df[a_t_rel_col] + t0

            ## add duration and other optional columns
            if ('end_t' in bad_epoch_intervals_df) and ('start_t' in bad_epoch_intervals_df):
                bad_epoch_intervals_df['duration'] = bad_epoch_intervals_df['end_t'] - bad_epoch_intervals_df['start_t']
            else:
                assert (('end_t_rel' in bad_epoch_intervals_df) and ('start_t_rel' in bad_epoch_intervals_df)), f"bad_epoch_intervals_df: {list(bad_epoch_intervals_df.columns)}"
                bad_epoch_intervals_df['duration'] = bad_epoch_intervals_df['end_t_rel'] - bad_epoch_intervals_df['start_t_rel']

            bad_epoch_intervals_df['label'] = bad_epoch_intervals_df.index.to_numpy().astype(int)

            out['bad_epoch_intervals_df'] = bad_epoch_intervals_df

        if ar_mask is not None:
            out["autoreject_sample_mask"] = ar_mask
        return out



def ensure_bad_epochs_interval_track(timeline, result: Mapping[str, Any], *, time_offset: float = 0.0, track_name: str = BAD_EPOCH_INTERVALS_TRACK_DEFAULT_NAME) -> None:
    """Add or refresh a :class:`~pypho_timeline.rendering.datasources.track_datasource.IntervalProvidingTrackDatasource` strip for bad epochs.

    Uses the same mapping as :func:`apply_bad_epochs_overlays_to_timeline`: each interval endpoint is
    ``time_offset`` plus raw-relative seconds from ``result[\"bad_epoch_intervals_rel\"]``. Pass the same
    ``time_offset`` as for overlays so the strip aligns with EEG/XDF timeline axes (often Unix seconds).
    """
    from datetime import datetime

    import pandas as pd

    from pypho_timeline.core.synchronized_plot_mode import SynchronizedPlotMode
    from pypho_timeline.docking.dock_display_configs import CustomCyclicColorsDockDisplayConfig, NamedColorScheme
    from pypho_timeline.rendering.datasources.stream_to_datasources import default_dock_named_color_scheme_key
    from pypho_timeline.rendering.datasources.track_datasource import IntervalProvidingTrackDatasource
    from pypho_timeline.utils.datetime_helpers import datetime_to_unix_timestamp, float_to_datetime

    if isinstance(result, pd.DataFrame):
        intervals_df = result.copy()
    elif isinstance(result, dict):
        off = float(time_offset)
        rows: List[Dict[str, float]] = []
        for a, b in list(result.get("bad_epoch_intervals_rel") or []):
            x0, x1 = off + float(a), off + float(b)
            if x1 <= x0:
                continue
            rows.append(dict(t_start=x0, t_duration=x1 - x0, t_end=x1))
        intervals_df = pd.DataFrame(rows) if rows else pd.DataFrame({"t_start": pd.Series(dtype=float), "t_duration": pd.Series(dtype=float), "t_end": pd.Series(dtype=float)})
    else:
        raise NotImplementedError(f'not implemented type: type(result): {type(result)},\n\tresult: {result}')


    new_ds = IntervalProvidingTrackDatasource(intervals_df=intervals_df, detailed_df=None, custom_datasource_name=track_name, max_points_per_second=1.0, enable_downsampling=False)

    if track_name not in timeline.track_renderers:
        _scheme_key = default_dock_named_color_scheme_key(track_name)
        display_config = CustomCyclicColorsDockDisplayConfig(named_color_scheme=NamedColorScheme[_scheme_key], showCloseButton=True, showCollapseButton=True, showGroupButton=False, corner_radius="0px")
        track_widget, _root_g, a_plot_item, a_dock = timeline.add_new_embedded_pyqtgraph_render_plot_widget(name=track_name, dockSize=(500, 20), dockAddLocationOpts=["bottom"], display_config=display_config, sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA)
        bottom_label_text = ""
        if isinstance(timeline.total_data_start_time, (datetime, pd.Timestamp)):
            unix_start = datetime_to_unix_timestamp(timeline.total_data_start_time)
            unix_end = datetime_to_unix_timestamp(timeline.total_data_end_time)
            a_plot_item.setXRange(unix_start, unix_end, padding=0)
            a_plot_item.setLabel("bottom", bottom_label_text)
        elif timeline.reference_datetime is not None:
            dt_start = float_to_datetime(timeline.total_data_start_time, timeline.reference_datetime)
            dt_end = float_to_datetime(timeline.total_data_end_time, timeline.reference_datetime)
            unix_start = datetime_to_unix_timestamp(dt_start)
            unix_end = datetime_to_unix_timestamp(dt_end)
            a_plot_item.setXRange(unix_start, unix_end, padding=0)
            a_plot_item.setLabel("bottom", bottom_label_text)
        else:
            a_plot_item.setXRange(timeline.total_data_start_time, timeline.total_data_end_time, padding=0)
            a_plot_item.setLabel("bottom", bottom_label_text, units="s")
        a_plot_item.setYRange(0, 1, padding=0)
        a_plot_item.setLabel("left", track_name)
        a_plot_item.hideAxis("left")
        timeline.add_track(new_ds, name=track_name, plot_item=a_plot_item)
        track_widget.optionsPanel = track_widget.getOptionsPanel()
        if hasattr(a_dock, "updateWidgetsHaveOptionsPanel"):
            a_dock.updateWidgetsHaveOptionsPanel()
        if hasattr(a_dock, "updateTitleBar"):
            a_dock.updateTitleBar()
        getattr(timeline, "_rebuild_timeline_overview_strip", lambda: None)()
        return

    extant_ds = timeline.track_datasources.get(track_name)
    tr = timeline.track_renderers.get(track_name)
    if extant_ds is not None and tr is not None and isinstance(extant_ds, IntervalProvidingTrackDatasource):
        extant_ds.intervals_df = new_ds.intervals_df.copy()
        tr.refresh_overview()


def apply_bad_epochs_overlays_to_timeline(timeline, result: Mapping[str, Any], *, time_offset: float = 0.0, z_value: float = 10.0, add_interval_track: bool = False) -> None:
    import pyqtgraph as pg

    from pypho_timeline.rendering.datasources.specific.eeg import EEGSpectrogramTrackDatasource, EEGTrackDatasource

    intervals = list(result.get("bad_epoch_intervals_rel") or [])
    prev = getattr(timeline, "_bad_epochs_overlay_regions", None)
    if isinstance(prev, dict):
        for track_name, items in prev.items():
            tw, _, _ = timeline.get_track_tuple(track_name)
            if tw is None:
                continue
            pi = tw.getRootPlotItem()
            for it in items:
                try:
                    pi.removeItem(it)
                except (AttributeError, RuntimeError, TypeError):
                    pass
    new_regions: Dict[str, List[Any]] = {}
    brush = pg.mkBrush(0, 0, 0, int(round(255 * 0.9)))
    pen = pg.mkPen(width=0)
    for track_name in timeline.get_all_track_names():
        ds = timeline.track_datasources.get(track_name)
        if ds is None or not isinstance(ds, (EEGTrackDatasource, EEGSpectrogramTrackDatasource)):
            continue
        tw, _, _ = timeline.get_track_tuple(track_name)
        if tw is None:
            continue
        pi = tw.getRootPlotItem()
        items: List[Any] = []
        for a, b in intervals:
            x0, x1 = float(time_offset) + float(a), float(time_offset) + float(b)
            if x1 <= x0:
                continue
            reg = pg.LinearRegionItem(values=(x0, x1), movable=False, brush=brush, pen=pen)
            reg.setZValue(z_value)
            pi.addItem(reg)
            items.append(reg)
        if items:
            new_regions[track_name] = items
    timeline._bad_epochs_overlay_regions = new_regions
    if add_interval_track:
        ensure_bad_epochs_interval_track(timeline, result, time_offset=time_offset)


__all__ = [
    "BAD_EPOCH_INTERVALS_TRACK_DEFAULT_NAME",
    "BAD_EPOCHS_QC_PARAM_KEYS",
    "BadEpochsQCComputation",
    "apply_bad_epochs_overlays_to_timeline",
    "ensure_bad_epochs_interval_track",
    "autoreject_bad_sample_mask",
    "bad_epochs_qc_params_fingerprint",
    "compute_bad_epochs_qc",
    "filter_bad_epochs_qc_params",
    "fit_autoreject_bad_sample_mask",
]
