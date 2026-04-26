from datetime import datetime, timedelta, timezone
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any
import logging

from pathlib import Path
import pandas as pd
from attrs import define, field, Factory

import mne
import pyxdf
import numpy as np
from benedict import benedict

from phopylslhelper.general_helpers import unwrap_single_element_listlike_if_needed, readable_dt_str
from phopylslhelper.easy_time_sync import EasyTimeSyncParsingMixin
from phopylslhelper.datetime_helpers import float_to_datetime, datetime_to_unix_timestamp

from phopymnehelper.SavedSessionsProcessor import DataModalityType #TODO: move somewhere common
from phopymnehelper.helpers.dataframe_accessor_helpers import CommonDataFrameAccessorMixin


logger = logging.getLogger(__name__)


modality_channels_dict = {'EEG': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
                        'MOTION': ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'],
                        'GENERIC': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
                        'LOG': ['msg'],
}

modality_sfreq_dict = {'EEG': 128, 'MOTION': 16,
                        'GENERIC': 128, 'LOG': -1,
}


def _resolve_stream_first_last_timestamp_sec(stream: dict, stream_info_dict: dict, stream_name: str, n_samples: int, fs: float) -> None:
    """Fill missing ``first_timestamp`` / ``last_timestamp`` (and ``sample_count`` if absent) into ``stream_info_dict``.

    Resolution order: (1) keys already merged from footer; (2) raw ``footer.info``; (3) min/max of ``time_stamps`` (source
    ``time_stamps``); (4) ``created_at`` plus ``(n_samples - 1) / fs`` if stamps are empty (``created_at_heuristic``).

    Logs a warning when salvaging. Raises ``ValueError`` if nothing can be inferred.
    """
    if stream_info_dict.get('sample_count', None) is None and n_samples > 0:
        stream_info_dict['sample_count'] = float(n_samples)
    need_first = stream_info_dict.get('first_timestamp', None) is None
    need_last = stream_info_dict.get('last_timestamp', None) is None
    if not need_first and not need_last:
        return
    finfo = stream.get('footer', {}).get('info', {})
    ft = unwrap_single_element_listlike_if_needed(finfo.get('first_timestamp'))
    lt = unwrap_single_element_listlike_if_needed(finfo.get('last_timestamp'))
    if need_first and ft is not None:
        stream_info_dict['first_timestamp'] = float(ft)
        need_first = False
    if need_last and lt is not None:
        stream_info_dict['last_timestamp'] = float(lt)
        need_last = False
    if not need_first and not need_last:
        return
    ts = np.asarray(stream.get('time_stamps', []), dtype=float)
    if ts.size >= 1:
        if need_first:
            stream_info_dict['first_timestamp'] = float(np.min(ts))
        if need_last:
            stream_info_dict['last_timestamp'] = float(np.max(ts))
        logger.warning('XDF stream "%s": incomplete or missing stream footer; filled first/last from time_stamps where needed', stream_name)
        return
    if n_samples > 0 and fs > 0 and stream_info_dict.get('created_at', None) is not None:
        c0 = float(stream_info_dict['created_at'])
        c1 = c0 + float(n_samples - 1) / float(fs)
        if need_first:
            stream_info_dict['first_timestamp'] = c0
        if need_last:
            stream_info_dict['last_timestamp'] = c1
        logger.warning('XDF stream "%s": missing footer and empty time_stamps; filled first/last from created_at and stream length', stream_name)
        return
    raise ValueError(f'XDF stream "{stream_name}": cannot resolve first_timestamp/last_timestamp (no footer, no time_stamps, no heuristic)')


def merge_streams_by_name(streams_by_file: List[Tuple[List, Path]]) -> Dict[str, List[Tuple[Dict, Path]]]:
    """Group streams by name across multiple XDF files.

    Args:
        streams_by_file: List of tuples (streams_list, file_path) where streams_list is a list of stream dictionaries
                         from pyxdf and file_path is the Path to the XDF file

    Returns:
        Dictionary mapping stream names to lists of (stream_dict, file_path) tuples
    """
    streams_by_name = {}
    for streams, file_path in streams_by_file:
        for stream in streams:
            stream_name = stream['info']['name'][0]
            if stream_name not in streams_by_name:
                streams_by_name[stream_name] = []
            streams_by_name[stream_name].append((stream, file_path))
    return streams_by_name





# Specific Stream Modality Helpers
def _is_motion_stream(stream_type: str, stream_name: str) -> bool:
    return (stream_type.upper() in ['SIGNAL', 'RAW']) and ('Motion' in stream_name)


def _is_eeg_quality_stream(stream_type: str, stream_name: str) -> bool:
    return (stream_type.upper() in ['RAW']) and (' eQuality' in stream_name)


def _is_eeg_stream(stream_type: str, stream_name: str) -> bool:
    return stream_type.upper() == 'EEG'


def _is_log_stream(stream_type: str, stream_name: str) -> bool:
    return (stream_type.upper() in ['MARKERS']) and (stream_name in ['EventBoard', 'TextLogger'])




def _get_channel_names_for_stream(stream_type: str, stream_name: str, n_columns: int) -> List[str]:
    if _is_motion_stream(stream_type, stream_name):
        return modality_channels_dict['MOTION']
    if _is_eeg_quality_stream(stream_type, stream_name) or _is_eeg_stream(stream_type, stream_name):
        return modality_channels_dict['EEG']
    if _is_log_stream(stream_type, stream_name):
        return modality_channels_dict['LOG']
    return [f'Channel_{i}' for i in range(n_columns)]










@pd.api.extensions.register_dataframe_accessor("xdf_streams")
class XDFDataStreamAccessor(CommonDataFrameAccessorMixin):
    """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals 
    
    from phopymnehelper.xdf_files import XDFDataStreamAccessor, LabRecorderXDF

    xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=active_only_out_eeg_raws)
    
    """

    dt_col_names = ['recording_datetime', 'recording_day_date']
    timestamp_column_names = ['created_at', 'first_timestamp', 'last_timestamp']
    timestamp_dt_column_names = ['created_at_dt', 'first_timestamp_dt', 'last_timestamp_dt']
    timestamp_rel_column_names = ['created_at_rel', 'first_timestamp_rel', 'last_timestamp_rel']

    # _required_column_names = ['start', 'stop', 'label', 'duration']

    # def __init__(self, pandas_obj):      
    #     pandas_obj = self._validate(pandas_obj)
    #     self._obj = pandas_obj

    @classmethod
    def init_from_results(cls, _out_xdf_stream_infos_df: pd.DataFrame, active_only_out_eeg_raws: List, max_num_to_process: Optional[int] = None):
        num_sessions: int = len(active_only_out_eeg_raws)

        # Determine which dataset indices to include based on recency (descending)
        selected_indices: List[int]
        if (max_num_to_process is not None) and (isinstance(max_num_to_process, int)) and (max_num_to_process > 0) and (num_sessions > 0):
            recency_candidates: List[Tuple[int, datetime]] = []
            for an_xdf_dataset_idx in np.arange(num_sessions):
                a_raw = active_only_out_eeg_raws[an_xdf_dataset_idx]
                recency_dt: Optional[datetime] = None

                # 1) Prefer embedded recording timestamp from the raw object
                try:
                    recency_dt = a_raw.info.get('meas_date', None)
                except Exception:
                    recency_dt = None

                # 2) Try dt columns already present on the incoming dataframe (if any)
                if recency_dt is None:
                    try:
                        row = _out_xdf_stream_infos_df.loc[an_xdf_dataset_idx]
                        for col_name in ('created_at_dt', 'first_timestamp_dt', 'last_timestamp_dt'):
                            if (col_name in _out_xdf_stream_infos_df.columns) and pd.notnull(row.get(col_name)):
                                recency_dt = row.get(col_name)
                                break
                    except Exception:
                        pass

                # 3) Fallback to filesystem mtime if we can resolve a path
                if recency_dt is None:
                    try:
                        src_desc = a_raw.info.get('description', None)
                        if isinstance(src_desc, str):
                            src_path = Path(src_desc)
                            if src_path.exists():
                                recency_dt = datetime.fromtimestamp(src_path.stat().st_mtime, tz=timezone.utc)
                    except Exception:
                        pass

                # 4) Final fallback: preserve order (later indices considered more recent)
                if recency_dt is None:
                    # Use a monotonic increasing surrogate based on index to preserve ordering
                    # Newer (higher) indices sort after older ones
                    recency_dt = datetime.fromtimestamp(float(an_xdf_dataset_idx), tz=timezone.utc)

                recency_candidates.append((int(an_xdf_dataset_idx), recency_dt))

            # Sort by recency descending and keep the top N
            recency_candidates.sort(key=lambda t: (t[1] is None, t[1]), reverse=True)
            selected_indices = [idx for idx, _ in recency_candidates[:max_num_to_process]]
        else:
            selected_indices = list(range(num_sessions))

        # Build selected raws in the same order as selected indices
        selected_raws: List = [active_only_out_eeg_raws[i] for i in selected_indices]

        # Work on a subset of the dataframe corresponding to selected indices
        xdf_stream_infos_df: pd.DataFrame = deepcopy(_out_xdf_stream_infos_df)
        try:
            xdf_stream_infos_df = xdf_stream_infos_df.loc[selected_indices]
        except Exception:
            xdf_stream_infos_df = xdf_stream_infos_df.iloc[selected_indices]

        # Initialize/ensure columns exist
        xdf_stream_infos_df['xdf_dataset_idx'] = -1
        xdf_stream_infos_df['recording_datetime'] = datetime.now()
        xdf_stream_infos_df['recording_day_date'] = datetime.now()
                

        # Populate per-selected dataset metadata
        for an_xdf_dataset_idx, a_raw in zip(selected_indices, selected_raws):
            a_meas_date = a_raw.info.get('meas_date')
            a_meas_day_date = a_meas_date.replace(hour=0, minute=0, second=0, microsecond=0)
            xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'recording_datetime'] = a_meas_date
            xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'recording_day_date'] = a_meas_day_date
            xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'xdf_dataset_idx'] = an_xdf_dataset_idx
        # end for an_xdf_dat... 

        xdf_stream_infos_df[cls.dt_col_names] = xdf_stream_infos_df[cls.dt_col_names].convert_dtypes()
        # xdf_stream_infos_df['created_at_rel'] = ((xdf_stream_infos_df['created_at_dt'] - xdf_stream_infos_df['recording_day_date']) / pd.Timedelta(hours=24.0))
        # xdf_stream_infos_df['first_timestamp']
        # xdf_stream_infos_df['duration_sec'] = [pd.Timedelta(seconds=v) for v in (xdf_stream_infos_df['n_samples'].astype(float) * (1.0/xdf_stream_infos_df['fs'].astype(float)))]
        xdf_stream_infos_df['duration_sec'] = [pd.Timedelta(seconds=v) if np.isfinite(v) else pd.NaT for v in (xdf_stream_infos_df['n_samples'].astype(float) * (1.0/xdf_stream_infos_df['fs'].astype(float)))]
        
        for a_ts_col_name, a_ts_dt_col_name, a_ts_rel_col_name in zip(cls.timestamp_column_names, cls.timestamp_dt_column_names, cls.timestamp_rel_column_names):
            try:
                # a_ts_dt_col_name: str = f'{a_ts_col_name}_dt'
                # xdf_stream_infos_df[a_ts_dt_col_name] = xdf_stream_infos_df['recording_datetime'] + [pd.Timestamp(v) for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                # xdf_stream_infos_df[a_ts_dt_col_name] = xdf_stream_infos_df['recording_datetime'] + [pd.Timedelta(seconds=float(v)) for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                # xdf_stream_infos_df[a_ts_dt_col_name] = [pd.Timedelta(seconds=float(v)) for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                # xdf_stream_infos_df[a_ts_dt_col_name] = [pd.Timedelta(seconds=float(v)) if np.isfinite(v) else 0.0 for v in xdf_stream_infos_df[a_ts_col_name].to_numpy()]
                xdf_stream_infos_df[a_ts_dt_col_name] = [pd.Timedelta(seconds=float(v)) if np.isfinite(v) else pd.NaT for v in xdf_stream_infos_df[a_ts_col_name].to_numpy().astype(float)]
                xdf_stream_infos_df[a_ts_rel_col_name] = (xdf_stream_infos_df[a_ts_dt_col_name] / pd.Timedelta(hours=24.0))
                xdf_stream_infos_df[a_ts_dt_col_name] = xdf_stream_infos_df['recording_datetime'] + xdf_stream_infos_df[a_ts_dt_col_name]

            except (ValueError, AttributeError) as e:
                logger.warning('failed to add column "%s" due to error: %s. Skipping col.', a_ts_dt_col_name, e)
                raise
            except Exception as e:
                raise
        ## END for a_ts_col_name, a_ts_dt_col_name, a_ts_...

        ## try to add the updated duration column
        try:
            active_duration_col_name: str = 'duration_sec'
            if active_duration_col_name in xdf_stream_infos_df.columns:
                active_duration_col_name = 'duration_sec_check'
            if ('last_timestamp_dt' in xdf_stream_infos_df.columns) and ('first_timestamp_dt' in xdf_stream_infos_df.columns):            
                xdf_stream_infos_df[active_duration_col_name] = xdf_stream_infos_df['last_timestamp_dt'] - xdf_stream_infos_df['first_timestamp_dt']
                
            assert active_duration_col_name in xdf_stream_infos_df.columns, f"active_duration_col_name: '{active_duration_col_name}' still missing from xdf_stream_infos_df.columns: {list(xdf_stream_infos_df.columns)}"
            xdf_stream_infos_df['duration_rel'] = (xdf_stream_infos_df[active_duration_col_name] / pd.Timedelta(hours=24.0))


        except (ValueError, AttributeError) as e:
            logger.warning('failed to add column "%s" due to error: %s. Skipping col.', a_ts_dt_col_name, e)
            raise
        except Exception as e:
            raise
        
        return xdf_stream_infos_df
    

    # @classmethod
    # def adding_needed_columns(cls, obj):


    #     xdf_stream_infos_df: pd.DataFrame = deepcopy(_out_xdf_stream_infos_df)
    #     xdf_stream_infos_df['recording_datetime'] = datetime.now()
    #     xdf_stream_infos_df['recording_day_date'] = datetime.now()


    #     for an_xdf_dataset_idx in np.arange(num_sessions):
    #         a_raw = active_only_out_eeg_raws[an_xdf_dataset_idx]
    #         a_meas_date = a_raw.info.get('meas_date')
    #         a_meas_day_date = a_meas_date.replace(hour=0, minute=0, second=0, microsecond=0)
    #         xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'recording_datetime'] = a_meas_date
    #         xdf_stream_infos_df.loc[an_xdf_dataset_idx, 'recording_day_date'] = a_meas_day_date
    #         # a_result = results[an_xdf_dataset_idx]
    #         # a_stream_info = deepcopy(xdf_stream_infos_df).loc[an_xdf_dataset_idx]    
    #         # # print(f'i: {i}, a_meas_date: {a_meas_date}, a_stream_info: {a_stream_info}\n\n')
    #         # print(f'i: {an_xdf_dataset_idx}, a_meas_date: {a_meas_date}')
    #         # a_df = a_raw.annotations.to_data_frame(time_format='datetime')
    #         # a_df = a_df[a_df['description'] != 'BAD_motion']
    #         # a_df['xdf_dataset_idx'] = an_xdf_dataset_idx
    #         # flat_annotations.append(a_df)
    #     # end for an_xdf_dat... 
    #     xdf_stream_infos_df[dt_col_names] = xdf_stream_infos_df[dt_col_names].convert_dtypes()
    #     xdf_stream_infos_df



    # @classmethod
    # def _validate(cls, obj):
    #     """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('neuron_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
    #     return obj # important! Must return the modified obj to be assigned (since its columns were altered by renaming


    # @property
    # def extra_data_column_names(self):
    #     """Any additional columns in the dataframe beyond those that exist by default. """
    #     return list(set(self._obj.columns) - set(self._required_column_names))

    # @property
    # def extra_data_dataframe(self) -> pd.DataFrame:
    #     """The subset of the dataframe containing additional information in its columns beyond that what is required. """
    #     return self._obj[self.extra_data_column_names]

    # def as_array(self) -> NDArray:
    #     return self._obj[["start", "stop"]].to_numpy()


    # def adding_or_updating_metadata(self, **metadata_update_kwargs) -> pd.DataFrame:
    #     """ updates the dataframe's `df.attrs` dictionary metadata, building it as a new dict if it doesn't yet exist

    #     Usage:
    #         from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch

    #         maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
    #         maze_epochs_df = maze_epochs_df.epochs.adding_or_updating_metadata(train_test_period='train')
    #         maze_epochs_df

    #     """
    #     ## Add the metadata:
    #     if self._obj.attrs is None:
    #         self._obj.attrs = {} # create a new metadata dict on the dataframe
    #     self._obj.attrs.update(**metadata_update_kwargs)
    #     return self._obj



@define(slots=False, eq=False)
class LabRecorderXDF:
    """ Loads a `.xdf` file saved by LabRecorder which may contain one or more LSL Streams of differing types
    
    from phopymnehelper.xdf_files import XDFDataStreamAccessor, LabRecorderXDF


    """
    lab_recorder_to_mne_to_type_dict = {'EEG':'eeg', 'ACC':'eeg', 'GYRO':'eeg', 'RAW': 'eeg'} # 'RAW' for eeg quality
    stream_name_to_modality_dict = {'Epoc X': DataModalityType.EEG, 'Epoc X Motion':DataModalityType.MOTION, 'Epoc X eQuality': None, 'TextLogger': DataModalityType.PHO_LOG_TO_LSL, 'EventBoard': DataModalityType.PHO_LOG_TO_LSL}


    xdf_file_path: Path = field()
    xdf_streams : List[Dict] = field(default=Factory(list))
    xdf_header : Dict = field(default=Factory(dict))
    skipped_stream_names: List[str] = field(default=Factory(list))

    file_datetime: datetime = field(default=None)
    stream_infos: pd.DataFrame = field(default=None)
    streams_timestamp_dfs: Dict[str, pd.DataFrame] = field(default=None)
    datasets: List[mne.io.Raw] = field(default=None)
    datasets_dict: Dict[DataModalityType, List[mne.io.Raw]] = field(default=None)


    # --------------------------------------------------------------------- #
    #                     EEG grouping / merging helpers                    #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _get_eeg_device_key(raw: mne.io.BaseRaw) -> str:
        """Returns a stable device key for grouping EEG raws from the same source.

        Prefers info['device_info']['stream_info']['source_id'] when available,
        otherwise falls back to a composite of hostname and uid, and finally the
        channel names + sfreq as a last resort.
        """
        info = getattr(raw, "info", {}) or {}
        device_info = info.get("device_info", {}) or {}
        stream_info = device_info.get("stream_info", {}) or {}

        source_id = stream_info.get("source_id", None)
        if source_id is not None:
            return f"source_id:{source_id}"

        hostname = stream_info.get("hostname", None)
        uid = stream_info.get("uid", None)
        if hostname is not None or uid is not None:
            return f"host:{hostname}|uid:{uid}"

        ch_names = tuple(info.get("ch_names", []) or [])
        sfreq = info.get("sfreq", None)
        return f"fallback:{ch_names}|sfreq:{sfreq}"

    @classmethod
    def merge_eeg_streams_by_device(cls, eeg_raws: List[mne.io.BaseRaw], strict_merge: bool = False, debug_print: bool = False) -> Tuple[List[mne.io.BaseRaw], List[Dict[str, Any]]]:
        """Group EEG raws by device identity and merge segments per device.

        `debug_print` is kept for call-site compatibility; merge diagnostics use logging DEBUG on this module.

        Returns:
            merged_eeg_raws: List of merged Raw objects, one per device group.
            merge_meta:      List of dicts with keys:
                             - 'device_key'
                             - 'segment_indices' (original indices from eeg_raws)
                             - 'n_segments'
        """
        _ = debug_print
        if not eeg_raws:
            return [], []

        # Up-convert once for safety/consistency
        from phopymnehelper.MNE_helpers import up_convert_raw_obj

        eeg_raws_uc = [up_convert_raw_obj(r) for r in eeg_raws]

        # 1) Group by device key
        groups: Dict[str, List[int]] = {}
        for idx, raw in enumerate(eeg_raws_uc):
            key = cls._get_eeg_device_key(raw)
            groups.setdefault(key, []).append(idx)

        merged_eeg_raws: List[mne.io.BaseRaw] = []
        merge_meta: List[Dict[str, Any]] = []

        # 2) For each device group, sort by start time and merge compatible segments
        for device_key, indices in groups.items():
            # Sort indices by recording start time for deterministic ordering
            def _segment_sort_key(i: int):
                r = eeg_raws_uc[i]
                # Prefer dedicated helpers if available, else fall back to meas_date
                try:
                    if hasattr(r, "raw_timerange"):
                        start, _ = r.raw_timerange()
                        return (start is None, start)
                except Exception:
                    pass
                meas_date = r.info.get("meas_date", None)
                return (meas_date is None, meas_date)

            sorted_indices = sorted(indices, key=_segment_sort_key)
            candidate_raws = [eeg_raws_uc[i] for i in sorted_indices]

            # Basic compatibility check: channel names/types and sampling rate
            base = candidate_raws[0]
            base_ch_names = tuple(base.info.get("ch_names", []) or [])
            base_ch_types = tuple(base.get_channel_types() or [])
            base_sfreq = base.info.get("sfreq", None)

            compat_raws: List[mne.io.BaseRaw] = [base]
            compat_indices: List[int] = [sorted_indices[0]]

            for raw_idx, raw in zip(sorted_indices[1:], candidate_raws[1:]):
                ch_names = tuple(raw.info.get("ch_names", []) or [])
                ch_types = tuple(raw.get_channel_types() or [])
                sfreq = raw.info.get("sfreq", None)

                is_compatible = (
                    ch_names == base_ch_names
                    and ch_types == base_ch_types
                    and sfreq == base_sfreq
                )

                if not is_compatible:
                    msg = (
                        f"LabRecorderXDF.merge_eeg_streams_by_device: "
                        f"incompatible EEG segment skipped for device '{device_key}': "
                        f"ch_names/ctypes/sfreq mismatch."
                    )
                    logger.warning("%s", msg)
                    if strict_merge:
                        raise ValueError(msg)
                    else:
                        continue

                compat_raws.append(raw)
                compat_indices.append(raw_idx)

            if not compat_raws:
                continue

            if len(compat_raws) == 1:
                merged = compat_raws[0]
            else:
                logger.debug("Merging %s EEG segments for device '%s'", len(compat_raws), device_key)
                # Allow discontinuities; mne will handle time within each segment
                merged = mne.concatenate_raws(compat_raws, preload=True)

            merged_eeg_raws.append(merged)
            merge_meta.append(
                {
                    "device_key": device_key,
                    "segment_indices": compat_indices,
                    "n_segments": len(compat_indices),
                }
            )

        return merged_eeg_raws, merge_meta
    

    @classmethod
    def init_basic_from_lab_recorder_xdf_file(cls, a_xdf_file: Path, skipped_stream_names: List[str]=None, debug_print=False, **kwargs) -> "LabRecorderXDF":
        """
        note: if debug_print == True, it enables verbose mode which produces tons of logs like:

        ....
        2026-04-01 06:14:14,869 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95846292 bytes, length=22, StreamId=3
        2026-04-01 06:14:14,870 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95846316 bytes, length=22, StreamId=2
        2026-04-01 06:14:14,871 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95846340 bytes, length=22, StreamId=6
        2026-04-01 06:14:14,872 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95846364 bytes, length=22, StreamId=4
        2026-04-01 06:14:14,873 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95846388 bytes, length=22, StreamId=5
        2026-04-01 06:14:14,874 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95846412 bytes, length=22, StreamId=1
        2026-04-01 06:14:14,875 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95846439 bytes, length=4171, StreamId=4
        2026-04-01 06:14:14,876 - pyxdf.pyxdf - DEBUG -   reading [14,64]
        2026-04-01 06:14:14,877 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95850615 bytes, length=4171, StreamId=2
        2026-04-01 06:14:14,879 - pyxdf.pyxdf - DEBUG -   reading [14,64]
        2026-04-01 06:14:14,879 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95854791 bytes, length=539, StreamId=3
        2026-04-01 06:14:14,880 - pyxdf.pyxdf - DEBUG -   reading [6,16]
        2026-04-01 06:14:15,136 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95855335 bytes, length=4236, StreamId=4
        2026-04-01 06:14:15,138 - pyxdf.pyxdf - DEBUG -   reading [14,65]
        2026-04-01 06:14:15,139 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95859576 bytes, length=4236, StreamId=2
        2026-04-01 06:14:15,147 - pyxdf.pyxdf - DEBUG -   reading [14,65]
        2026-04-01 06:14:15,148 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95863817 bytes, length=539, StreamId=3
        2026-04-01 06:14:15,149 - pyxdf.pyxdf - DEBUG -   reading [6,16]
        2026-04-01 06:14:15,151 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95864361 bytes, length=4106, StreamId=2
        2026-04-01 06:14:15,152 - pyxdf.pyxdf - DEBUG -   reading [14,63]
        2026-04-01 06:14:15,153 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95868472 bytes, length=4106, StreamId=4
        2026-04-01 06:14:15,154 - pyxdf.pyxdf - DEBUG -   reading [14,63]
        2026-04-01 06:14:15,155 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95872583 bytes, length=539, StreamId=3
        2026-04-01 06:14:15,155 - pyxdf.pyxdf - DEBUG -   reading [6,16]
        2026-04-01 06:14:15,156 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95873127 bytes, length=4301, StreamId=4
        2026-04-01 06:14:15,158 - pyxdf.pyxdf - DEBUG -   reading [14,66]
        2026-04-01 06:14:15,159 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95877433 bytes, length=4301, StreamId=2
        2026-04-01 06:14:15,160 - pyxdf.pyxdf - DEBUG -   reading [14,66]
        2026-04-01 06:14:15,160 - pyxdf.pyxdf - DEBUG -  Read tag: 3 at 95881739 bytes, length=539, StreamId=3
        2026-04-01 06:14:15,161 - pyxdf.pyxdf - DEBUG -   reading [6,16]
        2026-04-01 06:14:15,162 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95882280 bytes, length=22, StreamId=3
        2026-04-01 06:14:15,163 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95882304 bytes, length=22, StreamId=6
        2026-04-01 06:14:15,164 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95882328 bytes, length=22, StreamId=4
        2026-04-01 06:14:15,165 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95882352 bytes, length=22, StreamId=2
        2026-04-01 06:14:15,165 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95882376 bytes, length=22, StreamId=5
        2026-04-01 06:14:15,166 - pyxdf.pyxdf - DEBUG -  Read tag: 4 at 95882400 bytes, length=22, StreamId=1
        2026-04-01 06:14:15,167 - pyxdf.pyxdf - DEBUG -  Read tag: 6 at 95882427 bytes, length=88534, StreamId=5
        2026-04-01 06:14:15,174 - pyxdf.pyxdf - DEBUG -  Read tag: 6 at 95970966 bytes, length=88489, StreamId=2
        2026-04-01 06:14:15,181 - pyxdf.pyxdf - DEBUG -  Read tag: 6 at 96059460 bytes, length=88420, StreamId=4
        2026-04-01 06:14:15,189 - pyxdf.pyxdf - DEBUG -  Read tag: 6 at 96147885 bytes, length=88505, StreamId=3
        2026-04-01 06:14:15,195 - pyxdf.pyxdf - DEBUG -  Read tag: 6 at 96236395 bytes, length=88513, StreamId=6
        2026-04-01 06:14:15,278 - pyxdf.pyxdf - DEBUG -  Read tag: 6 at 96324913 bytes, length=88500, StreamId=1
        2026-04-01 06:14:15,916 - pyxdf.pyxdf - INFO -   performing clock synchronization...
        ...

        Which generally really slow things down.
        """
        # debug_print = kwargs.pop('debug_print', False)
        streams, header = pyxdf.load_xdf(a_xdf_file, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False, verbose=debug_print) ## disabled sync since it wasn't working anyway
        _obj = cls(xdf_file_path=a_xdf_file, xdf_streams=streams, xdf_header=header,
                # skipped_stream_names=kwargs.pop('skipped_stream_names', None),
                skipped_stream_names=skipped_stream_names,
                **kwargs)

        _obj.file_datetime: datetime = datetime.strptime(header['info']['datetime'][0], "%Y-%m-%dT%H:%M:%S%z") # '2025-09-11T17:04:20-0400' -> datetime.datetime(2025, 9, 11, 17, 4, 20, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=72000)))           
        _obj.file_datetime = _obj.file_datetime.astimezone(timezone.utc)
        return _obj


    def perform_process_xdf_streams(self, debug_print: bool=True):
        """ processes the loaded streams without loading the entire data from file.

        Per-stream detail is emitted at logging DEBUG for this module (`phopymnehelper.xdf_files`);
        `debug_print` still controls pyxdf / EasyTimeSyncMixin verbosity only.

        Updates:
            self.stream_infos, self.streams_timestamp_dfs

        """
        stream_infos = []
        streams_timestamp_dfs = {}

        for stream in self.xdf_streams:
            name: str = stream['info']['name'][0]
            a_modality: DataModalityType = self.stream_name_to_modality_dict.get(name, None)
            if a_modality is not None:
                a_modality = a_modality.value

            logger.debug('======== STREAM "%s":', name)

            fs = float(stream['info']['nominal_srate'][0])
            stream_info_dict: Dict = {'name': name, 'fs': fs}

            # sample_count: int = stream['footer']['info']['sample_count'][0]

            if (len(stream['time_series']) == 0):
                logger.warning('skipping empty stream: "%s"', name)
                continue ## skip this stream
            elif (name in self.skipped_stream_names):
                logger.warning('skipping "%s" with name in skipped_stream_names: %s', name, self.skipped_stream_names)
                continue ## skip this stream
            else:
                n_samples, n_channels = np.shape(stream['time_series'])
                stream_info_dict.update(**{'n_samples': n_samples, 'n_channels': n_channels})
                ## stream info keys:
                for a_key in ('type', 'stream_id', 'effective_srate', 'hostname', 'source_id', 'channel_count', 'channel_format', 'type', 'created_at', 'source_id', 'version', 'uid'):
                    a_value = stream['info'].get(a_key, None)
                    a_value = unwrap_single_element_listlike_if_needed(a_value)
                    if a_value is not None:
                        stream_info_dict[a_key] = a_value

                ## stream footer:
                for a_key in ('first_timestamp', 'last_timestamp', 'sample_count'):
                    a_value = stream.get('footer', {}).get('info', {}).get(a_key, None)
                    a_value = unwrap_single_element_listlike_if_needed(a_value)
                    if a_value is not None:
                        stream_info_dict[a_key] = float(a_value)

                _resolve_stream_first_last_timestamp_sec(stream, stream_info_dict, name, n_samples, fs)

                ## Update the timestamp keys to float values, and the create a datetime column by adding them to the `file_datetime`
                timestamp_keys = ('created_at', 'first_timestamp', 'last_timestamp')
                for a_key in timestamp_keys:
                    if stream_info_dict.get(a_key, None) is not None:
                        a_ts_value: float = float(stream_info_dict[a_key]) # ['169993.1081304000']
                        a_ts_value_dt: datetime = self.file_datetime + pd.Timedelta(nanoseconds=a_ts_value)
                        a_dt_key: str = f'{a_key}_dt'
                        stream_info_dict[a_dt_key] = a_ts_value_dt
                        logger.debug('\t%s: %s', a_dt_key, readable_dt_str(a_ts_value_dt))

                ## try to get the special marker timestamp helpers:
                desc_info_dict = dict(stream['info'].get('desc', [{}])[0])
                stream_info_dict = EasyTimeSyncParsingMixin.parse_and_add_lsl_outlet_info_from_desc(desc_info_dict=desc_info_dict, stream_info_dict=stream_info_dict, should_fail_on_missing=False, debug_print=debug_print) ## Returns the updated `stream_info_dict`

                ## Add stream info dict to the stream_infos list:
                stream_infos.append(stream_info_dict)

                ## Process Data:
                stream_first_timestamp = pd.Timedelta(seconds=float(stream_info_dict['first_timestamp']))
                stream_last_timestamp = pd.Timedelta(seconds=float(stream_info_dict['last_timestamp']))

                stream_approx_dur_sec: float = (stream_last_timestamp - stream_first_timestamp).total_seconds()
                logger.debug('\tstream_approx_dur_sec: %s', stream_approx_dur_sec)

                stream_timestamps = np.asarray(stream.get('time_stamps', []), dtype=float).copy()
                stream_clock_times = np.asarray(stream.get('clock_times', []), dtype=float).copy()

                logger.debug('\tstream_timestamps: %s', stream_timestamps.tolist())
                logger.debug('\tstream_clock_times: %s', stream_clock_times.tolist())

                zeroed_stream_timestamps = deepcopy(stream_timestamps)
                zeroed_stream_clock_times = deepcopy(stream_clock_times)

                if len(zeroed_stream_timestamps) > 0:
                    assert stream_info_dict.get('stream_start_lsl_local_offset_seconds', None) is not None
                    # zeroed_stream_timestamps = zeroed_stream_timestamps - zeroed_stream_timestamps[0] ## subtract out the first timestamp
                    zeroed_stream_timestamps = zeroed_stream_timestamps - stream_info_dict['stream_start_lsl_local_offset_seconds']
                if len(zeroed_stream_clock_times) > 0:
                    zeroed_stream_clock_times = zeroed_stream_clock_times - zeroed_stream_clock_times[0] ## subtract out the first timestamp
                
                zeroed_stream_timestamps_dt = np.array([pd.Timedelta(seconds=v) for v in zeroed_stream_timestamps]) ## convert to timedelta (for no reason)
                # stream_datetimes = np.array([stream_info_dict.get('recording_start_datetime', file_datetime) + pd.Timedelta(seconds=v) for v in zeroed_stream_timestamps]) ## List[datetime]
                assert stream_info_dict.get('stream_start_datetime', None) is not None
                stream_datetimes = np.array([stream_info_dict.get('stream_start_datetime', self.file_datetime) + pd.Timedelta(seconds=v) for v in zeroed_stream_timestamps]) ## compatibility

                ## OUTPUTS: stream_datetimes

                ## post-zeroed:
                logger.debug('\tpost-zeroed stream_timestamps: %s', stream_timestamps.tolist())
                logger.debug('\tpost-zeroed stream_clock_times: %s', stream_clock_times.tolist())

                ## STREAM OUTPUTS: stream_timestamps, stream_clock_times, zeroed_stream_timestamps, zeroed_stream_clock_times, zeroed_stream_timestamps_dt, stream_datetimes
                # a_raw_df: pd.DataFrame = pd.DataFrame(dict(onset=zeroed_stream_timestamps, onset_dt=zeroed_stream_timestamps_dt, duration=([0.0] * len(zeroed_stream_timestamps_dt)), description=logger_strings))
                # all_annotations.append(a_raw_df)

                ## UPDATE: `streams_timestamp_dfs`
                streams_timestamp_dfs[name] = pd.DataFrame(dict(stream_timestamps=stream_timestamps,
                    zeroed_stream_timestamps=zeroed_stream_timestamps, zeroed_stream_timestamps_dt=zeroed_stream_timestamps_dt,
                    # stream_clock_times=stream_clock_times,  zeroed_stream_clock_times=zeroed_stream_clock_times,
                    stream_datetimes = stream_datetimes,
                ))

                # ## In lightweight mode, only collect bare stream metadata and skip heavy data processing:
                # if not should_load_full_file_data:
                #     continue


        ## END for stream in streams...

        stream_infos: pd.DataFrame = pd.DataFrame.from_records(stream_infos)

        if ('stream_start_datetime' in stream_infos):
            stream_infos = stream_infos.sort_values('stream_start_datetime', ascending=True, inplace=False)
            earliest_stream_start_datetime: datetime = np.nanmin(stream_infos['stream_start_datetime'].to_numpy()) # Timestamp('2025-10-20 18:28:33-0400', tz='US/Eastern')
            stream_infos['stream_start_datetime_rel_to_earliest'] = (stream_infos['stream_start_datetime'] - earliest_stream_start_datetime) #.dt.total_seconds() #.to_numpy().total_seconds()
        else:
            earliest_stream_start_datetime = None

        if ('stream_start_lsl_local_offset_seconds' in stream_infos.columns) and (earliest_stream_start_datetime is not None):
            # np.nanmin(stream_infos['stream_start_lsl_local_offset_seconds'])
            earliest_stream_start_lsl_local_offset_seconds: float = np.nanmin(stream_infos['stream_start_lsl_local_offset_seconds'])
            stream_infos['earliest_stream_rel_lsl_local_offset_seconds'] = stream_infos['stream_start_lsl_local_offset_seconds'] - earliest_stream_start_lsl_local_offset_seconds

        # - [ ] TODO 2025-10-18 Attempt to appropriately re-zero each stream's `'stream_timestamps'` (seconds since recording start conceptually) to the same zero so they can easily be concatenated). Currently assumes they all started at the same time with no offset (which wouldn't be true if I started the logger after the EEG stream, for example).
        # if should_load_full_file_data and len(streams_timestamp_dfs) > 0:
        if len(streams_timestamp_dfs) > 0:
            ## streams_timestamp_dfs
            ## find earliest stream_timestamp across all streams:
            stream_earliest_timestamp_sec_dict = {k:np.nanmin(df['stream_timestamps']) for k, df in streams_timestamp_dfs.items()}
            absolute_earliest_ts_sec: float = np.nanmin([v for v in stream_earliest_timestamp_sec_dict.values()])

            earliest_stream_zeroed_stream_timestamps_dict = {}
            for k, df in streams_timestamp_dfs.items():
                earliest_stream_zeroed_stream_timestamps_dict[k] = df['stream_timestamps'] - absolute_earliest_ts_sec
            stream_earliest_timestamp_sec_dict = {k:np.nanmin(df['stream_timestamps']) }


        self.stream_infos = stream_infos
        self.streams_timestamp_dfs = streams_timestamp_dfs

        return stream_infos, streams_timestamp_dfs


    def perform_load_xdf_streams(self, should_load_full_file_data: bool=True, debug_print: bool=True):
        """ processes the loaded streams
        Updates:
            self.datasets, self.datasets_dict
        """
        self.datasets = []
        self.datasets_dict = {}
        stream_infos = []
        streams_timestamp_dfs = {}

        # stream_infos, streams_timestamp_dfs = self.perform_process_xdf_streams(debug_print=debug_print)

        # raws = self.datasets
        # raws_dict = self.datasets_dict
        all_annotations_dfs = []
        all_annotations_objs: List[mne.Annotations] = []

        for stream in self.xdf_streams:
            ## try/catch so that if a single stream fails or is mangled/unexpected that whole xdf isn't aborted/discarded:
            try:
                ## trying this tream...
                name: str = stream['info']['name'][0]
                a_modality: DataModalityType = self.stream_name_to_modality_dict.get(name, None)
                if a_modality is not None:
                    a_modality = a_modality.value
                if a_modality not in self.datasets_dict:
                    self.datasets_dict[a_modality] = []

                logger.debug('======== STREAM "%s":', name)

                fs = float(stream['info']['nominal_srate'][0])
                stream_info_dict: Dict = {'name': name, 'fs': fs}

                # sample_count: int = stream['footer']['info']['sample_count'][0]

                if (len(stream['time_series']) == 0):
                    logger.warning('skipping empty stream: "%s"', name)
                    continue ## skip this stream
                elif (name in self.skipped_stream_names):
                    logger.warning('skipping "%s" with name in skipped_stream_names: %s', name, self.skipped_stream_names)
                    continue ## skip this stream
                else:
                    n_samples, n_channels = np.shape(stream['time_series'])
                    stream_info_dict.update(**{'n_samples': n_samples, 'n_channels': n_channels})
                    ## stream info keys:
                    for a_key in ('type', 'stream_id', 'effective_srate', 'hostname', 'source_id', 'channel_count', 'channel_format', 'type', 'created_at', 'source_id', 'version', 'uid'):
                        a_value = stream['info'].get(a_key, None)
                        a_value = unwrap_single_element_listlike_if_needed(a_value)
                        if a_value is not None:
                            stream_info_dict[a_key] = a_value

                    ## stream footer:
                    for a_key in ('first_timestamp', 'last_timestamp', 'sample_count'):
                        a_value = stream.get('footer', {}).get('info', {}).get(a_key, None)
                        a_value = unwrap_single_element_listlike_if_needed(a_value)
                        if a_value is not None:
                            stream_info_dict[a_key] = float(a_value)

                    _resolve_stream_first_last_timestamp_sec(stream, stream_info_dict, name, n_samples, fs)

                    ## Update the timestamp keys to float values, and the create a datetime column by adding them to the `file_datetime`
                    timestamp_keys = ('created_at', 'first_timestamp', 'last_timestamp')
                    for a_key in timestamp_keys:
                        if stream_info_dict.get(a_key, None) is not None:
                            a_ts_value: float = float(stream_info_dict[a_key]) # ['169993.1081304000']
                            a_ts_value_dt: datetime = self.file_datetime + pd.Timedelta(nanoseconds=a_ts_value)
                            a_dt_key: str = f'{a_key}_dt'
                            stream_info_dict[a_dt_key] = a_ts_value_dt
                            logger.debug('\t%s: %s', a_dt_key, readable_dt_str(a_ts_value_dt))

                    ## try to get the special marker timestamp helpers:
                    desc_info_dict = dict(stream['info'].get('desc', [{}])[0])
                    stream_info_dict = EasyTimeSyncParsingMixin.parse_and_add_lsl_outlet_info_from_desc(desc_info_dict=desc_info_dict, stream_info_dict=stream_info_dict, should_fail_on_missing=False, debug_print=debug_print) ## Returns the updated `stream_info_dict`
                    
                    ## Add stream info dict to the stream_infos list:
                    stream_infos.append(stream_info_dict)

                    ## Process Data:
                    stream_first_timestamp = pd.Timedelta(seconds=float(stream_info_dict['first_timestamp']))
                    stream_last_timestamp = pd.Timedelta(seconds=float(stream_info_dict['last_timestamp']))

                    stream_approx_dur_sec: float = (stream_last_timestamp - stream_first_timestamp).total_seconds()
                    logger.debug('\tstream_approx_dur_sec: %s', stream_approx_dur_sec)

                    stream_timestamps = np.asarray(stream.get('time_stamps', []), dtype=float).copy()
                    stream_clock_times = np.asarray(stream.get('clock_times', []), dtype=float).copy()

                    logger.debug('\tstream_timestamps: %s', stream_timestamps.tolist())
                    logger.debug('\tstream_clock_times: %s', stream_clock_times.tolist())

                    zeroed_stream_timestamps = deepcopy(stream_timestamps)
                    zeroed_stream_clock_times = deepcopy(stream_clock_times)

                    if len(zeroed_stream_timestamps) > 0:
                        assert stream_info_dict.get('stream_start_lsl_local_offset_seconds', None) is not None
                        # zeroed_stream_timestamps = zeroed_stream_timestamps - zeroed_stream_timestamps[0] ## subtract out the first timestamp
                        zeroed_stream_timestamps = zeroed_stream_timestamps - stream_info_dict['stream_start_lsl_local_offset_seconds']
                    if len(zeroed_stream_clock_times) > 0:
                        zeroed_stream_clock_times = zeroed_stream_clock_times - zeroed_stream_clock_times[0] ## subtract out the first timestamp

                    zeroed_stream_timestamps_dt = np.array([pd.Timedelta(seconds=v) for v in zeroed_stream_timestamps]) ## convert to timedelta (for no reason)
                    # stream_datetimes = np.array([stream_info_dict.get('recording_start_datetime', file_datetime) + pd.Timedelta(seconds=v) for v in zeroed_stream_timestamps]) ## List[datetime]
                    assert stream_info_dict.get('stream_start_datetime', None) is not None
                    stream_datetimes = np.array([stream_info_dict.get('stream_start_datetime', self.file_datetime) + pd.Timedelta(seconds=v) for v in zeroed_stream_timestamps]) ## compatibility

                    ## OUTPUTS: stream_datetimes

                    ## post-zeroed:
                    logger.debug('\tpost-zeroed stream_timestamps: %s', stream_timestamps.tolist())
                    logger.debug('\tpost-zeroed stream_clock_times: %s', stream_clock_times.tolist())

                    ## STREAM OUTPUTS: stream_timestamps, stream_clock_times, zeroed_stream_timestamps, zeroed_stream_clock_times, zeroed_stream_timestamps_dt, stream_datetimes
                    # a_raw_df: pd.DataFrame = pd.DataFrame(dict(onset=zeroed_stream_timestamps, onset_dt=zeroed_stream_timestamps_dt, duration=([0.0] * len(zeroed_stream_timestamps_dt)), description=logger_strings))
                    # all_annotations.append(a_raw_df)

                    ## UPDATE: `streams_timestamp_dfs`
                    streams_timestamp_dfs[name] = pd.DataFrame(dict(stream_timestamps=stream_timestamps,
                        zeroed_stream_timestamps=zeroed_stream_timestamps, zeroed_stream_timestamps_dt=zeroed_stream_timestamps_dt,
                        # stream_clock_times=stream_clock_times,  zeroed_stream_clock_times=zeroed_stream_clock_times,
                        stream_datetimes = stream_datetimes,
                    ))


                    # ## In lightweight mode, only collect bare stream metadata and skip heavy data processing:
                    # if not should_load_full_file_data:
                    #     continue

                    if (fs == 0):  
                        # irregular event streams
                        ch_names = ['TextLogger_Markers']
                        ch_types = ['misc']
                        logger_strings = [unwrap_single_element_listlike_if_needed(v) for v in stream['time_series']]
                        assert len(stream_timestamps) == len(logger_strings), f"len(stream_timestamps): {len(stream_timestamps)} != len(logger_strings): {len(logger_strings)}"

                        ## check
                        assert ((stream_info_dict['created_at_dt'] - self.file_datetime).total_seconds() < (90.0 * 60.0)) # should be less than 10 seconds between the file start and the logging stream (usually...)

                        # a_raw_df: pd.DataFrame = pd.DataFrame(dict(onset=zeroed_stream_timestamps, onset_dt=zeroed_stream_timestamps_dt, converted_dt=converted_dt, duration=([0.0] * len(zeroed_stream_timestamps_dt)), description=logger_strings))
                        a_raw_df: pd.DataFrame = pd.DataFrame(dict(onset=stream_datetimes, duration=([0.0] * len(zeroed_stream_timestamps_dt)), description=logger_strings))
                        all_annotations_dfs.append(a_raw_df)

                        #TODO 2026-03-02 18:00: - [ ] See if the local import of MNE fixes this tz=UTC != tz=UTC issue:
                        from datetime import timezone

                        # If it might be zoneinfo/pytz/other UTC:
                        orig_time = stream_info_dict['stream_start_datetime']
                        if orig_time is not None and orig_time.tzinfo is not None:
                            orig_time = orig_time.astimezone(timezone.utc)
                        # raw = mne.Annotations(..., orig_time=orig_time)

                        ## In lightweight mode, only collect bare stream metadata and skip heavy data processing:
                        raw = mne.Annotations(onset=zeroed_stream_timestamps, duration=([0.0] * len(zeroed_stream_timestamps)), description=logger_strings, orig_time=orig_time) ## set orig_time=None # #TODO 2026-03-02 17:33: - [ ] this is raising and aborting the whole function (just one stream, the 'TextLogger_Markers' stream
                        ## #TODO 2026-03-02 17:44: - [ ] stream_info_dict['stream_start_datetime']: `datetime.datetime(2026, 3, 1, 2, 9, 18, tzinfo=<UTC>)` is the problem, it's raising "ValueError: Date must be datetime object in UTC: datetime.datetime(2026, 3, 1, 2, 9, 18, tzinfo=<UTC>)"
                        ## UPDATE `raws` and `raws_dict` with the new raw object:
                        self.datasets.append(raw)
                        all_annotations_objs.append(raw)

                        if a_modality is not None:
                            self.datasets_dict[a_modality].append(raw)

                    else:
                        ## fixed sampling rate streams:
                        _channels_dict = benedict(stream['info']['desc'][0]['channels'][0])
                        channels_df: pd.DataFrame = pd.DataFrame.from_records([{k:v[0] for k, v in ch_v.items()} for ch_v in _channels_dict.flatten()['channel']])
                        data = np.array(stream['time_series']).T
                        if (stream_info_dict['type'] == 'EEG'):
                            pass
                        # ch_names = [f"{name}_{i}" for i in range(data.shape[0])]
                        # ch_types = ["eeg"] * data.shape[0]  # adjust depending on stream type
                        ch_names = channels_df['label'].to_list()
                        ch_types = [self.lab_recorder_to_mne_to_type_dict[v] for v in channels_df['type']]
                        
                        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
                        info = info.set_meas_date(self.file_datetime)
                        info['description'] = self.xdf_file_path.as_posix()
                        info['device_info'] = {'type':'USB', 'model':'EpocX', 'serial': '', 'site':'pho', 'stream_info': {}} # #TODO 2025-09-22 08:51: - [ ] Add Hostname<USB> or Hostname<BLE>
                        # info['temp']
                        ## add in the 'stream_info' properties:
                        info['device_info']['stream_info'] = {}
                        for k, v in stream_info_dict.items():
                            info['device_info']['stream_info'][k] = deepcopy(v)

                        raw = mne.io.RawArray(data, info) ## also have , first_samp=0

                        ## UPDATE `raws` and `raws_dict` with the new raw object:
                        self.datasets.append(raw)
                        if a_modality is not None:
                            self.datasets_dict[a_modality].append(raw)

            except Exception as e:
                logger.error('stream: %s failed with error: %s. Continuing with the remainder of xdf_streams...', stream, e)
                continue

            # raise e

        ## END for stream in self.xdf_streams...

        stream_infos: pd.DataFrame = pd.DataFrame.from_records(stream_infos)
        if len(stream_infos) == 0:
            raise ValueError(f'stream_infos is empty after loading the XDF! No streams found!')


        if ('stream_start_datetime' in stream_infos):
            stream_infos = stream_infos.sort_values('stream_start_datetime', ascending=True, inplace=False, na_position='last')
            earliest_stream_start_datetime: datetime = stream_infos['stream_start_datetime'].dropna().min() # Timestamp('2025-10-20 18:28:33-0400', tz='US/Eastern') # TODO#TODO 2026-03-29 05:41: - [ ] 'VideoRecorderMarkers' added by `continuous_video_recorder` is missing: ['stream_start_lsl_local_offset_seconds', 'stream_start_datetime', ...]
            stream_infos['stream_start_datetime_rel_to_earliest'] = (stream_infos['stream_start_datetime'] - earliest_stream_start_datetime) #.dt.total_seconds() #.to_numpy().total_seconds()
        else:
            earliest_stream_start_datetime = None
            assert (not should_load_full_file_data), f"we need this unless in `(should_load_full_file_data==False)` mode."

        if ('stream_start_lsl_local_offset_seconds' in stream_infos.columns) and (earliest_stream_start_datetime is not None):
            # np.nanmin(stream_infos['stream_start_lsl_local_offset_seconds'])
            earliest_stream_start_lsl_local_offset_seconds: float = np.nanmin(stream_infos['stream_start_lsl_local_offset_seconds'])
            stream_infos['earliest_stream_rel_lsl_local_offset_seconds'] = stream_infos['stream_start_lsl_local_offset_seconds'] - earliest_stream_start_lsl_local_offset_seconds

        # - [ ] TODO 2025-10-18 Attempt to appropriately re-zero each stream's `'stream_timestamps'` (seconds since recording start conceptually) to the same zero so they can easily be concatenated). Currently assumes they all started at the same time with no offset (which wouldn't be true if I started the logger after the EEG stream, for example).
        if should_load_full_file_data and len(streams_timestamp_dfs) > 0:
            ## streams_timestamp_dfs
            ## find earliest stream_timestamp across all streams:
            stream_earliest_timestamp_sec_dict = {k:np.nanmin(df['stream_timestamps']) for k, df in streams_timestamp_dfs.items()}
            absolute_earliest_ts_sec: float = np.nanmin([v for v in stream_earliest_timestamp_sec_dict.values()])

            earliest_stream_zeroed_stream_timestamps_dict = {}
            for k, df in streams_timestamp_dfs.items():
                earliest_stream_zeroed_stream_timestamps_dict[k] = df['stream_timestamps'] - absolute_earliest_ts_sec
            stream_earliest_timestamp_sec_dict = {k:np.nanmin(df['stream_timestamps']) }

        self.stream_infos = stream_infos
        self.streams_timestamp_dfs = streams_timestamp_dfs
        return self.stream_infos, self.streams_timestamp_dfs, self.datasets, self.datasets_dict


    @classmethod
    def init_from_lab_recorder_xdf_file(cls, a_xdf_file: Path, should_load_full_file_data: bool=True, debug_print: bool=False, skipped_stream_names: Optional[List[str]]=None):
        """

            Conclusions: `stream_clock_times` is not really needed if auto-sync is working.


            =========================================
            With `synchronize_clocks=True`:
                trying to process XDF file 0/1: "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-18T192330.926Z_eeg.xdf"...
                file_datetime: 2025-10-18 03:23:30 PM
                ======== STREAM "TextLogger":
                    created_at_dt: 2025-10-18 03:23:30 PM
                    first_timestamp_dt: 2025-10-18 03:23:30 PM
                    last_timestamp_dt: 2025-10-18 03:23:30 PM
                    FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_lsl_local_offset_seconds": 309833.9379807
                    FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_datetime": 2025-10-18 15:18:52-04:56
                    stream_approx_dur_sec: 19.940502
                    stream_timestamps: [310118.9797208478, 310132.17171209387, 310138.9202364168]
                    stream_clock_times: [310117.99792570004, 310122.99849055, 310127.99873414997, 310132.99948935, 310137.99964735, 310143.0005398, 310148.00089784997, 310153.00175355]
                    post-zeroed stream_timestamps: [0.0, 13.191991246072575, 19.940515569003765]
                    post-zeroed stream_clock_times: [0.0, 5.000564849935472, 10.000808449927717, 15.001563649973832, 20.00172164995456, 25.00261409993982, 30.00297214993043, 35.0038278499851]
                ======== STREAM "EventBoard":
                    created_at_dt: 2025-10-18 03:23:30 PM
                    first_timestamp_dt: 2025-10-18 03:23:30 PM
                    last_timestamp_dt: 2025-10-18 03:23:30 PM
                    FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_lsl_local_offset_seconds": 309833.9379807
                    FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_datetime": 2025-10-18 15:18:52-04:56
                    stream_approx_dur_sec: 0.0
                    stream_timestamps: [310141.53154871357]
                    stream_clock_times: [310117.99793914997, 310122.99851445, 310127.99872775003, 310132.99950185, 310137.99965555, 310143.00056144997, 310148.00089795, 310153.0017462]
                    post-zeroed stream_timestamps: [0.0]
                    post-zeroed stream_clock_times: [0.0, 5.0005753000150435, 10.000788600067608, 15.001562700024806, 20.001716400031, 25.00262230000226, 30.002958800061606, 35.00380705005955]
                ======== STREAM "Epoc X Motion":
                    created_at_dt: 2025-10-18 03:23:30 PM
                    first_timestamp_dt: 2025-10-18 03:23:30 PM
                    last_timestamp_dt: 2025-10-18 03:23:30 PM
                    stream_approx_dur_sec: 39.980819
                    stream_timestamps: [310112.3528809373, 310112.38261473324, 310112.4146451288, 310112.44568922446, 310112.4766869202,..., ]
                    stream_clock_times: [204.484430350014, 209.48498810001183, 214.4852262000204, 219.48597295000218, 224.4861887000734, 229.48706944996957, 234.48741724999854, 239.4882807499962]
                    post-zeroed stream_timestamps: [0.0, 0.02973379596369341, 0.061764191545080394, 0.09280828718328848, 0.12380598293384537, 0.15590557851828635, 0.1929814734030515, 0.21782896999502555, 0.2487867656745948, ..., ]
                    post-zeroed stream_clock_times: [0.0, 5.000557749997824, 10.000795850006398, 15.00154259998817, 20.00175835005939, 25.00263909995556, 30.002986899984535, 35.003850399982184]
                ======== STREAM "Epoc X":
                    created_at_dt: 2025-10-18 03:23:30 PM
                    first_timestamp_dt: 2025-10-18 03:23:30 PM
                    last_timestamp_dt: 2025-10-18 03:23:30 PM
                    stream_approx_dur_sec: 39.989998
                    stream_timestamps: [310112.34278103936, 310112.3521599422, 310112.3596535444, 310112.36563644616, 310112.3766195494, 310112.3816435509, 310112.3937489544, 310112.3987795559, 310112.4097387591, 310112.4137093603, ..., ]
                    stream_clock_times: [204.4844580499921, 209.4850055000279, 214.48526310001034, 219.48599920002744, 224.48616700002458, 229.48707915004343, 234.48742110002786, 239.4882879499928]
                    post-zeroed stream_timestamps: [0.0, 0.009378902846947312, 0.01687250501709059, 0.022855406801681966, 0.03383851004764438, 0.03886251151561737, 0.05096791504183784, 0.05599851656006649, 0.06695771974045783, ..., ]
                    post-zeroed stream_clock_times: [0.0, 5.0005474500358105, 10.000805050018243, 15.001541150035337, 20.00170895003248, 25.00262110005133, 30.00296305003576, 35.00382990000071]

                =========================================
                With `synchronize_clocks=False`:
                    limiting to included_xdf_file_names: ['LabRecorder_Apogee_2025-10-18T192330.926Z_eeg.xdf']...
                    limited to 1/49 files
                    trying to process XDF file 0/1: "E:/Dropbox (Personal)/Databases/UnparsedData/LabRecorderStudies/sub-P001/LabRecorder_Apogee_2025-10-18T192330.926Z_eeg.xdf"...
                    file_datetime: 2025-10-18 03:23:30 PM
                    ======== STREAM "TextLogger":
                        created_at_dt: 2025-10-18 03:23:30 PM
                        first_timestamp_dt: 2025-10-18 03:23:30 PM
                        last_timestamp_dt: 2025-10-18 03:23:30 PM
                        FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_lsl_local_offset_seconds": 309833.9379807
                        FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_datetime": 2025-10-18 15:18:52-04:56
                        stream_approx_dur_sec: 19.940502
                        stream_timestamps: [310118.9797418, 310132.1717244, 310138.9202443]
                        stream_clock_times: [310117.99792570004, 310122.99849055, 310127.99873414997, 310132.99948935, 310137.99964735, 310143.0005398, 310148.00089784997, 310153.00175355]
                        post-zeroed stream_timestamps: [0.0, 13.191982600023039, 19.940502499986906]
                        post-zeroed stream_clock_times: [0.0, 5.000564849935472, 10.000808449927717, 15.001563649973832, 20.00172164995456, 25.00261409993982, 30.00297214993043, 35.0038278499851]
                    ======== STREAM "EventBoard":
                        created_at_dt: 2025-10-18 03:23:30 PM
                        first_timestamp_dt: 2025-10-18 03:23:30 PM
                        last_timestamp_dt: 2025-10-18 03:23:30 PM
                        FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_lsl_local_offset_seconds": 309833.9379807
                        FOUND CUSTOM TIMESTAMP SYNC KEY: "recording_start_datetime": 2025-10-18 15:18:52-04:56
                        stream_approx_dur_sec: 0.0
                        stream_timestamps: [310141.5315568]
                        stream_clock_times: [310117.99793914997, 310122.99851445, 310127.99872775003, 310132.99950185, 310137.99965555, 310143.00056144997, 310148.00089795, 310153.0017462]
                        post-zeroed stream_timestamps: [0.0]
                        post-zeroed stream_clock_times: [0.0, 5.0005753000150435, 10.000788600067608, 15.001562700024806, 20.001716400031, 25.00262230000226, 30.002958800061606, 35.00380705005955]
                    ======== STREAM "Epoc X Motion":
                        created_at_dt: 2025-10-18 03:23:30 PM
                        first_timestamp_dt: 2025-10-18 03:23:30 PM
                        last_timestamp_dt: 2025-10-18 03:23:30 PM
                        stream_approx_dur_sec: 39.980819
                        stream_timestamps: [198.8393982, 198.869132, 198.9011624, 198.9322065, 198.9632042, 198.9953038, 199.0323797, 199.0572272, 199.088185, 199.1191771, 199.1532168, 199.1831693, 199.2131749, 199.250184, 199.2752947, ..., ]
                        stream_clock_times: [204.484430350014, 209.48498810001183, 214.4852262000204, 219.48597295000218, 224.4861887000734, 229.48706944996957, 234.48741724999854, 239.4882807499962]
                        post-zeroed stream_timestamps: [0.0, 0.029733800000002475, 0.06176419999999894, 0.09280830000000151, 0.12380600000000186, 0.15590559999998277, 0.19298150000000192, 0.21782899999999472, 0.24878680000000486, 0.2797788999999966, ..., ]
                        post-zeroed stream_clock_times: [0.0, 5.000557749997824, 10.000795850006398, 15.00154259998817, 20.00175835005939, 25.00263909995556, 30.002986899984535, 35.003850399982184]
                    ======== STREAM "Epoc X":
                        created_at_dt: 2025-10-18 03:23:30 PM
                        first_timestamp_dt: 2025-10-18 03:23:30 PM
                        last_timestamp_dt: 2025-10-18 03:23:30 PM
                        stream_approx_dur_sec: 39.989998
                        stream_timestamps: [198.829322, 198.8387009, 198.8461945, 198.8521774, 198.8631605, 198.8681845, 198.8802899, 198.8853205, 198.8962797, 198.9002503, 198.9123016, 198.9162398, 198.922399, 198.9312574, 198.9392109, 198.9472146, 198.9552784, ..., ]
                        stream_clock_times: [204.4844580499921, 209.4850055000279, 214.48526310001034, 219.48599920002744, 224.48616700002458, 229.48707915004343, 234.48742110002786, 239.4882879499928]
                        post-zeroed stream_timestamps: [0.0, 0.009378900000001522, 0.016872500000005175, 0.022855399999997417, 0.03383850000000166, 0.03886250000002178, 0.05096790000001761, 0.05599850000001538, 0.06695770000001744, 0.07092830000001982, 0.08297960000001581, ..., ]
                        post-zeroed stream_clock_times: [0.0, 5.0005474500358105, 10.000805050018243, 15.001541150035337, 20.00170895003248, 25.00262110005133, 30.00296305003576, 35.00382990000071]
                    n_unique_xdf_datasets: 1
        """
        if skipped_stream_names is None:
            # skipped_stream_names: List[str] = [
            #     # 'TextLogger',
            #     'EventBoard',
            # ]

            skipped_stream_names: List[str] = [
                # 'TextLogger',
                # 'EventBoard',
            ]

        # Load .xdf
        _obj: "LabRecorderXDF" = cls.init_basic_from_lab_recorder_xdf_file(a_xdf_file=a_xdf_file, skipped_stream_names=skipped_stream_names, debug_print=debug_print)
        file_datetime = _obj.file_datetime
        if not should_load_full_file_data:
            stream_infos, streams_timestamp_dfs = _obj.perform_process_xdf_streams(debug_print=debug_print)
            stream_infos = _obj.stream_infos
            raws = _obj.datasets
            raws_dict = _obj.datasets_dict
        else:
            stream_infos, streams_timestamp_dfs, datasets, datasets_dict = _obj.perform_load_xdf_streams(debug_print=debug_print) # #TODO 2026-03-02 17:19: - [ ] ValueError: Date must be datetime object in UTC: datetime.datetime(2026, 3, 1, 2, 9, 18, tzinfo=<UTC>)
            raws = datasets
            raws_dict = datasets_dict

        return _obj


    # @classmethod
    # def up_convert_and_process_raw(cls, eeg_raw):
    #     """ slow but inline 

    #     eeg_raw = LabRecorderXDF.up_convert_and_process_raw(eeg_raw=eeg_raw)

    #     """
    #     from phopymnehelper.MNE_helpers import up_convert_raw_objects, up_convert_raw_obj
    #     from phopymnehelper.EEG_data import EEGData
    #     eeg_raw = up_convert_raw_obj(eeg_raw)
    #     EEGData.set_montage(datasets_EEG=[eeg_raw])
    #     return eeg_raw


    @classmethod
    def load_and_process_all(cls, lab_recorder_output_path: Path, 
                                  labRecorder_PostProcessed_path: Optional[Path] = Path("E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/LabRecorder_PostProcessed").resolve(),
                                    should_write_final_merged_eeg_fif: bool = True,
                                    should_load_full_file_data: bool = False,
                                    debug_print: bool = False,
                                    included_xdf_file_names=None,
                                    fail_on_exception: bool=False,
                                                          ):

        """ main load function for all XDF files exported by LabRecorder

        if `not should_load_full_file_data`, only return the `_out_xdf_stream_infos_df, lab_recorder_xdf_files` and not `_out_eeg_raw` (which will be None)

        """
        from phopymnehelper.MNE_helpers import up_convert_raw_objects, up_convert_raw_obj
        from phopymnehelper.EEG_data import EEGData
                                       
        assert lab_recorder_output_path.exists()

        lab_recorder_xdf_files: List[Path] = list(lab_recorder_output_path.glob('*.xdf'))
        n_total_found_files: int = len(lab_recorder_xdf_files)
        if included_xdf_file_names is not None:
            logger.info('limiting to included_xdf_file_names: %s...', included_xdf_file_names)
            lab_recorder_xdf_files = [v for v in lab_recorder_xdf_files if v.name in included_xdf_file_names]
            n_filtered_found_files: int = len(lab_recorder_xdf_files)
            logger.info('\tlimited to %s/%s files', n_filtered_found_files, n_total_found_files)

        if not should_load_full_file_data:
            assert (not should_write_final_merged_eeg_fif)

        if (labRecorder_PostProcessed_path is not None) and should_write_final_merged_eeg_fif:
            labRecorder_PostProcessed_path.mkdir(exist_ok=True)
        
        # a_xdf_file = lab_recorder_xdf_files[-3]
        # a_xdf_file = lab_recorder_xdf_files[-1]
        # a_xdf_file = Path(r"E:\Dropbox (Personal)\Databases\UnparsedData\LabRecorderStudies\sub-P001\LabRecorder_2025-09-18T031842.989Z_eeg.xdf").resolve()
        # a_xdf_file = Path(r"E:\Dropbox (Personal)\Databases\UnparsedData\LabRecorderStudies\sub-P001\LabRecorder_2025-09-18T121337.267Z_eeg.xdf").resolve()

        _out_eeg_raw = []
        _out_xdf_stream_infos_df = []

        for an_xdf_file_idx, a_xdf_file in enumerate(lab_recorder_xdf_files):
            logger.info('trying to process XDF file %s/%s: "%s"...', an_xdf_file_idx, len(lab_recorder_xdf_files), a_xdf_file.as_posix())
            try:
                _obj = cls.init_from_lab_recorder_xdf_file(a_xdf_file=a_xdf_file, should_load_full_file_data=should_load_full_file_data, debug_print=debug_print)
                stream_infos = _obj.stream_infos
                raws = _obj.datasets
                raws_dict = _obj.datasets_dict


                eeg_raws = raws_dict.get(DataModalityType.EEG.value, [])
                if len(eeg_raws) == 0:
                    logger.warning('no EEG streams found in "%s". Skipping file.', a_xdf_file.as_posix())
                    continue

                # Merge by device so we can handle multiple EEG streams per XDF
                merged_eeg_raws, merge_meta = cls.merge_eeg_streams_by_device(
                    eeg_raws=eeg_raws, strict_merge=False, debug_print=False
                )
                if len(merged_eeg_raws) == 0:
                    logger.warning('could not produce any merged EEG datasets for "%s". Skipping file.', a_xdf_file.as_posix())
                    continue

                # Optionally write FIF/MAT once per merged dataset
                exports_dict = None
                if should_write_final_merged_eeg_fif and labRecorder_PostProcessed_path is not None:
                    _, exports_dict = cls.save_post_processed_to_fif(
                        raws_dict=raws_dict,
                        a_xdf_file=a_xdf_file,
                        labRecorder_PostProcessed_path=labRecorder_PostProcessed_path,
                    )

                for local_idx, eeg_raw in enumerate(merged_eeg_raws):
                    this_stream_infos = deepcopy(stream_infos)
                    this_stream_infos['lab_recorder_xdf_file_idx'] = an_xdf_file_idx
                    this_stream_infos['xdf_filename'] = a_xdf_file.name
                    this_stream_infos['eeg_device_group_idx'] = local_idx
                    this_stream_infos['eeg_device_key'] = merge_meta[local_idx].get('device_key', f'device_{local_idx}')
                    this_stream_infos['n_eeg_segments_in_group'] = merge_meta[local_idx].get('n_segments', 1)

                    if exports_dict is not None:
                        for a_format, per_idx_dict in exports_dict.items():
                            export_path = per_idx_dict.get(local_idx, None)
                            if export_path is not None:
                                this_stream_infos[f'proccessed_{a_format}_filename'] = Path(export_path).name

                    this_stream_infos['xdf_dataset_idx'] = len(_out_xdf_stream_infos_df)

                    if should_load_full_file_data:
                        eeg_raw = up_convert_raw_obj(eeg_raw)
                        EEGData.set_montage(datasets_EEG=[eeg_raw])
                        eeg_raw.debug_test_annotations_timestamps()
                        _out_eeg_raw.append(eeg_raw)

                    _out_xdf_stream_infos_df.append(this_stream_infos)
                
            except (ValueError, KeyError, AssertionError, TypeError) as e:
                logger.warning('failed with error: %s\nskipping file.', e)
                if fail_on_exception:
                    raise
                else:
                    continue

            except Exception as e:
                logger.exception('failed with error: %s\nskipping file.', e)
                raise
                # continue
        ## END for an_xdf_file_idx, a_x...

        if len(_out_xdf_stream_infos_df) > 0:
            _out_xdf_stream_infos_df = pd.concat(_out_xdf_stream_infos_df)
            _out_xdf_stream_infos_df = _out_xdf_stream_infos_df.set_index('xdf_dataset_idx')
        else:
            _out_xdf_stream_infos_df = pd.DataFrame()

        if not should_load_full_file_data:
            _out_eeg_raw = None
            return _out_eeg_raw, _out_xdf_stream_infos_df, lab_recorder_xdf_files

        _out_eeg_raw = up_convert_raw_objects(_out_eeg_raw)
        _out_eeg_raw.sort(key=lambda r: (r.raw_timerange()[0] is None, r.raw_timerange()[0]))

        EEGData.set_montage(datasets_EEG=_out_eeg_raw)

        # _out_xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=_out_eeg_raw) # [_out_xdf_stream_infos_df['name'] == 'Epoc X']

        try:
            logger.info('trying to finalize _out_xdf_stream_infos_df columns...')
            _out_xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=_out_eeg_raw) # [_out_xdf_stream_infos_df['name'] == 'Epoc X']

        except Exception as e:
            logger.warning('finalization failed: %s\nYou can call it post-hoc like:\n\t`_out_xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=_out_xdf_stream_infos_df, active_only_out_eeg_raws=_out_eeg_raw)`\nreturning the non-processed _out_xdf_stream_infos_df.', e)
            # raise
            pass

        return _out_eeg_raw, _out_xdf_stream_infos_df, lab_recorder_xdf_files


    # ==================================================================================================================================================================================================================================================================================== #
    # Export/Saving Methods                                                                                                                                                                                                                                                                #
    # ==================================================================================================================================================================================================================================================================================== #

    @classmethod
    def save_post_processed_to_fif(cls, raws_dict, a_xdf_file: Path, labRecorder_PostProcessed_path: Path, export_mat: bool=True):
        """ 

        eeg_raw, a_lab_recorder_filepath = LabRecorderXDF.save_post_processed_to_fif(
            raws_dict=raws_dict,
            a_xdf_file=a_xdf_file,
            labRecorder_PostProcessed_path=sso.eeg_analyzed_parent_export_path.joinpath(f'LabRecorder_PostProcessed'),
        )

        LabRecorder_Apogee_2025-09-18T15-18-39
        LabRecorder_2025-09-19T02-22-10.mat
        
                 
        
        """
        ## When done processing the entire LabRecorder.xdf, save EEG data (with all
        ## annotations and such added) to new files. Supports multiple EEG devices
        ## per XDF by writing one output per merged EEG dataset.
        eeg_raws = raws_dict.get(DataModalityType.EEG.value, [])
        if len(eeg_raws) == 0:
            logger.warning('save_post_processed_to_fif found no EEG streams in "%s". Skipping.', a_xdf_file.as_posix())
            return None, None

        # Merge streams by device (one Raw per physical device)
        merged_eeg_raws, merge_meta = cls.merge_eeg_streams_by_device(
            eeg_raws=eeg_raws, strict_merge=False, debug_print=False
        )
        if len(merged_eeg_raws) == 0:
            logger.warning('save_post_processed_to_fif could not produce any merged EEG datasets for "%s". Skipping.', a_xdf_file.as_posix())
            return None, None

        labRecorder_PostProcessed_path.mkdir(exist_ok=True)

        a_lab_recorder_filename: str = a_xdf_file.stem
        a_clean_filename: str = a_lab_recorder_filename.removeprefix('LabRecorder_').removesuffix('_eeg')
        a_lab_recorder_filename_parts = a_clean_filename.split('_')

        datetime_part = a_lab_recorder_filename_parts[-1] if len(a_lab_recorder_filename_parts) > 0 else ""
        if len(a_lab_recorder_filename_parts) > 2:
            hostname_parts = '_'.join(a_lab_recorder_filename_parts[:-1])
            logger.debug('hostname_parts: %s will be discarded', hostname_parts)

        export_filepaths_dict = {}
        # For backward compatibility, return the first merged EEG Raw (primary)
        primary_eeg_raw = merged_eeg_raws[0]

        for idx, eeg_raw in enumerate(merged_eeg_raws):
            meas_date = eeg_raw.info.get('meas_date', None)
            if meas_date is not None:
                base_dt_str = meas_date.strftime("%Y-%m-%dT%H-%M-%S")
            else:
                base_dt_str = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")

            # Add device index suffix if there are multiple EEG datasets
            if len(merged_eeg_raws) > 1:
                device_suffix = f"eeg{idx}"
                final_output_filename = f"{base_dt_str}_{device_suffix}"
            else:
                final_output_filename = base_dt_str

            a_lab_recorder_filepath = labRecorder_PostProcessed_path.joinpath(final_output_filename).with_suffix('.fif')
            logger.info('saving finalized EEG data out to "%s"', a_lab_recorder_filepath.as_posix())
            eeg_raw.save(a_lab_recorder_filepath, overwrite=True)

            # Record per-format exports keyed by dataset index
            export_filepaths_dict.setdefault('fif', {})[idx] = a_lab_recorder_filepath

            if export_mat:
                mat_export_folder = a_lab_recorder_filepath.parent.joinpath('mat')
                mat_export_folder.mkdir(exist_ok=True)
                mat_export_path = mat_export_folder.joinpath(final_output_filename).with_suffix('.mat')
                mat_path = eeg_raw.save_to_fieldtrip_mat(mat_export_path)
                export_filepaths_dict.setdefault('mat', {})[idx] = mat_path

        return primary_eeg_raw, export_filepaths_dict
    

    @classmethod
    def to_hdf(cls, active_only_out_eeg_raws, results, xdf_stream_infos_df: pd.DataFrame, file_path: Path, root_key: str='/', debug_print=True):
        """ 
        from phopymnehelper.PendingNotebookCode import batch_compute_all_eeg_datasets
                
        LabRecorderXDF.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

        from phopymnehelper.EEG_data import EEGComputations

        active_only_out_eeg_raws, results = batch_compute_all_eeg_datasets(eeg_raws=_out_eeg_raw, limit_num_items=150, max_workers = 4)
                
        # EEGComputations.to_hdf(a_result=results[0], file_path="")
        hdf5_out_path: Path = Path('E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/outputs').joinpath('2025-09-23_eegComputations.h5').resolve()
        hdf5_out_path

        for idx, (a_raw, a_raw_outputs) in enumerate(zip(active_only_out_eeg_raws, results)):
            # a_path: Path = Path(a_raw.filenames[0])
            # basename: str = a_path.stem
            # basename: str = a_raw.info.get('meas_date')
            src_file_path: Path = Path(a_raw.info.get('description')).resolve()
            basename: str = src_file_path.stem

            _log.debug('basename: %s', basename)
            EEGComputations.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

            # EEGComputations.to_hdf(a_result=results[0], file_path="", root_key=f"/{basename}/")

            # for an_output_key, an_output_dict in a_raw_outputs.items():
            #     for an_output_subkey, an_output_value in an_output_dict.items():
            #         final_data_key: str = '/'.join([basename, an_output_key, an_output_subkey])
            #         print(f'\tfinal_data_key: "{final_data_key}"')
            #         # all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)

            # spectogram_result_dict = a_raw_outputs['spectogram']['spectogram_result_dict']
            # fs = a_raw_outputs['spectogram']['fs']

            # for ch_idx, (a_ch, a_ch_spect_result_tuple) in enumerate(spectogram_result_dict.items()):
            #     all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)
            #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)

            #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)


        # E:\Dropbox (Personal)\Databases\AnalysisData\MNE_preprocessed\outputs\


        """
        import h5py
        from phopymnehelper.EEG_data import EEGComputations

        write_mode = 'a'
        if (not file_path.exists()):
            write_mode = 'w'

        num_sessions: int = len(active_only_out_eeg_raws)
        xdf_stream_infos_df: pd.DataFrame = XDFDataStreamAccessor.init_from_results(_out_xdf_stream_infos_df=xdf_stream_infos_df, active_only_out_eeg_raws=active_only_out_eeg_raws)
        # xdf_stream_infos_df.to_hdf(file_path, key='/xdf_stream_infos_df', append=True) ## append=False to overwrite existing
        xdf_stream_infos_df.to_hdf(file_path, key='/xdf_stream_infos_df', append=True)

        flat_annotations = []

        for an_xdf_dataset_idx in np.arange(num_sessions):
            a_raw = active_only_out_eeg_raws[an_xdf_dataset_idx]
            a_meas_date = a_raw.info.get('meas_date')
            a_raw_key: str = a_meas_date.strftime("%Y-%m-%d/%H-%M-%S") # '2025-09-22/21-35-47'

            a_result = results[an_xdf_dataset_idx]
            with h5py.File(file_path, 'a') as f:
                EEGComputations.perform_write_to_hdf(a_result=a_result, f=f, root_key=f'/result/{a_raw_key}')

            # a_stream_info = deepcopy(xdf_stream_infos_df).loc[an_xdf_dataset_idx]    
            # print(f'i: {i}, a_meas_date: {a_meas_date}, a_stream_info: {a_stream_info}\n\n')
            # print(f'i: {an_xdf_dataset_idx}, a_meas_date: {a_meas_date}')
            # a_raw.to_data_frame(time_format='datetime').to_hdf(file_path, key=f'/raw/{a_raw_key}/df', append=True)
            a_raw.to_data_frame(time_format='datetime').to_hdf(file_path, key=f'/raw/{a_raw_key}', append=True)
            # EEGComputations.to_hdf(a_result=a_result, file_path=file_path, root_key=f'/result/{a_raw_key}')
            a_df = a_raw.annotations.to_data_frame(time_format='datetime')
            a_df = a_df[a_df['description'] != 'BAD_motion']
            # a_df['xdf_dataset_idx'] = an_xdf_dataset_idx
            flat_annotations.append(a_df)
                

        flat_annotations = pd.concat(flat_annotations, ignore_index=True)
        flat_annotations['onset_str'] = flat_annotations['onset'].dt.strftime("%Y-%m-%d_%I:%M:%S.%f %p")

        if flat_annotations is not None:
            flat_annotations.to_hdf(file_path, key='/flat_annotations_df', append=True)


        return file_path
    

    @classmethod
    def get_reference_datetime_from_xdf_header(cls, file_header: dict) -> Optional[datetime]:
        """Extract reference datetime from XDF file header.
        
        Args:
            file_header: XDF file header dictionary from pyxdf.load_xdf()
            
        Returns:
            datetime object if found, None otherwise
            
        The function checks multiple possible locations in the XDF header:
        - file_header['info']['recording']['start_time']
        - file_header['info']['recording']['start_time_s']
        - file_header['first_timestamp']
        - Other common locations

        ref_dt = LabRecorderXDF.get_reference_datetime_from_xdf_header(file_header=file_header)

        """
        if file_header is None:
            return None
        
        # Try various possible locations for recording start time
        possible_paths = [
            ['info', 'recording', 'start_time'],
            ['info', 'recording', 'start_time_s'],
            ['info', 'recording', 'startTime'],
            ['first_timestamp'],
            ['info', 'first_timestamp'],
            ['recording', 'start_time'],
        ]
        
        for path in possible_paths:
            try:
                value = file_header
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                        # XDF headers often have values as lists with single element
                        if isinstance(value, list) and len(value) > 0:
                            value = value[0]
                    else:
                        value = None
                        break
                
                if value is not None:
                    # Try to parse as datetime
                    if isinstance(value, (int, float)):
                        # Unix timestamp (seconds since epoch)
                        try:
                            dt = datetime.fromtimestamp(value, tz=timezone.utc)
                            return dt
                        except (ValueError, OSError):
                            # Try as milliseconds if seconds fails
                            try:
                                dt = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
                                return dt
                            except (ValueError, OSError):
                                logger.warning(f"Could not parse timestamp value {value} as Unix timestamp")
                                continue
                    elif isinstance(value, str):
                        # Try ISO format or other string formats
                        try:
                            # Try ISO format first
                            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            # Make timezone-aware if naive
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            return dt
                        except ValueError:
                            # Try other common formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']:
                                try:
                                    dt = datetime.strptime(value, fmt)
                                    # Make timezone-aware (assume UTC)
                                    dt = dt.replace(tzinfo=timezone.utc)
                                    return dt
                                except ValueError:
                                    continue
                            logger.warning(f"Could not parse datetime string: {value}")
                            continue
                    elif isinstance(value, datetime):
                        # Make timezone-aware if naive
                        if value.tzinfo is None:
                            value = value.replace(tzinfo=timezone.utc)
                        return value
            except (KeyError, TypeError, AttributeError) as e:
                continue
        
        logger.debug("Could not find recording start time in XDF header")
        return None



    @classmethod
    def perform_process_all_streams_multi_xdf(cls, streams_list: List[List], xdf_file_paths: List[Path], file_headers: Optional[List[Optional[dict]]] = None, **kwargs) -> Tuple[Dict, Dict]:
        """Process streams from multiple XDF files and **merge streams with the same name**.

        Streams with the same name across different files will be merged into a single datasource.
        Timestamps are converted to use a common reference datetime (earliest file's reference) to ensure
        proper alignment across multiple files.

        Args:
            streams_list: List of stream lists (one per XDF file), where each stream list contains
                        stream dictionaries from pyxdf
            xdf_file_paths: List of Path objects corresponding to each stream list
            file_headers: Optional list of XDF file header dictionaries (one per file)

        Returns:
            Tuple of (all_streams dict, all_streams_datasources dict) where:
            - all_streams: Dictionary mapping stream names to merged interval DataFrames
            - all_streams_datasources: Dictionary mapping stream names to merged TrackDatasource instances

        from phopymnehelper.xdf_files import XDFDataStreamAccessor, LabRecorderXDF
        from phopymnehelper.xdf_files import _get_channel_names_for_stream, _is_motion_stream, _is_eeg_quality_stream, _is_eeg_stream, _is_log_stream, merge_streams_by_name
        from phopymnehelper.xdf_files import modality_channels_dict, modality_sfreq_dict

        all_streams, all_streams_detailed_data = LabRecorderXDF.perform_process_all_streams_multi_xdf(streams_list=all_streams_by_file, xdf_file_paths=all_loaded_xdf_file_paths, file_headers=all_file_headers)


        """
        from phopymnehelper.historical_data import HistoricalData
        # from phopymnehelper.xdf_files import LabRecorderXDF


        def _subfn_build_intervals_df(stream_info: Dict, timestamps) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
            """ , series_vertical_offset: float = 0.0, color_name: str = 'blue', color_alpha: float = 0.3 """
            timestamps = np.asarray(timestamps, dtype=float)
            if len(timestamps) == 0:
                return timestamps, None

            stream_start = float(timestamps[0])
            stream_end = float(timestamps[-1])
            stream_duration = stream_end - stream_start
            if stream_duration <= 0 and len(timestamps) > 1:
                diffs = np.diff(timestamps)
                median_dt = float(np.median(diffs)) if len(diffs) > 0 else 0.0
                if median_dt > 0:
                    stream_duration = median_dt * (len(timestamps) - 1)
                else:
                    try:
                        nominal_srate = float(stream_info.get('nominal_srate', [[128.0]])[0][0])
                        stream_duration = (len(timestamps) - 1) / max(nominal_srate, 1.0)
                    except (TypeError, KeyError, IndexError, ValueError):
                        stream_duration = 1.0
                stream_end = stream_start + stream_duration

            intervals_df = pd.DataFrame({'t_start': [stream_start], 't_duration': [stream_duration], 't_end': [stream_end]})
            # intervals_df['series_vertical_offset'] = series_vertical_offset
            # intervals_df['series_height'] = 0.9

            # color = pg.mkColor(color_name)
            # color.setAlphaF(color_alpha)
            # intervals_df['pen'] = [pg.mkPen(color, width=1)]
            # intervals_df['brush'] = [pg.mkBrush(color)]
            return timestamps, intervals_df


        def _subfn_build_detailed_df(stream_info: Dict, stream_type: str, stream_name: str, timestamps: np.ndarray, time_series, strict_validation: bool = False) -> Optional[pd.DataFrame]:
            if time_series is None or len(time_series) == 0:
                return None
            n_channels = int(stream_info['channel_count'][0])
            n_t_stamps, n_columns = np.shape(time_series)
            if strict_validation:
                assert n_channels == n_columns, f"n_channels: {n_channels} != n_columns: {n_columns}"
                assert len(timestamps) == n_t_stamps, f"len(timestamps): {len(timestamps)} != n_t_stamps: {n_t_stamps}"
            elif not ((n_channels == n_columns) and (len(timestamps) == n_t_stamps)):
                return None

            time_series_df = pd.DataFrame(time_series, columns=_get_channel_names_for_stream(stream_type, stream_name, n_columns))
            time_series_df['t'] = timestamps
            return time_series_df


        def _subfn_process_xdf_file(xdf_path_for_raw: Path):
            """ 
                xdf_paths_for_raw = [v[1] for v in stream_file_pairs]
                raws_dict_dict = {}
                lab_obj_dict = {}
                for a_xdf_path in xdf_paths_for_raw:
                    a_lab_obj, a_raws_dict = _subfn_process_xdf_file(xdf_path_for_raw=a_xdf_path)
                    lab_obj_dict[a_xdf_path] = a_lap_obj
                    raws_dict_dict[a_xdf_path] = a_raws_dict


            """
            a_lab_obj = None
            a_raws_dict = {}
            logger.info(f'enable_raw_xdf_processing is True so this stream will be processed as MNE raw...')
            # xdf_path_for_raw = stream_file_pairs[0][1]
            if not xdf_path_for_raw.exists():
                return a_lab_obj, None
            logger.info(f'\ttrying to load raw XDF file load for stream_name: "{stream_name}" with xdf_path: "{xdf_path_for_raw}"...')
            try:
                a_lab_obj = cls.init_from_lab_recorder_xdf_file(a_xdf_file=xdf_path_for_raw, should_load_full_file_data=True)
            except ValueError as e:
                if 'datetime' in str(e).lower() or 'UTC' in str(e):
                    logger.warning(f'\tSkipping raw XDF file load for "{stream_name}" with xdf_path: {xdf_path_for_raw}: LabRecorderXDF load failed (UTC/datetime issue): {e}')
                else:
                    raise
            
            if a_lab_obj is not None:
                a_raws_dict = a_lab_obj.datasets_dict or {}
                logger.info(f'\traws_dict: {a_raws_dict}')

            return a_lab_obj, a_raws_dict

        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #

        if len(streams_list) != len(xdf_file_paths):
            raise ValueError(f"streams_list length ({len(streams_list)}) must match xdf_file_paths length ({len(xdf_file_paths)})")

        # Extract reference datetimes from file headers
        file_reference_datetimes = {}

        xdf_recording_file_metadata_df: pd.DataFrame = HistoricalData.build_file_comparison_df(recording_files=xdf_file_paths) ## this should be cheap because most will already be cached

        if file_headers is None:
            file_headers = [None for _ in xdf_file_paths]

        for file_header, file_path in zip(file_headers, xdf_file_paths):
            ref_dt = None
            if file_header is not None:
                ref_dt = cls.get_reference_datetime_from_xdf_header(file_header=file_header)
            if ref_dt is not None:
                file_reference_datetimes[file_path] = ref_dt
            else:
                resolved_file_path = Path(file_path).resolve()
                resolved_src_files = [Path(str(src_file)).resolve() == resolved_file_path for src_file in xdf_recording_file_metadata_df['src_file'].tolist()]
                found_file_df_matches = xdf_recording_file_metadata_df[resolved_src_files]
                if len(found_file_df_matches) == 1:
                    meas_datetime = found_file_df_matches.iloc[0]['meas_datetime'] if not found_file_df_matches.empty else None
                    ref_dt = meas_datetime
                else:
                    print(f'WARN: failed to find xdf file metadata for file file_path: "{file_path.as_posix()}" in xdf_recording_file_metadta_df: {xdf_recording_file_metadata_df}\n\tfound_file_df_matches: {found_file_df_matches}')

            if ref_dt is not None:
                file_reference_datetimes[file_path] = ref_dt
        ## END for file_header, file_path in zip(file_headers, xdf_file_paths):...

        # Find earliest reference datetime (common reference for all timestamps)
        earliest_reference_datetime = None
        if file_reference_datetimes:
            earliest_reference_datetime = min(file_reference_datetimes.values())
            print(f"Using earliest reference datetime: {earliest_reference_datetime} for timestamp normalization")

        # Group streams by name across all files
        streams_by_file = list(zip(streams_list, xdf_file_paths))
        streams_by_name = merge_streams_by_name(streams_by_file)

        all_streams = {}
        all_streams_detailed_data = {}

        # Process each unique stream name
        for stream_name, stream_file_pairs in streams_by_name.items():
            print(f"\nProcessing stream '{stream_name}' from {len(stream_file_pairs)} file(s)...")

            # Collect intervals and detailed data from all files for this stream name
            all_intervals_dfs = []
            all_detailed_dfs = [] ## #TODO 2026-03-02 08:56: - [ ] these detailed_dfs are being built synchronously in the following loop too, PERFORMANCE: defer this to an async call
            stream_type = None
            lab_obj_dict: Dict[str, Optional[LabRecorderXDF]] = {}

            for stream, file_path in stream_file_pairs:
                current_stream_type = stream['info']['type'][0]
                if stream_type is None:
                    stream_type = current_stream_type
                elif stream_type != current_stream_type:
                    print(f"WARN: Stream '{stream_name}' has different types across files: {stream_type} vs {current_stream_type}")

                timestamps = stream['time_stamps']
                time_series = stream['time_series']

                if len(timestamps) == 0:
                    print(f"  Skipping empty stream from {file_path.name}")
                    continue

                # Convert timestamps: from relative (to file's reference) to absolute, then to relative (to earliest reference)
                file_ref_dt = file_reference_datetimes.get(file_path)
                if (file_ref_dt is not None) and (timestamps is not None):
                    timestamps_absolute = float_to_datetime(timestamps, file_ref_dt)
                    timestamps = datetime_to_unix_timestamp(timestamps_absolute)
                else:
                    timestamps = np.asarray(timestamps, dtype=float)
                    if file_ref_dt is None:
                        print(f"  WARN: No reference datetime found for {file_path.name}, timestamps may be misaligned")

                timestamps = np.asarray(timestamps, dtype=float)

                timestamps, intervals_df = _subfn_build_intervals_df(stream['info'], timestamps) # , series_vertical_offset=0.0, color_name='blue', color_alpha=0.3
                all_intervals_dfs.append(intervals_df)

                # Create *detailed* DataFrame if time_series exists
                time_series_df = _subfn_build_detailed_df(stream['info'], stream_type, stream_name, timestamps, time_series, strict_validation=False)
                if time_series_df is not None:
                    all_detailed_dfs.append(time_series_df)
            ## END for stream, file_path in stream_file_pairs


            # Build the simple intervals _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            # Check if we have valid intervals
            if not all_intervals_dfs:
                print(f"  No valid intervals for stream '{stream_name}'")
                continue

            # Merge intervals for display (all_streams dict)
            merged_intervals_df = pd.concat(all_intervals_dfs, ignore_index=True).sort_values('t_start')
            all_streams[stream_name] = merged_intervals_df
            assert stream_type is not None

            has_valid_intervals = len(merged_intervals_df) > 0
            has_detailed_data = len(all_detailed_dfs) > 0
            if has_detailed_data:
                merged_detailed_df = pd.concat(all_detailed_dfs, ignore_index=True).sort_values('t')
                all_streams_detailed_data[stream_name] = merged_detailed_df

        ## END for stream_name, stream_file_pairs in streams_by_name.items()

        return all_streams, all_streams_detailed_data
