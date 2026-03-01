import time
import re
from datetime import datetime, timezone

import uuid
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from matplotlib import pyplot as plt

from pathlib import Path
import numpy as np
import pandas as pd

import mne
from mne import set_log_level
from copy import deepcopy
import mne

from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers
from phoofflineeeganalysis.analysis.historical_data import HistoricalData ## for creating single EDF+ files containing channel with different sampling rates (e.g. EEG and MOTION data)

mne.viz.set_browser_backend("Matplotlib")

# from ..EegProcessing import bandpower
from numpy.typing import NDArray


set_log_level("WARNING")


class EventData(HistoricalData):
    """ Methods related to processing of logging/event data
    
    from phoofflineeeganalysis.analysis.event_data import EventData
    
    (all_data_PHO_LOG, all_times_PHO_LOG), datasets_PHO_LOG, df_PHO_LOG = flat_data_modality_dict['PHO_LOG_TO_LSL']  ## Unpacking
    
    (active_PHO_LOG_IDXs, analysis_results_PHO_LOG) = EventData.preprocess(datasets_PHO_LOG, n_most_recent_sessions_to_preprocess=5)


    """
    @classmethod
    def extract_datetime_from_filename(cls, filename) -> datetime:
        """Extract recording start datetime from filename.
        Examples:
            '20250820_035626_log.fif'
            'Debut_2025-08-18T122633.words.lsl.fif'
        """
        patterns = [
            (r'(\d{8})_(\d{6})', "%Y%m%d%H%M%S"),         # 20250820_035626
            (r'(\d{4}-\d{2}-\d{2})T(\d{6})', "%Y-%m-%d%H%M%S")  # 2025-08-18T122633
        ]
        for pat, fmt in patterns:
            match = re.search(pat, filename)
            if match:
                date_str, time_str = match.groups()
                return datetime.strptime(date_str + time_str, fmt)
        raise ValueError(f"Filename '{filename}' does not contain a recognized datetime format.")


    @classmethod
    def get_or_parse_datetime_from_raw(cls, raw, allow_setting_meas_date_from_filename:bool=True, force_override_from_parsed_filename: bool=True) -> datetime:
        """ Get the recording start datetime from the raw.info['meas_date'] or parse from filename if not present
        if allow_setting_meas_date_from_filename is True, it will set the raw.info['meas_date'] if it was None
        """        
        metadata_recording_start_datetime = raw.info.get('meas_date', None)
        if (metadata_recording_start_datetime is None) or force_override_from_parsed_filename:
            parsed_recording_start_datetime = cls.extract_datetime_from_filename(Path(raw.filenames[0]).name)
            metadata_recording_start_datetime = parsed_recording_start_datetime
            assert metadata_recording_start_datetime is not None, f"Failed to parse recording start datetime from filename '{Path(raw.filenames[0]).name}'"
            if allow_setting_meas_date_from_filename:                
                # Make timezone-aware (UTC)
                # dt = dt.rerawe(tzinfo=timezone.utc)
                raw.info.set_meas_date(deepcopy(parsed_recording_start_datetime).replace(tzinfo=timezone.utc))
                ## WAS UPDATED, probably need to re-save or something
                meas_datetime = raw.info['meas_date']
            else:
                ## don't set it, but still use the parsed datetime
                meas_datetime = deepcopy(parsed_recording_start_datetime).replace(tzinfo=timezone.utc) # raw.info['meas_date']  # This is an absolute datetime or tuple (timestamp, 0)

        else:                    
            meas_datetime = raw.info['meas_date']  # This is an absolute datetime or tuple (timestamp, 0)

        return meas_datetime
    

    @classmethod
    def _perform_process_event_session(cls, raw_event: mne.io.Raw):
        """ Called by `preprocess` on each rawMotion session
         
            motion_analysis_results = _perform_process_motion_session(raw_motion)
            motion_analysis_results['quaternion_df'].head()
        """
        an_analysis_results = {}
        annots = raw_event.annotations
        metadata_recording_start_datetime = raw_event.info.get('meas_date', None)
        assert metadata_recording_start_datetime is not None, f"raw_event.info['meas_date'] is None, cannot proceed"
        # annots._orig_time = metadata_recording_start_datetime.timestamp()  # Set the origin time for annotations
        annots_df = annots.to_data_frame()
        annots_df['onset'] = metadata_recording_start_datetime.timestamp() + annots_df['onset'].dt.second  ## Convert onset to float if not already
        annots_df = MNEHelpers.convert_df_columns_to_datetime(annots_df, dt_col_names=["onset"])        
        # annots_df['onset_dt'] = pd.to_datetime(annots_df['onset'], unit='s') ## add datetime column
        # annots.set_orig_time(metadata_recording_start_datetime)  # Set the origin time for annotations        
        an_analysis_results['PHO_LOG'] = {'annots': annots, 'df': annots_df, 'meas_datetime': metadata_recording_start_datetime}
        return an_analysis_results


    @classmethod
    def preprocess(cls, datasets_PHO_LOG, n_most_recent_sessions_to_preprocess: Optional[int] = 5):
        """ 
        preprocessed_EEG_save_path: Path = eeg_analyzed_parent_export_path.joinpath('preprocessed_EEG').resolve()
        preprocessed_EEG_save_path.mkdir(exist_ok=True)

        ## INPUTS: flat_data_modality_dict
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = flat_data_modality_dict['EEG']  ## Unpacking


        """
        ## BEGIN ANALYSIS of EEG Data
        num_EVENT_files: int = len(datasets_PHO_LOG)
        event_session_IDXs = np.arange(num_EVENT_files)

        if (n_most_recent_sessions_to_preprocess is not None) and (n_most_recent_sessions_to_preprocess > 0):
            n_most_recent_sessions_to_preprocess = min(n_most_recent_sessions_to_preprocess, num_EVENT_files) ## don't process more than we have
            active_event_IDXs = event_session_IDXs[-n_most_recent_sessions_to_preprocess:]
        else:
            ## ALL sessions
            active_event_IDXs = deepcopy(event_session_IDXs)

        analysis_results_EVENT = []
        valid_active_IDXs = []
        
        for i in active_event_IDXs:
            event_analysis_results = None
            try:
                a_raw_event_log = datasets_PHO_LOG[i]
                # good_channels = a_raw_motion.pick_types(eeg=True)
                # sampling_rate = a_raw_motion.info["sfreq"]  # Store the sampling rate
                # print(f'sampling_rate: {sampling_rate}')
                a_meas_datetime = cls.get_or_parse_datetime_from_raw(a_raw_event_log, allow_setting_meas_date_from_filename=True)            
                a_raw_event_log.load_data()
                # datasets_MOTION[i] = a_raw_motion ## update it and put it back
                event_analysis_results = cls._perform_process_event_session(a_raw_event_log)
                
            except ValueError as e:
                print(f'Encountered value error: {e} while trying to processing EVENT file {i}/{len(datasets_PHO_LOG)}: {a_raw_event_log}. Skipping')
                datasets_PHO_LOG[i] = None ## drop result
                event_analysis_results = None ## no analysis result
                pass
            except Exception as e:
                raise e

            if event_analysis_results is not None:
                analysis_results_EVENT.append(event_analysis_results)
                valid_active_IDXs.append(i)


        valid_active_IDXs = np.array(valid_active_IDXs)
        
        ## OUTPUTS: analysis_results_EEG
        ## UPDATES: eeg_session_IDXs
        return (valid_active_IDXs, analysis_results_EVENT)
    

    @classmethod
    def join_event_dfs(cls, active_PHO_LOG_IDXs, analysis_results_PHO_LOG, debug_print=False):
        n_split_sessions = len(analysis_results_PHO_LOG)
        # analysis_results_PHO_LOG[0]

        all_annotations = []
        all_annotations_df = []
        for i, a_pho_log_IDX in enumerate(active_PHO_LOG_IDXs):
            if debug_print:
                print(f"Session {i+1} of {n_split_sessions}:")
            a_log_result_dict = analysis_results_PHO_LOG[i]['PHO_LOG']
            a_log_df = a_log_result_dict['df']
            a_log_meas_datetime = a_log_result_dict['meas_datetime']
            if debug_print:
                print(f'a_log_meas_datetime: {a_log_meas_datetime}')
            an_annotation = mne.Annotations(
                onset=a_log_df['onset'].values,
                duration=a_log_df['duration'].values,
                description=a_log_df['description'].values,
                orig_time=a_log_meas_datetime
            )

            # a_log_df['onset'] = a_log_meas_datetime.timestamp() + a_log_df['onset'].dt.second  ## Convert onset to float if not already
            # a_log_df = MNEHelpers.convert_df_columns_to_datetime(a_log_df, dt_col_names=["onset"])
            # print("\n")
            all_annotations.append(an_annotation)
            # all_annotations.append(a_log.to_data_frame())
            all_annotations_df.append(a_log_df)

        all_annotations_df = pd.concat(all_annotations_df, ignore_index=True)
        return all_annotations_df, all_annotations
    


    @classmethod
    def complete_correct_COMMON_annotation_df(cls, a_logging_modality, dataset_idx_col_name: str='DATASET_idx', include_full_file_path: bool=False) -> pd.DataFrame:
        """INPUTS: sso.modalities['PHO_LOG_TO_LSL']
        Usage:
            from phoofflineeeganalysis.analysis.event_data import EventData

            all_pho_log_to_lsl_df: pd.DataFrame = EventData.complete_correct_COMMON_annotation_df(a_logging_modality=sso.modalities['PHO_LOG_TO_LSL'], dataset_idx_col_name='MOTION_idx')
            all_pho_log_to_lsl_df
        """
        all_dfs = []

        for raw_idx, active_DATASET_idx in enumerate(a_logging_modality.active_indices):
            a_raw_data: mne.io.Raw = a_logging_modality.datasets[active_DATASET_idx]
            an_info: mne.Info = a_raw_data.info
            a_meas_date: datetime = deepcopy(an_info.get('meas_date')) ## Importantly, the startdate in absolute datetime of the file
            a_file_paths = a_raw_data.filenames
            if len(a_file_paths) > 0:
                a_file_path = Path(a_file_paths[-1]) ## last filename
            else:
                a_file_path = None ## None


            an_annotations = a_raw_data.annotations
            is_annotation_orig_time_wrong: bool = (an_annotations.orig_time != a_meas_date)
            if is_annotation_orig_time_wrong:
                # print(f'an_annotations.orig_time: {an_annotations.orig_time} != a_meas_date: {a_meas_date}')
                # an_annotations.set_orig_time(a_meas_date)
                # an_annotations.orig_time = a_meas_date
                a_df: pd.DataFrame = deepcopy(an_annotations.to_data_frame(time_format='timedelta'))
                ## Update the "onset" column so that it's correct:
                onset_abstimes = pd.to_datetime(a_meas_date) + pd.to_timedelta(a_df['onset'].to_numpy(), unit='s')
                a_df['onset'] = onset_abstimes

            else:
                print(f'original annotation time correct!')
                a_df: pd.DataFrame = deepcopy(an_annotations.to_data_frame(time_format='datetime')) ## should be absolute now


            ## Add extra columns:
            if include_full_file_path:
                a_df['filepath'] = a_file_path
                
            a_df['filename'] = a_file_path.name
            # a_df['raw_idx'] = raw_idx
            if dataset_idx_col_name is not None:
                a_df[dataset_idx_col_name] = active_DATASET_idx
                
            a_df['file_meas_date'] = a_meas_date

            ## OUTPUTS: a_df
            all_dfs.append(a_df)
        ## END for active_PHO_LOG_TO_LSL_idx in sso.moda...    
        # Concatenate all dfs to a single one:
        all_dfs: pd.DataFrame = pd.concat(all_dfs, ignore_index=True, verify_integrity=True)
        return all_dfs


    @classmethod
    def perform_fixup_WHISPER_annotation_df(cls, all_WHISPER_df: pd.DataFrame, WHISPER_idx_col_name: str='WHISPER_idx') -> pd.DataFrame:
        """ Called by `cls.complete_correct_WHISPER_annotation_df(...)
        ## Process WHISPER transcribed dataframes, performing the following operations:
        ### 1. Removing redundant entries, keeping only the "onset" and computing the correct "duration" for each entry.
        ### 2. Dropping common error transcriptions, such as the empty transcript "..." or "Thanks for watching!".
        ### 3. Returning a valid dataframe containing all of the transcription data with real/proper absolute datetime timestamps.
        """
        groups = all_WHISPER_df.groupby((all_WHISPER_df['description'] != all_WHISPER_df['description'].shift()).cumsum()) # grouped by 'description' value
        results = []
        for _, g in groups:
            if len(g) > 1:
                first, last = g.iloc[0], g.iloc[-1]
                duration: float = last['onset'] - first['onset']
                # results.append({'description': first['description'], 'start': first['onset'], 'end': last['onset'], 'duration': duration})
                _additional_col_names = ['filepath', 'filename', WHISPER_idx_col_name, 'file_meas_date']
                # _additional_cols = {'filepath': first['filepath'], 'filename': first['filename'], 'WHISPER_idx': first['WHISPER_idx'], 'file_meas_date': first['file_meas_date']}                
                _additional_cols = {k:first[k] for k in _additional_col_names if (first.get(k, None) is not None)}
                results.append({'onset': first['onset'], 'duration': duration, 'description': first['description'], **_additional_cols})

        dedup_WHISPER_df = pd.DataFrame(results)
        return dedup_WHISPER_df

    @classmethod
    def complete_correct_WHISPER_annotation_df(cls, a_logging_modality, **kwargs) -> pd.DataFrame:
        """INPUTS: sso.modalities['WHISPER']
        Calls: cls.perform_fixup_WHISPER_annotation_df(...)
        
        Usage:
            from phoofflineeeganalysis.analysis.event_data import EventData

            all_WHISPER_df: pd.DataFrame = EventData.complete_correct_WHISPER_annotation_df(a_logging_modality=sso.modalities['WHISPER'])
            all_WHISPER_df
        """
        WHISPER_idx_col_name: str = 'WHISPER_idx'
        all_dfs: pd.DataFrame = cls.complete_correct_COMMON_annotation_df(a_logging_modality=a_logging_modality, dataset_idx_col_name=WHISPER_idx_col_name, include_full_file_path=False, **kwargs)
        all_dfs = cls.perform_fixup_WHISPER_annotation_df(all_WHISPER_df=all_dfs, WHISPER_idx_col_name=WHISPER_idx_col_name)
        return all_dfs
    



    @classmethod
    def complete_correct_Pho_Log_To_LSL_annotation_df(cls, a_logging_modality, **kwargs) -> pd.DataFrame:
        """INPUTS: sso.modalities['PHO_LOG_TO_LSL']
        Usage:
            from phoofflineeeganalysis.analysis.event_data import EventData

            all_pho_log_to_lsl_df: pd.DataFrame = EventData.complete_correct_Pho_Log_To_LSL_annotation_df(a_logging_modality=sso.modalities['PHO_LOG_TO_LSL'])
            all_pho_log_to_lsl_df
        """
        all_dfs: pd.DataFrame = cls.complete_correct_COMMON_annotation_df(a_logging_modality=a_logging_modality, dataset_idx_col_name='PHO_LOG_TO_LSL_idx', include_full_file_path=False, **kwargs)
        return all_dfs



