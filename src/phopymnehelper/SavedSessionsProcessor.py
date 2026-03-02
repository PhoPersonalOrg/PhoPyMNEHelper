from datetime import datetime
# from pytz import timezone

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import dill

from typing import Dict, List, Tuple, Optional, Callable, Any

from pathlib import Path
import pandas as pd

from mne import set_log_level
from copy import deepcopy
import mne

# from phopymnehelper.tzinfo_examples import Eastern


datasets = []
mne.viz.set_browser_backend("Matplotlib")
from attrs import define, field

# from ..EegProcessing import bandpower

from phopymnehelper.analysis.computations.EEG_data import EEGData
from phopymnehelper.motion_data import MotionData
from phopymnehelper.event_data import EventData
from phopymnehelper.historical_data import HistoricalData

set_log_level("WARNING")

from phopylslhelper.core.data_modalities import DataModalityType

# import pyxdf
import mne
import numpy as np
# from benedict import benedict

# from phopylslhelper.general_helpers import unwrap_single_element_listlike_if_needed, readable_dt_str, from_readable_dt_str, localize_datetime_to_timezone, tz_UTC, tz_Eastern, _default_tz
# from phopylslhelper.easy_time_sync import EasyTimeSyncParsingMixin


@define(slots=False)
class SessionModality:
    """ Data corresponding to a specific type or 'modality' of input (e.g. EEG, MOTION, PHO_LOG_TO_LSL, WHISPER, etc.
    """
    all_data: Optional[Any] = field(default=None)
    all_times: Optional[Any] = field(default=None)
    datasets: Optional[Any] = field(default=None)
    df: Optional[pd.DataFrame] = field(default=None)
    active_indices: Optional[Any] = field(default=None)
    analysis_results: Optional[Any] = field(default=None)


    def filtered_by_day_date(self, search_day_date: datetime, debug_print=False) -> "SessionModality":
        """ Returns a new SessionModality instance filtered to only include datasets from the specified date.
        
        today_only_modality = a_modality.filtered_by_day_date(search_day_date=datetime(2025, 8, 8))
        
        
        """
        if self.df is None or self.datasets is None:
            raise ValueError("Both 'df' and 'datasets' must be loaded to filter by date.")

        # Ensure the date has no time component
        search_day_date = search_day_date.replace(hour=0, minute=0, second=0, microsecond=0)

        today_only_modality = deepcopy(self)
        is_dataset_included = np.isin(self.active_indices, self.df[self.df['day'] == search_day_date]['dataset_IDX'].values)
        if debug_print:
            print(f'\tis_dataset_included: {is_dataset_included}')
        today_only_modality.df = self.df[self.df['day'] == search_day_date] ## filter the today_only modalities version
        today_only_modality.active_indices = self.active_indices[is_dataset_included]
        # _curr_included_IDXs = np.arange(len(a_modality.datasets))[is_dataset_included]
        # print(f'\t_curr_included_IDXs: {_curr_included_IDXs}')
        # today_only_modality.datasets = [a_modality.datasets[i] for i in _curr_included_IDXs]
        today_only_modality.datasets = [self.datasets[i] for i in today_only_modality.active_indices]
        today_only_modality.analysis_results = [self.analysis_results[i] for i in today_only_modality.active_indices]
        return today_only_modality



@define(slots=False)
class SavedSessionsProcessor:
    """ Top-level manager of EEG recordings
    

    from phopymnehelper.SavedSessionsProcessor import SavedSessionsProcessor, SessionModality, DataModalityType
     
    sso: SavedSessionsProcessor = SavedSessionsProcessor()
    sso

    """
    eeg_recordings_file_path: Path = field(default=Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/fif'))
    headset_motion_recordings_file_path: Path = field(default=Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif'))
    WhisperVideoTranscripts_LSL_Converted_file_path: Path = field(default=Path(r"E:/Dropbox (Personal)/Databases/UnparsedData/WhisperVideoTranscripts_LSL_Converted"))
    pho_log_to_LSL_recordings_path: Path = field(default=Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/PhoLogToLabStreamingLayer_logs'))
    ## These contain little LSL .fif files with names like: '20250808_062814_log.fif', 

    eeg_analyzed_parent_export_path: Path = field(default=Path("E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed"))

    # n_most_recent_sessions_to_preprocess: int = None # None means all sessions
    n_most_recent_sessions_to_preprocess: int = field(default=10) #
    should_load_data: bool = field(default=False)
    should_load_preprocessed: bool = field(default=False)

    ## Loaded variables
    found_recording_file_modality_dict: Dict[str, List[Path]] = field(factory=dict, init=False)
    flat_data_modality_dict: Dict[str, Tuple] = field(factory=dict, init=False)

    ## This is the core data-storage variable for this class, that holds all the loaded/parsed results and datasets
    modalities: Dict[str, SessionModality] = field(factory=dict, init=False)


    def run(self):
        """ Loads data (either fresh or pre-processed) and then calls `self.perform_post_processing()`

        Calls:        
                self.perform_post_processing()
                
        """
        ## Load pre-proocessed EEG data:
        if self.should_load_preprocessed:
            self.flat_data_modality_dict, self.found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
                eeg_recordings_file_path = self.eeg_analyzed_parent_export_path,
                # headset_motion_recordings_file_path = self.headset_motion_recordings_file_path,
                # WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path,
                # pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path,
                should_load_data=self.should_load_data,
            )
            ## Just get the previously processed EEG data, do not load other modalities            

            # self.flat_data_modality_dict, self.found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
            #     eeg_recordings_file_path = self.eeg_analyzed_parent_export_path,
            #     # headset_motion_recordings_file_path = self.headset_motion_recordings_file_path,
            #     WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path,
            #     pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path,
            #     should_load_data=self.should_load_data,
            # )

            ## #TODO 2025-09-09 16:14: - [ ] Find the files that changed since last processing, and only load those:
            self.flat_data_modality_dict, self.found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
                eeg_recordings_file_path = self.eeg_recordings_file_path,
                headset_motion_recordings_file_path = self.headset_motion_recordings_file_path,
                WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path,
                pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path,
                should_load_data=self.should_load_data,
            )


        else:
            ## Old way:
            self.flat_data_modality_dict, self.found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
                eeg_recordings_file_path = self.eeg_recordings_file_path,
                headset_motion_recordings_file_path = self.headset_motion_recordings_file_path,
                WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path,
                pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path,
                should_load_data=self.should_load_data,
            )


        # 1m 10s

        self.perform_post_processing()
        


    def perform_post_processing(self) -> Dict[str, SessionModality]:
        """Performs batch post-processing on all loaded modalities in `self.flat_data_modality_dict`.

        Runs each modality's preprocessing in parallel (threaded) since operations
        are independent and operate on different files. Returns a mapping from
        modality key to `SessionModality` with results.

        
        Calls: 
            self.perform_extended_post_processing_steps()
            
        """
        # Map modality keys to their preprocessors and any relevant param
        preprocessors: Dict[str, Callable[..., Tuple[Any, Any]]] = {
            "EEG": EEGData.preprocess,
            "MOTION": MotionData.preprocess,
            "PHO_LOG_TO_LSL": EventData.preprocess,
            "WHISPER": EventData.preprocess,
        }

        # Only process modalities that are actually present
        keys_to_process: List[str] = [k for k in preprocessors.keys() if k in self.flat_data_modality_dict]

        results: Dict[str, SessionModality] = {}
        errors: Dict[str, Exception] = {}

        def _process_modality(key: str) -> Tuple[str, SessionModality]:
            preproc_func = preprocessors[key]
            unpacked = self.flat_data_modality_dict[key]
            all_data, all_times = unpacked[0]
            datasets = unpacked[1]
            df = unpacked[2]

            print(f'\tstarting post-process modality: {key}')
            if key == "EEG":
                active_indices, analysis_results = preproc_func(
                    datasets_EEG=datasets,
                    preprocessed_EEG_save_path=None,
                    n_most_recent_sessions_to_preprocess=self.n_most_recent_sessions_to_preprocess,
                )
            else:
                active_indices, analysis_results = preproc_func(
                    datasets,
                    n_most_recent_sessions_to_preprocess=self.n_most_recent_sessions_to_preprocess,
                )

            modality_result = SessionModality(
                all_data=all_data,
                all_times=all_times,
                datasets=datasets,
                df=df,
                active_indices=active_indices,
                analysis_results=analysis_results,
            )
            print(f'\tfinished post-process modality: {key}')
            return key, modality_result

        max_workers = max(1, min(len(keys_to_process), (os.cpu_count() or 4)))
        if keys_to_process:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {executor.submit(_process_modality, key): key for key in keys_to_process}
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        k, modality_result = future.result()
                        results[k] = modality_result
                    except Exception as e:
                        print(f"\tERROR while post-processing modality '{key}': {e}")
                        errors[key] = e

        # Update self.modalities with successful results
        for k, modality_result in results.items():
            self.modalities[k] = modality_result

        # Perform extended steps that depend on multiple modalities
        try:
            self.perform_extended_post_processing_steps()
        except (ValueError, TypeError, AttributeError) as e:
            print(f'encountered error: {e} while trying to perform perform_extended_post_processing_steps(). Skipping and returning.')
        except Exception as e:
            raise e

        return results
        
    def setup_specific_modality(self, modality_type: List[DataModalityType], should_load_data: bool=False):
        """ called to discover and load all files related to a specific modality, such as EEG, WHISPER recordings, etc.
        
        """
        if not isinstance(modality_type, (list, tuple)):
            ## wrap in a list
            modality_type = [modality_type] ## single element list


        MAIN_process_recording_files_kwargs = {}
        for a_modality in modality_type:
            ## find the correct kwarg name and corresponding value
            if a_modality.name == DataModalityType.EEG.name:
                MAIN_process_recording_files_kwargs.update(dict(eeg_recordings_file_path = self.eeg_recordings_file_path))
            elif a_modality.name == DataModalityType.MOTION.name:
                MAIN_process_recording_files_kwargs.update(dict(headset_motion_recordings_file_path = self.headset_motion_recordings_file_path))
            elif a_modality.name == DataModalityType.PHO_LOG_TO_LSL.name:
                MAIN_process_recording_files_kwargs.update(dict(pho_log_to_LSL_recordings_path = self.pho_log_to_LSL_recordings_path))
            elif a_modality.name == DataModalityType.WHISPER.name:
                MAIN_process_recording_files_kwargs.update(dict(WhisperVideoTranscripts_LSL_Converted = self.WhisperVideoTranscripts_LSL_Converted_file_path))
            # elif a_modality.name == DataModalityType.EEG.name:
            # 	MAIN_process_recording_files_kwargs.update(dict(eeg_recordings_file_path = self.eeg_recordings_file_path))
            else:
                raise NotImplementedError(f'Unknown modality type: {a_modality}')


        flat_data_modality_dict, found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(
                        **MAIN_process_recording_files_kwargs,
                        should_load_data=should_load_data,
        )
        

        ## iterate and add to self
        for k, v in flat_data_modality_dict.items():
            self.flat_data_modality_dict[k] = v
        
        for k, v in found_recording_file_modality_dict.items():
            self.found_recording_file_modality_dict[k] = v

        ## self.modalities is not changed :[

        return (flat_data_modality_dict, found_recording_file_modality_dict)
    


    def perform_extended_post_processing_steps(self):
        # Do annotation/join only if needed, still avoid repetition:
        if ("PHO_LOG_TO_LSL" in self.modalities):
            (dataset_PHOLOG_df, dataset_EEG_df_PHOLOG) = HistoricalData.add_additional_LOGGING_annotations(
                active_EEG_IDXs=self.modalities["EEG"].active_indices,
                datasets_EEG=self.modalities["EEG"].datasets,
                active_LOGGING_IDXs=self.modalities["PHO_LOG_TO_LSL"].active_indices,
                datasets_LOGGING=self.modalities["PHO_LOG_TO_LSL"].datasets,
                analysis_results_LOGGING=self.modalities["PHO_LOG_TO_LSL"].analysis_results,
                logging_series_identifier="PHO_LOG",
                preprocessed_EEG_save_path=None
            )
            if dataset_EEG_df_PHOLOG is not None:
                self.modalities["EEG"].df = dataset_EEG_df_PHOLOG
            if dataset_PHOLOG_df is not None:
                self.modalities["PHO_LOG_TO_LSL"].df = dataset_PHOLOG_df


        if ("WHISPER" in self.modalities):
            (dataset_WHISPER_df, dataset_EEG_df_WHISPER) = HistoricalData.add_additional_LOGGING_annotations(
                active_EEG_IDXs=self.modalities["EEG"].active_indices,
                datasets_EEG=self.modalities["EEG"].datasets,
                active_LOGGING_IDXs=self.modalities["WHISPER"].active_indices,
                datasets_LOGGING=self.modalities["WHISPER"].datasets,
                analysis_results_LOGGING=self.modalities["WHISPER"].analysis_results,
                logging_series_identifier="WHISPER",
                preprocessed_EEG_save_path=None
            )
            self.modalities["EEG"].df = dataset_EEG_df_WHISPER
            self.modalities["WHISPER"].df = dataset_WHISPER_df

        if ("EEG" in self.modalities) and ("MOTION" in self.modalities):
            dataset_MOTION_df, dataset_EEG_df = HistoricalData.add_bad_periods_from_MOTION_data(active_EEG_IDXs=self.modalities["EEG"].active_indices,
                                                        datasets_EEG=self.modalities["EEG"].datasets,
                                                        active_motion_IDXs=self.modalities["MOTION"].active_indices, datasets_MOTION=self.modalities["MOTION"].datasets, analysis_results_MOTION=self.modalities["MOTION"].analysis_results,
                                                        preprocessed_EEG_save_path=self.eeg_analyzed_parent_export_path)
            self.modalities["EEG"].df = dataset_EEG_df
            self.modalities["MOTION"].df = dataset_MOTION_df


    # ==================================================================================================================================================================================================================================================================================== #
    # Pickling/Exporting                                                                                                                                                                                                                                                                   #
    # ==================================================================================================================================================================================================================================================================================== #

    def save(self, pkl_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/PICKLED_COLLECTION")):
        """ Pickles the object 
        """
        # pkl_path

        # data_path = Path(r"C:/Users/pho/repos/EmotivEpoc/PhoLabStreamingReceiver/data").resolve()
        # assert data_path.exists()

        # pickled_data_path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/PICKLED_COLLECTION")
        if pkl_path.is_dir():
            assert pkl_path.exists(), f"Directory {pkl_path.as_posix()} must exist!"
            pkl_path = pkl_path.joinpath("2025-09-02_50records_SSO_all.pkl")
        else:
            print(f'pkl_path is already a direct pkl file name: "{pkl_path.as_posix()}"')

        print(f'Pickling all data to "{pkl_path.as_posix()}"...')
        with open(pkl_path, "wb") as f:
            dill.dump(self, f)
        print(f'\tdone.')


    @classmethod
    def load(cls, pkl_file: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/PICKLED_COLLECTION/records_SSO_all.pkl")) -> "SavedSessionsProcessor":
        """ un-Pickles the object 
        
        sso: SavedSessionsProcessor = SavedSessionsProcessor.load(pkl_file=Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/PICKLED_COLLECTION/2025-09-02_50records_SSO_all.pkl")
        """
        assert pkl_file.exists(), f"'{pkl_file.as_posix()}' must exist!"
        assert pkl_file.exists(), f"'{pkl_file.is_file()}' must be a pickle file!"
        with open(pkl_file, "rb") as f:
            loaded_instance = dill.load(f)
            return loaded_instance



    # ==================================================================================================================================================================================================================================================================================== #
    # Exporting to other formats                                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #
    def save_to_EDF(self, edf_export_parent_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exported_EDF")) -> List[Path]:
        """ saves the EEG files (post-processing) out to EDF files for viewing in EDFViewer or similar applications.

        edf_export_parent_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exported_EDF")
                
        written_EDF_file_paths = sso.save_to_EDF()
        
        """
        from phopymnehelper.MNE_helpers import up_convert_raw_objects

        edf_export_parent_path.mkdir(exist_ok=True)
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = self.flat_data_modality_dict['EEG']  ## Unpacking
        datasets_EEG = up_convert_raw_objects(datasets_EEG) ## upconvert
        written_EDF_file_paths = []
        for i, raw_eeg in enumerate(datasets_EEG):
            ## INPUTS: raw_eeg
            ## Get paths for current raw:
            try:
                curr_fif_file_path: Path = Path(raw_eeg.filenames[0])
                curr_file_edf_name: str = curr_fif_file_path.with_suffix('.edf').name
                curr_file_edf_path: Path = edf_export_parent_path.joinpath(curr_file_edf_name)
                curr_file_edf_path = raw_eeg.save_to_edf(output_path=curr_file_edf_path)
                # EEGData.save_mne_raw_to_edf(raw_eeg, curr_file_edf_path)
                written_EDF_file_paths.append(curr_file_edf_path)
            except (ValueError, FileNotFoundError, FileExistsError, AttributeError, OSError, TypeError) as e:
                print(f'\tWARNING: could not export EEG dataset index {i} to EDF file, skipping... Error: {e}')
                
            except Exception as e:
                raise
        # END for i, raw_eeg in enumerate(datasets_EEG)...
        
        return written_EDF_file_paths
    

@define(slots=False)
class EntireDayMergedData:
    """ Manages data merged for an entire day
    
    from phopymnehelper.SavedSessionsProcessor import EntireDayMergedData
    
    """
    datasets: List[mne.io.Raw] = field(default=None)
    

    @classmethod
    def concatenate_with_gaps(cls, datasets: list[mne.io.Raw]) -> mne.io.Raw:
        """ #TODO 2025-09-09 22:09: - [ ] IMPORTATANT - the default MNE merge does not respect time at all
        """
        raws = []
        annotations = []
        total_duration = 0.0

        # Use the first dataset's orig_time as reference
        base_orig_time = datasets[0].annotations.orig_time

        for i, raw in enumerate(datasets):
            this_raw = deepcopy(raw)
            # Align annotation origins to the base_orig_time
            if this_raw.annotations is not None:
                this_raw.set_annotations(this_raw.annotations.copy())
                this_raw.annotations._orig_time = base_orig_time

            if i > 0:
                onset = total_duration
                ann = mne.Annotations(onset=[onset], duration=[0], description=["BAD_DISCONTINUITY"], orig_time=base_orig_time)
                annotations.append(ann)

            total_duration += this_raw.times[-1] + 1 / this_raw.info['sfreq']
            raws.append(this_raw)

        merged = mne.concatenate_raws(raws, preload=True)

        if annotations:
            combined = merged.annotations
            for ann in annotations:
                ann._orig_time = base_orig_time
                combined += ann
            merged.set_annotations(combined)

        return merged
    

    # @classmethod
    # def concatenate_datasets(cls, datasets: List[mne.io.Raw]) -> mne.io.Raw:
    #     """ Concatenates a list of mne.io.Raw datasets into a single Raw dataset.

    #     Args:
    #         datasets (List[mne.io.Raw]): List of Raw datasets to concatenate.

    #     Returns:
    #         mne.io.Raw: Concatenated Raw dataset.
    #     """
    #     if not datasets:
    #         raise ValueError("The datasets list is empty.")

    #     assert len(datasets) > 0, "The datasets list must contain at least one Raw object."

    #     concatenated_raw = deepcopy(datasets[0])
    #     # concatenated_raw = datasets[0]
    #     for raw in datasets[1:]:
    #         a_ds = deepcopy(raw)
    #         # concatenated_raw.append(deepcopy(raw))
    #         concatenated_raw.append(a_ds)

    #     return concatenated_raw
    

    @classmethod
    def find_and_merge_for_day_date(cls, sso: SavedSessionsProcessor, search_day_date: datetime,
                                    edf_export_parent_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exported_EDF"),
                                    save_edf: bool=False, save_fif: bool=False) -> mne.io.Raw:
        """ Finds all EEG datasets in the SavedSessionsProcessor for the specified date and merges them into a single Raw dataset.

        Args:
            sso (SavedSessionsProcessor): The SavedSessionsProcessor instance containing the datasets.
            search_day_date (datetime): The date for which to find and merge datasets.

        Returns:
            mne.io.Raw: Merged Raw dataset for the specified date.
        """
        from phopymnehelper.MNE_helpers import up_convert_raw_objects, up_convert_raw_obj

        if "EEG" not in sso.modalities:
            raise ValueError("The SavedSessionsProcessor does not contain any EEG modality data.")

        eeg_modality = sso.modalities["EEG"]
        today_only_eeg_modality = eeg_modality.filtered_by_day_date(search_day_date=search_day_date)

        if not today_only_eeg_modality.datasets:
            raise ValueError(f"No EEG datasets found for the date {search_day_date.date()}.")

        today_only_eeg_modality.datasets = up_convert_raw_objects(today_only_eeg_modality.datasets)
        ## Flatten the EEG sessions into a single dataset for the entire day
        # concatenated_raw = cls.concatenate_datasets(today_only_eeg_modality.datasets)
        concatenated_raw = cls.concatenate_with_gaps(today_only_eeg_modality.datasets)
        concatenated_raw = up_convert_raw_obj(concatenated_raw)

        ## convert to day-specific version:
        if save_fif:
            ## Save out the concatenated raw to a specific folder:
            day_grouped_processed_output_parent_path: Path = sso.eeg_analyzed_parent_export_path.joinpath('dayProcessed')
            day_grouped_processed_output_parent_path.mkdir(parents=True, exist_ok=True)

            ## INPUTS: search_day_date
            curr_day_grouped_output_folder: Path = day_grouped_processed_output_parent_path.joinpath(search_day_date.strftime("%Y-%m-%d"))
            curr_day_grouped_output_folder.mkdir(parents=True, exist_ok=True)
            print(f'curr_day_grouped_output_folder: "{curr_day_grouped_output_folder.as_posix()}"')            

            a_path = Path(concatenated_raw.filenames[0])
            name_parts = a_path.name.split('-', maxsplit=4) # ['20250908', '121104', 'Epoc X', 'raw.fif']
            name_parts[1] = '000000'  # Set time part to '000000'
            new_name: str = '-'.join(name_parts)
            new_path: Path = curr_day_grouped_output_folder.joinpath(new_name)


            # TODO 2025-09-09 22:03: - [ ] IMPORTANT:
            # If Raw is a concatenation of several raw files, **be warned** that only the measurement information from the first raw file is stored. This likely means that certain operations with external tools may not work properly on a saved concatenated file (e.g., probably some or all forms of SSS). It is recommended not to concatenate and then save raw files for this reason.
            # Samples annotated BAD_ACQ_SKIP are not stored in order to optimize memory. Whatever values, they will be loaded as 0s when reading file.        
            concatenated_raw.save(new_path.as_posix(), overwrite=True)
        else:
            print(f'save_fif is False so skipping save.')
            
        ## Save EDF:

        ## INPUTS: raw_eeg
        if save_edf:
            if edf_export_parent_path is None:
                edf_export_parent_path: Path = Path(r"E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/exported_EDF")
                
            edf_export_parent_path.mkdir(exist_ok=True)

            ## Get paths for current raw:
            curr_file_edf_name: str = new_path.with_suffix('.edf').name
            curr_file_edf_path: Path = edf_export_parent_path.joinpath(curr_file_edf_name)
            # EEGData.save_mne_raw_to_edf(concatenated_raw, curr_file_edf_path)
            curr_file_edf_path = concatenated_raw.save_to_edf(output_path=curr_file_edf_path)
        else:
            print(f'save_edf is False so skipping save.')

        return concatenated_raw
    

