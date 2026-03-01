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

from mne.io import read_raw
import pyedflib

from phoofflineeeganalysis.analysis.motion_data import MotionData ## for creating single EDF+ files containing channel with different sampling rates (e.g. EEG and MOTION data)

mne.viz.set_browser_backend("Matplotlib")

from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream

# from phoofflineeeganalysis.EegProcessing import bandpower
from phoofflineeeganalysis.analysis.MNE_helpers import MNEHelpers
# from ..EegProcessing import bandpower
from numpy.typing import NDArray
# from nptyping import NDArray

set_log_level("WARNING")

from phoofflineeeganalysis.helpers.indexing_helpers import reorder_columns_relative

from phopylslhelper.file_metadata_caching.data_file_metadata import DataFileMetadataParser


class HistoricalData:
    """ Methods related to retrospective processing of recorded data
        
    from phoofflineeeganalysis.analysis.historical_data import HistoricalData
        
    """
    modality_channels_dict = {'EEG': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
                            'MOTION': ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'],
                            'GENERIC': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
    }
    
    modality_sfreq_dict = {'EEG': 128, 'MOTION': 16,
                           'GENERIC': 128, 
    }


    # modality_to_num_valid_columns_dict = {'MOTION': 7, 'EEG': 31}
    

    @classmethod
    def get_recording_files(cls, recordings_dir: Union[Path, List[Path]], recordings_extensions = ['.fif']):
        found_recording_files = []
        for ext in recordings_extensions:
            if isinstance(recordings_dir, (List, Tuple)):
                ## iterate through to get the files
                for a_recordings_dir in recordings_dir:
                    found_recording_files.extend(a_recordings_dir.glob(f"*{ext}"))
            else:
                ## single file
                found_recording_files.extend(recordings_dir.glob(f"*{ext}"))
            # found_recording_files.extend(recordings_dir.glob(f"*{ext.upper()}"))
        try:
            found_recording_files.sort(key=lambda f: (-(f.stat().st_mtime), f.name.lower()[::-1]))
        except Exception as e:
            raise
        
        return found_recording_files


    @classmethod
    def extract_datetime_from_filename(cls, filename) -> datetime:
        """Extract recording start datetime from filename by searching for any valid datetime substring.
        Examples: 
            '20250730-195857-Epoc X Motion-raw.fif',
            '20250730-200710-Epoc X-raw.fif',
            '20250618-185519-Epoc X-raw.fif',
            '20250618-185519-Epoc X-raw.fif', # '20250618-185519'
            'eeg_data_2025-08-12T02-56-32.509841.csv',
            'LabRecorder_Apogee_2025-11-04T105347.435Z_eeg.xdf',
        """
        candidates = re.findall(r'\d{4}[-_]?\d{2}[-_]?\d{2}[ T_-]?\d{2}[:\-]?\d{2}[:\-]?\d{2}', filename)
        for cand in candidates:
            # normalized = cand.replace("_", "-").replace(" ", "T").replace("-", "T")
            normalized = cand.replace("_", "T").replace(" ", "T") # .replace("-", "T")
            for fmt in [
                "%Y-%m-%dT%H-%M-%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H%M%S",
                "%Y%m%dT%H%M%S",
                "%Y%m%d_%H%M%S",
                "%Y%m%d-%H%M%S",
                "%Y%m%d%H%M%S"
            ]:
                try:
                    return datetime.strptime(normalized, fmt)
                except ValueError:
                    continue
        raise ValueError(f"Filename '{filename}' does not contain a recognized datetime.")




    @classmethod
    def get_or_parse_datetime_from_raw(cls, raw, override_filepath: Optional[Path]=None, allow_setting_meas_date_from_filename:bool=True) -> datetime:
        """ Get the recording start datetime from the raw.info['meas_date'] or parse from filename if not present
        if allow_setting_meas_date_from_filename is True, it will set the raw.info['meas_date'] if it was None
        """        
        metadata_recording_start_datetime = raw.info.get('meas_date', None)
        if metadata_recording_start_datetime is None:
            if override_filepath is None:
                override_filepath = Path(raw.filenames[0])
            parsed_recording_start_datetime = cls.extract_datetime_from_filename(override_filepath.name)
            metadata_recording_start_datetime = parsed_recording_start_datetime
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
    def MAIN_process_recording_files(cls, eeg_recordings_file_path: Optional[Path] = None, headset_motion_recordings_file_path: Optional[Path] = None, WhisperVideoTranscripts_LSL_Converted: Optional[Path]=None, pho_log_to_LSL_recordings_path: Optional[Path]=None, should_load_data: bool=False):
                  
        """        
            eeg_recordings_file_path: Path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/fif').resolve()
            headset_motion_recordings_file_path: Path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif').resolve()
            WhisperVideoTranscripts_LSL_Converted = Path(r"E:/Dropbox (Personal)/Databases/UnparsedData/WhisperVideoTranscripts_LSL_Converted").resolve()
            
            flat_data_modality_dict, found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(eeg_recordings_file_path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/fif').resolve(),
                            headset_motion_recordings_file_path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif').resolve(),
                            WhisperVideoTranscripts_LSL_Converted = Path(r"E:/Dropbox (Personal)/Databases/UnparsedData/WhisperVideoTranscripts_LSL_Converted").resolve(),
            )
            flat_data_modality_dict
        """
        found_recording_file_modality_dict = {} ## for finding input files

        if eeg_recordings_file_path is not None:
            assert eeg_recordings_file_path.exists()
            found_EEG_recording_files = cls.get_recording_files(recordings_dir=eeg_recordings_file_path)
            found_recording_file_modality_dict.update({'EEG': found_EEG_recording_files})
        
        if headset_motion_recordings_file_path is not None:
            assert headset_motion_recordings_file_path.exists()
            found_MOTION_recording_files = cls.get_recording_files(recordings_dir=headset_motion_recordings_file_path)
            found_recording_file_modality_dict.update({'MOTION': found_MOTION_recording_files})
            
        if WhisperVideoTranscripts_LSL_Converted is not None:
            assert WhisperVideoTranscripts_LSL_Converted.exists()
            found_WHISPER_recording_files = cls.get_recording_files(recordings_dir=WhisperVideoTranscripts_LSL_Converted)
            found_recording_file_modality_dict.update({'WHISPER': found_WHISPER_recording_files})

        if pho_log_to_LSL_recordings_path is not None:
            assert pho_log_to_LSL_recordings_path.exists()
            found_PHO_LOG_TO_LSL_recordings_files = cls.get_recording_files(recordings_dir=pho_log_to_LSL_recordings_path)
            found_recording_file_modality_dict.update({'PHO_LOG_TO_LSL': found_PHO_LOG_TO_LSL_recordings_files})

        ## Load and process the found recording file types:
        flat_data_modality_dict = cls.read_recording_files(found_recording_file_modality_dict=found_recording_file_modality_dict, should_load_data=should_load_data)

        return flat_data_modality_dict, found_recording_file_modality_dict


    @classmethod
    def read_exported_fif_files(cls, found_recording_files: List[Path], allow_setting_meas_date_from_filename: bool=True, constrain_channels: Optional[List[str]]=None, include_src_file_column: bool=True, file_type: str='MISC', should_load_data: bool=False):
        """
        # found_recording_files = deepcopy(found_EEG_recording_files)
                found_recording_files = deepcopy(found_MOTION_recording_files)
          HistoricalData.read_exported_fif_files(found_recording_files=found_recording_files)

        """
        all_data = []
        all_times = []
        datasets = []
        n_records = []
        src_file = []
        expected_channels = cls.modality_channels_dict.get(file_type, None)
        
        for a_recording_file in found_recording_files:
            assert a_recording_file.exists()
            try:            
                raw = read_raw(a_recording_file, preload=False)
                meas_datetime = cls.get_or_parse_datetime_from_raw(raw, allow_setting_meas_date_from_filename=allow_setting_meas_date_from_filename)
                if should_load_data:
                    # raw.info.ch_names
                    # raw.to_data_frame()
                    data, times = raw.get_data(picks=constrain_channels, return_times=True)

                    # Convert relative times to absolute timestamps (in seconds since epoch)
                    start_time = meas_datetime.timestamp() if hasattr(meas_datetime, 'timestamp') else meas_datetime[0]
                    abs_times = start_time + times
                    a_n_records: int = len(times) ## n_records per src file
                    all_data.append(data)
                    all_times.append(abs_times)
                    n_records.append(a_n_records) ## n_records per src file
                    if include_src_file_column:
                        src_file.extend(([a_recording_file.stem] * a_n_records))            
                # END if should_load_data
                ## Append to datasets
                datasets.append(raw)
                                    
            except (ValueError, AttributeError, TypeError) as e:
                print(f'Encountered error: {e} while trying to read CSV file {a_recording_file}. Skipping')
                # datasets.append(None)
                pass
            except Exception as e:
                raise
        
        ## END `for a_recording_file in found_recording_files`
        
        if should_load_data:
            all_data_shapes = [np.shape(v) for v in all_data]
            all_data_ch_names = [raw.info.ch_names for raw in datasets]
            full_data = np.concatenate(all_data, axis=1)  # concatenate over time
            full_times = np.concatenate(all_times)

            if constrain_channels is None:
                constrain_channels = all_data_ch_names[0] ## take the first list of channel names (all of them) as the channels for the data. They better be the same!
            ## Hopefully same size as number of channels!
            df = pd.DataFrame(full_data.T, columns=constrain_channels)
            df['timestamp'] = full_times
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s') ## add datetime column
            if include_src_file_column:
                assert len(src_file) == len(df), f"len(src_file): {len(src_file)}, len(df): {len(df)}"
                df['src_file'] = src_file
            # Sort by column: 'timestamp_dt' (ascending)
            df = df.sort_values(['timestamp_dt'], na_position='first').reset_index(drop=True)
            
            if file_type == 'MOTION':
                ## Estimate Quaternions from motion data:
                df = MotionData.compute_quaternions(df)
            elif file_type == 'EEG':
                pass
            else:
                pass            

            # df.to_csv('eeg_concatenated.csv', index=False)
        else:
            df = None
        
        return (all_data, all_times), datasets, df
    

    @classmethod
    def read_exported_csv_files(cls, found_recording_files: List[Path], allow_setting_meas_date_from_filename: bool=True, constrain_channels: Optional[List[str]]=None, include_src_file_column: bool=True, file_type: str='MISC',
             should_load_data: bool=False, debug_n_max_files_to_load: int = 5):
        """ Reads .csv exported by the Flutter BLE iOS/Android app in the ~2025-09-09 format
        
        # found_recording_files = deepcopy(found_EEG_recording_files)
                found_recording_files = deepcopy(found_MOTION_recording_files)
            HistoricalData.read_exported_fif_files(found_recording_files=found_recording_files)

        """
        from phoofflineeeganalysis.analysis.MNE_helpers import DatasetDatetimeBoundsRenderingMixin, RawArrayExtended, RawExtended, up_convert_raw_objects, up_convert_raw_obj
        
        # import csv
        
        # def _subfn_load_eeg_bytes(path):
        #     with open(path, newline="") as f:
        #         reader = csv.reader(f)
        #         for row in reader:
        #             if not row: 
        #                 continue
        #             # skip timestamp (row[0])
        #             timestamp = int(row[0])
        #             raw_bytes = bytes(int(x.strip('\n')) for x in row[1:])
        #             yield (timestamp, raw_bytes)


        # all_data = []
        # all_times = []
        # datasets = []
        # n_records = []
        # src_file = []
        
        all_data = {}
        all_times = {}
        datasets = {}
        n_records = {}
        src_file = {}

        # expected_channels = cls.modality_channels_dict.get(file_type, None)
        


        # def _subfn_build_RawEEG_from_df(raw_df: pd.DataFrame, a_recording_file: Path, a_file_type='EEG'):
        def _subfn_build_RawFile_from_df(raw_df: pd.DataFrame, a_recording_file: Path, a_file_type='EEG'):
            """ 
            captures: cls, should_load_data, include_src_file_column
            captures and updates: all_data, all_times, datasets, n_records, src_file, expected_channels

            (data, times), raw, meas_datetime = _subfn_build_RawEEG_from_df(raw_df=raw_df, a_recording_file=a_recording_file)
            
            a_recording_file: USED FOR PARSING DATETIME FROM FILENAME AND A FEW OTHER THINGS!
            
            """
            curr_expected_channels = cls.modality_channels_dict.get(a_file_type, None)

            # ch_names = raw_df.columns.to_list()[1:]
            ch_names = raw_df.columns.to_list() ## Includes timestamp
            # if np.isin(ch_names, expected_channels).all():
            found_good_channels = np.array(curr_expected_channels)[np.isin(curr_expected_channels, ch_names)]
            if len(found_good_channels) == 0:
                raise ValueError(f'loaded CSV has the wrong channel names: ch_names: {ch_names}, curr_expected_channels: {curr_expected_channels}')
            sfreq = cls.modality_sfreq_dict['EEG']
            info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types=(['misc'] + ['eeg'] * (len(ch_names)-1)))
            # info.set_channel_types(mapping={ch: 'eeg' for ch in ch_names if ch != 'timestamp'})

            raw = mne.io.RawArray(raw_df.T, info)
            # raw.set_filename(a_recording_file.as_posix())
            meas_datetime = cls.get_or_parse_datetime_from_raw(raw, override_filepath=a_recording_file, allow_setting_meas_date_from_filename=allow_setting_meas_date_from_filename)

            if should_load_data:
                # raw.info.ch_names
                # raw.to_data_frame()
                data, times = raw.get_data(picks=constrain_channels, return_times=True)

                # Convert relative times to absolute timestamps (in seconds since epoch)
                start_time = meas_datetime.timestamp() if hasattr(meas_datetime, 'timestamp') else meas_datetime[0]
                abs_times = start_time + times
                a_n_records: int = len(times) ## n_records per src file
                
                if a_file_type not in all_data:
                    all_data[a_file_type] = []
                    
                if a_file_type not in all_times:
                    all_times[a_file_type] = []

                if a_file_type not in n_records:
                    n_records[a_file_type] = []

                all_data[a_file_type].append(data)
                all_times[a_file_type].append(abs_times)
                n_records[a_file_type].append(a_n_records) ## n_records per src file
                if include_src_file_column:
                    if a_file_type not in src_file:
                        src_file[a_file_type] = []
                    src_file[a_file_type].extend(([a_recording_file.stem] * a_n_records))
            else:
                # data = raw_df[[c for c in raw_df.columns if c not in ('timestamp')]].to_numpy()
                # times = raw_df['timestamp'].to_numpy()                
                data = None
                times = None
                
            # END if should_load_data
            
            ## Append to datasets
            if a_file_type not in datasets:
                datasets[a_file_type] = []
                
            if len(raw.times) == 0:
                raise ValueError(f"len(raw.times) == 0!! {len(raw.times) == 0}")
            else:
                up_convert_raw_obj(raw)
                
                datasets[a_file_type].append(raw)

            return (data, times), raw, meas_datetime


        # ==================================================================================================================================================================================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #


        ## the two lengths are (65, 85)
        if (file_type == 'GENERIC'):
            print(f'GENERIC FILE PROCESSING -- found {len(found_recording_files)} files, loading only {(debug_n_max_files_to_load or len(found_recording_files))}: ')

        else:
            ## standard type (exported/decoded EEG or MOTION file output)
            print(f'file_type: {file_type} PROCESSING -- found {len(found_recording_files)} files, loading only {(debug_n_max_files_to_load or len(found_recording_files))}: ')
            
            assert file_type in ['EEG', 'MOTION'], f"file_type: {file_type} was no in the expected list!"
            data_channel_names = deepcopy(cls.modality_channels_dict[file_type])
            all_channel_names = ['timestamp'] + data_channel_names
            required_num_columns: int = len(all_channel_names) # len(channel_names) + 1 # +1 for the timestamp column
            required_channel_index_map = {ch_name:i for i, ch_name in enumerate(all_channel_names)} ## like {'timestamp': 0}

            ## These above properties are only used in the standard type mode



        for a_file_idx, a_recording_file in enumerate(found_recording_files):
            if (debug_n_max_files_to_load is not None) and ((a_file_idx+1) > debug_n_max_files_to_load):
                continue ## stop processing the rest

            if a_recording_file.exists():
                print(f'FILE[{a_file_idx}]: {a_recording_file.stem} | file_type: "{file_type}" PROCESSING:')
                try:
                    ## the two lengths are (65, 85)
                    if (file_type == 'GENERIC'):
                        # ## GENERIC TYPE:
                        # ## Process line-by-line to handle the variable line lengths
                        modality_out_dfs_dict, modality_out_linedata_df_dict = GenericRawDebugFileProcessor.main_perform_decode(a_recording_file=a_recording_file)
                        for a_modality_name, (final_modality_decoded_df, _remaineder_decoded_df) in modality_out_dfs_dict.items():
                            # final_eeg_decoded_df, _remaineder_decoded_df = modality_out_dfs_dict['EEG']
                            (data, times), raw, meas_datetime = _subfn_build_RawFile_from_df(raw_df=final_modality_decoded_df, a_recording_file=a_recording_file, a_file_type=a_modality_name)
                        
                        ## #TODO 2025-09-18 10:27: - [ ] we would add MOTION packets here

                    else:
                        ## standard type (exported/decoded EEG or MOTION file output)
                        raw_df: pd.DataFrame = pd.read_csv(a_recording_file, low_memory=False, skiprows=0, header=0) # , parse_dates=['timestamp']

                        # Loaded variable 'decoded_packets_list' from kernel state
                        raw_df['timestamp'] = pd.to_numeric(raw_df['timestamp'], errors='coerce') # .astype('int64') # Use non-nullable integer type
                        
                        for col in data_channel_names:
                            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

                        raw_df = raw_df.drop_duplicates() # Drop duplicate rows across all columns (importantly including 'timestamp')
                        raw_df = raw_df.sort_values(['timestamp']) # Sort by column: 'timestamp' (ascending)

                        ## Find rows with any missing elmeents:
                        df_is_na = raw_df.isna()
                        # num_missing_elements = df_is_na.sum(axis='columns')
                        num_valid_columns = np.logical_not(df_is_na).sum(axis='columns')

                        valid_eeg_row_record_indicies = (num_valid_columns == required_num_columns)
                        ## Just the EEG rows with the right number of columns:
                        final_eeg_decoded_df: pd.DataFrame = raw_df[valid_eeg_row_record_indicies]

                        if not np.all(list(final_eeg_decoded_df.columns) != all_channel_names):
                            print(f'WARN: reordering columns!')
                            ## Move timestamp column to the end:
                            final_eeg_decoded_df = reorder_columns_relative(final_eeg_decoded_df, column_names=required_channel_index_map, relative_mode='start')


                        (data, times), raw, meas_datetime = _subfn_build_RawFile_from_df(raw_df=final_eeg_decoded_df, a_recording_file=a_recording_file, a_file_type=file_type)
                        
                except (ValueError, AttributeError, TypeError) as e:
                    print(f'\tEncountered error: {e} while trying to read CSV file {a_recording_file}. Skipping')
                    # datasets.append(None)
                    pass
                except Exception as e:
                    raise
            ## END if a_recording_file.exists()
            
        ## END `for a_file_idx, a_recording_file in enu...`

        modality_dfs = {}

        if should_load_data:            
            for a_modality_name, a_datasets in datasets.items():
                if len(a_datasets) == 0:
                    df = None
                else:
                    # all_data_shapes = [np.shape(v) for v in all_data]
                    all_data_ch_names = [raw.info.ch_names for raw in a_datasets]

                    full_data = np.concatenate(all_data.get(a_modality_name, []), axis=1)  # concatenate over time
                    full_times = np.concatenate(all_times.get(a_modality_name, []))

                    if constrain_channels is None:
                        constrain_channels = all_data_ch_names[0] ## take the first list of channel names (all of them) as the channels for the data. They better be the same!
                    ## Hopefully same size as number of channels!
                    df = pd.DataFrame(full_data.T, columns=constrain_channels)
                    df['timestamp'] = full_times
                    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s') ## add datetime column
                    if include_src_file_column:
                        assert len(src_file.get(a_modality_name, [])) == len(df), f"len(src_file): {len(src_file.get(a_modality_name, []))}, len(df): {len(df)}"
                        df['src_file'] = src_file.get(a_modality_name, [])
                    # Sort by column: 'timestamp_dt' (ascending)
                    df = df.sort_values(['timestamp_dt'], na_position='first').reset_index(drop=True)
                    # if file_type == 'MOTION':
                    #     ## Estimate Quaternions from motion data:
                    #     df = MotionData.compute_quaternions(df)
                    # elif file_type == 'EEG':
                    #     pass
                    # else:
                    #     pass            

                modality_dfs[a_modality_name] = df
            # df.to_csv('eeg_concatenated.csv', index=False)
        else:
            df = None


        return datasets, modality_dfs

        # return (all_data, all_times), datasets, df
        

    @classmethod
    def read_recording_files(cls, found_recording_file_modality_dict: Dict[str, List[Path]], should_load_data: bool=False):
        """ Load and process the found recording file types
            eeg_recordings_file_path: Path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/fif').resolve()
            headset_motion_recordings_file_path: Path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif').resolve()
            WhisperVideoTranscripts_LSL_Converted = Path(r"E:/Dropbox (Personal)/Databases/UnparsedData/WhisperVideoTranscripts_LSL_Converted").resolve()

            flat_data_modality_dict, found_recording_file_modality_dict = HistoricalData.MAIN_process_recording_files(eeg_recordings_file_path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/fif').resolve(),
                            headset_motion_recordings_file_path = Path(r'E:/Dropbox (Personal)/Databases/UnparsedData/EmotivEpocX_EEGRecordings/MOTION_RECORDINGS/fif').resolve(),
                            WhisperVideoTranscripts_LSL_Converted = Path(r"E:/Dropbox (Personal)/Databases/UnparsedData/WhisperVideoTranscripts_LSL_Converted").resolve(),
            )
            flat_data_modality_dict
        """
        flat_data_modality_dict = {} ## For collecting outputs 
        ## Load and process the found recording file types:
        for k, a_found_recording_files in found_recording_file_modality_dict.items():
            flat_data_modality_dict[k] = cls.read_exported_fif_files(found_recording_files=a_found_recording_files, constrain_channels=cls.modality_channels_dict.get(k, None), file_type=k, should_load_data=should_load_data)
            # (all_data, all_times), datasets, df = flat_data_modality_dict[k] ## Unpacking
        return flat_data_modality_dict


    @classmethod
    def add_bad_periods_from_MOTION_data(cls,
            active_EEG_IDXs, datasets_EEG,
            active_motion_IDXs, datasets_MOTION, analysis_results_MOTION,
            preprocessed_EEG_save_path: Optional[Path]=None, debug_print: bool = False,
        ):
        """ Find periods that overlap the motion data
        
        from phoofflineeeganalysis.analysis.EEG_data import EEGData
        from phoofflineeeganalysis.analysis.motion_data import MotionData
        from phoofflineeeganalysis.analysis.historical_data import HistoricalData
        
        n_most_recent_sessions_to_preprocess: int = 10
        
        ## BEGIN ANALYSIS of EEG Data
        preprocessed_EEG_save_path: Path = eeg_analyzed_parent_export_path.joinpath('preprocessed_EEG').resolve()
        preprocessed_EEG_save_path.mkdir(exist_ok=True)

        ## INPUTS: flat_data_modality_dict
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = flat_data_modality_dict['EEG']  ## Unpacking

        active_EEG_IDXs, analysis_results_EEG = EEGData.preprocess(datasets_EEG=datasets_EEG, preprocessed_EEG_save_path=preprocessed_EEG_save_path, n_most_recent_sessions_to_preprocess=n_most_recent_sessions_to_preprocess)
        
        (all_data_MOTION, all_times_MOTION), datasets_MOTION, df_MOTION = flat_data_modality_dict['MOTION']  ## Unpacking

        (active_motion_IDXs, analysis_results_MOTION) = MotionData.preprocess(datasets_MOTION, n_most_recent_sessions_to_preprocess=n_most_recent_sessions_to_preprocess)

        dataset_MOTION_df, dataset_EEG_df = HistoricalData.add_bad_periods_from_MOTION_data(active_EEG_IDXs=active_EEG_IDXs, datasets_EEG=datasets_EEG,
                                                        active_motion_IDXs=active_motion_IDXs, datasets_MOTION=datasets_MOTION, analysis_results_MOTION=analysis_results_MOTION)

        """
        active_datasets_MOTION = [datasets_MOTION[i] for i in active_motion_IDXs]
        
        ## Find periods that overlap the motion data:
        ## INPUTS: active_motion_IDXs, active_datasets_MOTION
        dataset_MOTION_df = []
        # analysis_results_EEG
        for i, a_ds in enumerate(active_datasets_MOTION):
            ## Find all motion datasets that overlap:
            motion_dataset_IDX: int = active_motion_IDXs[i]
            # a_ds_EEG.last_samp
            abs_start_t = a_ds.times[0] + a_ds.info['meas_date'].timestamp()
            abs_end_t = a_ds.times[-1] + a_ds.info['meas_date'].timestamp()
            dataset_MOTION_df.append({'motion_dataset_IDX': motion_dataset_IDX, 'start_time': abs_start_t, 'end_time': abs_end_t})
            # display(abs_start_t, abs_end_t)

        dataset_MOTION_df = pd.DataFrame(dataset_MOTION_df)	

        ## Find periods that overlap the motion data:
        ## INPUTS: analysis_results_MOTION
        dataset_EEG_df = []
        # analysis_results_EEG
        # for i, a_ds_EEG in enumerate(datasets_EEG):
        for i, a_EEG_IDX in enumerate(active_EEG_IDXs):
            ## Find all motion datasets that overlap:
            a_ds_EEG = datasets_EEG[a_EEG_IDX]
            # datasets_MOTION
            abs_start_t = a_ds_EEG.times[0] + a_ds_EEG.info['meas_date'].timestamp()
            abs_end_t = a_ds_EEG.times[-1] + a_ds_EEG.info['meas_date'].timestamp()

            ## Find all motion datasets that overlap:
            mask = (abs_start_t <= dataset_MOTION_df['end_time']) & (dataset_MOTION_df['start_time'] <= abs_end_t) ## doesn't this mask miss cases where the a motion data starts or ends outside the EEG data?
            # mask = np.logical_or((abs_start_t <= dataset_MOTION_df['end_time']), (dataset_MOTION_df['start_time'] <= abs_end_t))
            df_MOTION_overlaps: pd.DataFrame = dataset_MOTION_df[mask].copy()
            # an_overlapping_motion_IDXs = df_MOTION_overlaps['motion_dataset_IDX'].to_list()
            an_overlapping_motion_IDXs = df_MOTION_overlaps.index.to_list()
            if debug_print:
                print(f'i: {i}, an_overlapping_motion_IDXs: {an_overlapping_motion_IDXs}')
            combined_annotations = None
            # Use the first dataset's orig_time as reference
            base_orig_time = None
            for an_overlapping_motion_IDX in an_overlapping_motion_IDXs: 
                if debug_print:
                    print(f'\tan_overlapping_motion_IDX: {an_overlapping_motion_IDX}')
                # motion_ds = datasets_MOTION[an_overlapping_motion_IDX]
                a_bad_periods_annot = analysis_results_MOTION[an_overlapping_motion_IDX]['bad_periods_annotations']['high_accel']
                if combined_annotations is None:
                    combined_annotations = a_bad_periods_annot.copy() ## first annotation
                    base_orig_time = combined_annotations.orig_time
                else:
                    # assert base_orig_time is not None
                    a_bad_periods_annot._orig_time = base_orig_time
                    combined_annotations += a_bad_periods_annot
            ## END for an_overlapping_motion_IDX in an_overlapping_motion_IDXs...
            ## Set the `combined_annotations` to the EEG dataset:
            if combined_annotations is not None:
                a_ds_EEG.set_annotations(combined_annotations)      

            if preprocessed_EEG_save_path is not None:
                if not preprocessed_EEG_save_path.exists():
                    preprocessed_EEG_save_path.mkdir(parents=True, exist_ok=True)
                a_raw_savepath: Path = preprocessed_EEG_save_path.joinpath(Path(a_ds_EEG.filenames[0]).name).resolve()
                print(f'saving to {a_raw_savepath}...', end='\t')
                a_ds_EEG.save(a_raw_savepath, overwrite=True)
                print(f'\tdone.')
            dataset_EEG_df.append({'dataset_IDX': a_EEG_IDX, 'start_time': abs_start_t, 'end_time': abs_end_t, 'motion_idxs': an_overlapping_motion_IDXs})
            # display(abs_start_t, abs_end_t)

        dataset_EEG_df = pd.DataFrame(dataset_EEG_df)	
    
        ## convert columns to datetime:    
        dataset_EEG_df = MNEHelpers.convert_df_columns_to_datetime(dataset_EEG_df, dt_col_names=["start_time", "end_time"])
        dataset_MOTION_df = MNEHelpers.convert_df_columns_to_datetime(dataset_MOTION_df, dt_col_names=["start_time", "end_time"])

        return (dataset_MOTION_df, dataset_EEG_df)


    @classmethod
    def add_additional_LOGGING_annotations(cls,
            active_EEG_IDXs, datasets_EEG,
            active_LOGGING_IDXs, datasets_LOGGING, analysis_results_LOGGING, logging_series_identifier: str = 'PHO_LOG', # ['PHO_LOG', 'WHISPER'],
            preprocessed_EEG_save_path: Optional[Path]=None, debug_print: bool = False,
        ):
        """ Find periods that overlap the motion data

        from phoofflineeeganalysis.analysis.EEG_data import EEGData
        from phoofflineeeganalysis.analysis.motion_data import MotionData
        from phoofflineeeganalysis.analysis.historical_data import HistoricalData

        n_most_recent_sessions_to_preprocess: int = 10

        ## BEGIN ANALYSIS of EEG Data
        preprocessed_EEG_save_path: Path = eeg_analyzed_parent_export_path.joinpath('preprocessed_EEG').resolve()
        preprocessed_EEG_save_path.mkdir(exist_ok=True)

        ## INPUTS: flat_data_modality_dict
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = flat_data_modality_dict['EEG']  ## Unpacking

        active_EEG_IDXs, analysis_results_EEG = EEGData.preprocess(datasets_EEG=datasets_EEG, preprocessed_EEG_save_path=preprocessed_EEG_save_path, n_most_recent_sessions_to_preprocess=n_most_recent_sessions_to_preprocess)

        (all_data_MOTION, all_times_MOTION), datasets_MOTION, df_MOTION = flat_data_modality_dict['MOTION']  ## Unpacking

        (active_motion_IDXs, analysis_results_MOTION) = MotionData.preprocess(datasets_MOTION, n_most_recent_sessions_to_preprocess=n_most_recent_sessions_to_preprocess)

        dataset_LOGGING_df, dataset_EEG_df = HistoricalData.add_additional_LOGGING_annotations(active_EEG_IDXs=active_EEG_IDXs, datasets_EEG=datasets_EEG,
                                                        active_LOGGING_IDXs=active_LOGGING_IDXs, datasets_LOGGING=datasets_MOTION, analysis_results_LOGGING=analysis_results_LOGGING, logging_series_identifier='PHO_LOG',
                                                        preprocessed_EEG_save_path=None)

        """
        active_datasets_LOGGING = [datasets_LOGGING[i] for i in active_LOGGING_IDXs]
        curr_logging_datset_index_col_name: str = f"{logging_series_identifier}_dataset_IDX"
        curr_logging_dataset_matching_indicies_col_name: str = f'{logging_series_identifier}_idxs' ## the column added to the EEG dataframe
        

        ## Find periods that overlap the motion data:
        ## INPUTS: active_motion_IDXs, active_datasets_MOTION
        dataset_LOGGING_df = []
        # analysis_results_EEG
        for i, a_ds in enumerate(active_datasets_LOGGING):
            ## Find all motion datasets that overlap:
            curr_dataset_IDX: int = active_LOGGING_IDXs[i]
            # a_ds_EEG.last_samp
            abs_start_t = a_ds.times[0] + a_ds.info['meas_date'].timestamp()
            abs_end_t = a_ds.times[-1] + a_ds.info['meas_date'].timestamp()
            # dataset_LOGGING_df.append({'motion_dataset_IDX': motion_dataset_IDX, 'start_time': abs_start_t, 'end_time': abs_end_t})
            dataset_LOGGING_df.append({'dataset_IDX': curr_dataset_IDX, curr_logging_datset_index_col_name: curr_dataset_IDX, 'start_time': abs_start_t, 'end_time': abs_end_t}) # 'motion_dataset_IDX': curr_dataset_IDX
            # display(abs_start_t, abs_end_t)
        ## END for i, a_ds in enumerate(active_datasets_LOGGING)...
        
        dataset_LOGGING_df = pd.DataFrame(dataset_LOGGING_df)	

        ## Find periods that overlap the motion data:
        ## INPUTS: analysis_results_MOTION
        dataset_EEG_df = []
        # analysis_results_EEG
        # for i, a_ds_EEG in enumerate(datasets_EEG):
        for i, a_EEG_IDX in enumerate(active_EEG_IDXs):
            ## Find all motion datasets that overlap:
            a_ds_EEG = datasets_EEG[a_EEG_IDX]
            # datasets_MOTION
            abs_start_t = a_ds_EEG.times[0] + a_ds_EEG.info['meas_date'].timestamp()
            abs_end_t = a_ds_EEG.times[-1] + a_ds_EEG.info['meas_date'].timestamp()

            ## Find all motion datasets that overlap:
            mask = (abs_start_t <= dataset_LOGGING_df['end_time']) & (dataset_LOGGING_df['start_time'] <= abs_end_t) ## doesn't this mask miss cases where the a motion data starts or ends outside the EEG data?
            # mask = np.logical_or((abs_start_t <= dataset_MOTION_df['end_time']), (dataset_MOTION_df['start_time'] <= abs_end_t))
            df_LOGGING_overlaps: pd.DataFrame = dataset_LOGGING_df[mask].copy()
            # an_overlapping_motion_IDXs = df_MOTION_overlaps['motion_dataset_IDX'].to_list()
            an_overlapping_logging_IDXs = df_LOGGING_overlaps.index.to_list()
            if debug_print:
                print(f'i: {i}, an_overlapping_motion_IDXs: {an_overlapping_logging_IDXs}')
            # combined_annotations = None
            for an_overlapping_logging_IDX in an_overlapping_logging_IDXs: 
                if debug_print:
                    print(f'\tan_overlapping_motion_IDX: {an_overlapping_logging_IDX}')
                # motion_ds = datasets_MOTION[an_overlapping_motion_IDX]
                curr_annot = active_datasets_LOGGING[an_overlapping_logging_IDX].annotations ## should this be into the active datasets of all? ANSWER: active_datasets_LOGGING
                if curr_annot is not None:
                    a_ds_EEG = MNEHelpers.merge_annotations(a_ds_EEG, curr_annot) ## why does this use `MNEHelpers.merge_annotations(...)`?
                    
                    # if combined_annotations is None:
                    #     combined_annotations = curr_annot.copy()                                                                                                
                    # else:
                    #     combined_annotations += curr_annot
                        
            ## END for an_overlapping_motion_IDX in an_overlapping_motion_IDXs...
            ## Set the `combined_annotations` to the EEG dataset:
            # if combined_annotations is not None:
            #     # a_ds_EEG.set_annotations(combined_annotations) ## overwrite existing annotations
            #     a_ds_EEG.set_annotations(a_ds_EEG.annotations + combined_annotations)

            if preprocessed_EEG_save_path is not None:
                if not preprocessed_EEG_save_path.exists():
                    preprocessed_EEG_save_path.mkdir(parents=True, exist_ok=True)
                a_raw_savepath: Path = preprocessed_EEG_save_path.joinpath(a_ds_EEG.filenames[0].name).resolve()
                print(f'saving to {a_raw_savepath}...', end='\t')
                a_ds_EEG.save(a_raw_savepath, overwrite=True)
                print(f'\tdone.')                
            ## END if preprocessed_EEG_save_path is not None...
            
            dataset_EEG_df.append({'dataset_IDX': a_EEG_IDX, 'start_time': abs_start_t, 'end_time': abs_end_t, curr_logging_dataset_matching_indicies_col_name: an_overlapping_logging_IDXs})
            # display(abs_start_t, abs_end_t)
        ## END for i, a_EEG_IDX in enumerate(active_EEG_IDXs)...
        
        dataset_EEG_df = pd.DataFrame(dataset_EEG_df)	

        ## convert columns to datetime:    
        dataset_EEG_df = MNEHelpers.convert_df_columns_to_datetime(dataset_EEG_df, dt_col_names=["start_time", "end_time"])
        dataset_LOGGING_df = MNEHelpers.convert_df_columns_to_datetime(dataset_LOGGING_df, dt_col_names=["start_time", "end_time"])

        return (dataset_LOGGING_df, dataset_EEG_df)


    @classmethod
    def build_file_comparison_df(cls, recording_files: List[Path], max_workers: int = 3) -> pd.DataFrame:
        """ returns a dataframe with info about each file such as their modification time, etc
        
        Processes files in parallel using ThreadPoolExecutor for improved performance.
        Uses disk-persisted cache via DataFileMetadataParser to speed up subsequent runs.
        
        Args:
            recording_files: List of Path objects to recording files
            max_workers: Maximum number of parallel threads (default: 3)
        
        Usage:
        
            from phoofflineeeganalysis.analysis.historical_data import HistoricalData

            pre_processed_EEG_recording_files = HistoricalData.get_recording_files(recordings_dir=sso.eeg_analyzed_parent_export_path)
            pre_processed_EEG_recording_file_df: pd.DataFrame = HistoricalData.build_file_comparison_df(recording_files=pre_processed_EEG_recording_files)
            pre_processed_EEG_recording_file_df

            ## OUTPUTS: pre_processed_EEG_recording_file_df, pre_processed_EEG_recording_files
            
            
            modern_found_EEG_recording_files = HistoricalData.get_recording_files(recordings_dir=sso.eeg_recordings_file_path)
            modern_found_EEG_recording_file_df: pd.DataFrame = HistoricalData.build_file_comparison_df(recording_files=modern_found_EEG_recording_files)
            modern_found_EEG_recording_file_df

            ## OUTPUTS: modern_found_EEG_recording_file_df, modern_found_EEG_recording_files
            
            
            
            
        """
        if not recording_files:
            return pd.DataFrame()
        
        # Determine cache path (use first file's parent directory)
        cache_path = recording_files[0].parent / "_data_file_metadata_cache.csv"
        
        # Use DataFileMetadataParser for cache-backed processing
        df = DataFileMetadataParser.build_file_comparison_df_cached(
            recording_files=recording_files,
            cache_path=cache_path,
            max_workers=max_workers,
            use_cache=True,
            force_rebuild=False
        )
        
        # The DataFileMetadataParser already returns the correct format with:
        # - src_file, src_file_name, start_datetime, start_t, meas_datetime, file_size, size, ctime, mtime
        # All columns are already in the correct format, so we can return it directly
        return df
    

    @classmethod
    def discover_updated_recording_files(cls, eeg_recordings_file_path: Path, eeg_analyzed_parent_export_path: Path=None, recordings_extensions = ['.fif']):
        """ discover recording files that have been updated since the last run of the script
        
        updated_file_paths, (pending_updated_recording_file_df, modern_found_EEG_recording_file_df, pre_processed_EEG_recording_file_df) = HistoricalData.discover_updated_recording_files(eeg_recordings_file_path=sso.eeg_recordings_file_path, eeg_analyzed_parent_export_path=sso.eeg_analyzed_parent_export_path)
        """
        if eeg_analyzed_parent_export_path is not None:
            pre_processed_EEG_recording_files = cls.get_recording_files(recordings_dir=eeg_analyzed_parent_export_path, recordings_extensions = recordings_extensions)
            pre_processed_EEG_recording_file_df: pd.DataFrame = cls.build_file_comparison_df(recording_files=pre_processed_EEG_recording_files)
        else:
            pre_processed_EEG_recording_file_df = None

        modern_found_EEG_recording_files = cls.get_recording_files(recordings_dir=eeg_recordings_file_path)
        modern_found_EEG_recording_file_df: pd.DataFrame = cls.build_file_comparison_df(recording_files=modern_found_EEG_recording_files)
        
        file_identifier_col_name: str = 'src_file_name'
        # file_identifier_col_name: str = 'src_file'
        modern_files = set(modern_found_EEG_recording_file_df[file_identifier_col_name].values).difference()
        prev_files = set(pre_processed_EEG_recording_file_df[file_identifier_col_name].values)

        modern_only_files = modern_files.difference(prev_files)

        prev_only_files = prev_files.difference(modern_files)
        # prev_only_files
        print(f'prev_only_files: {prev_only_files}')
        
        pending_updated_recording_file_df = deepcopy(modern_found_EEG_recording_file_df).join(deepcopy(pre_processed_EEG_recording_file_df).set_index(file_identifier_col_name), on=file_identifier_col_name, how='left', rsuffix='_prev')
        was_updated = (pending_updated_recording_file_df['size'] > pending_updated_recording_file_df['size_prev']) ## size should be strictly less than the prev_size, which the prev size has a bunch of other data.
        was_updated = np.logical_or(was_updated, (pending_updated_recording_file_df['mtime'] < pending_updated_recording_file_df['mtime_prev'])) # < means modification time is NEWER than the 'mtime_prev'
        was_updated = np.logical_or(was_updated, pending_updated_recording_file_df['src_file_prev'].isna())
        pending_updated_recording_file_df['was_updated'] = deepcopy(was_updated)
        
        # pending_updated_recording_file_df = pending_updated_recording_file_df[was_updated] ## get only the updated files
        pending_updated_only_recording_file_df = deepcopy(pending_updated_recording_file_df[was_updated])
        
        # updated_file_paths = [Path(v) for v in pending_updated_recording_file_df['src_file'].values.tolist()]
        updated_file_paths = [Path(v) for v in pending_updated_only_recording_file_df['src_file'].values.tolist()]
        
        return updated_file_paths, (pending_updated_recording_file_df, modern_found_EEG_recording_file_df, pre_processed_EEG_recording_file_df)
        
        

        

# datasets = []
# mne.viz.set_browser_backend("Matplotlib")

# data = read_raw("E:\Dropbox (Personal)\Databases\UnparsedData\EmotivEpocX_EEGRecordings\fif\20250808-052237-Epoc X-raw.fif", preload=True)
# datasets.insert(0, data)
# data = read_raw("E:\Dropbox (Personal)\Databases\UnparsedData\EmotivEpocX_EEGRecordings\fif\20250808-015634-Epoc X-raw.fif", preload=True)
# datasets.insert(1, data)
# data.plot(events=events, n_channels=15)
# datasets.insert(2, deepcopy(data))
# data = datasets[2]
# mne.concatenate_raws(data, datasets[0])
