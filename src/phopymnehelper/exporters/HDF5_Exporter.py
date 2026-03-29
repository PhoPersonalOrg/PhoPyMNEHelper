from typing import List, Tuple, Optional, Union
import numpy as np
from pathlib import Path

def save_all_to_HDF5(_active_only_out_eeg_raw, _active_all_outputs_dict, hdf5_out_path: Path):
    """
		Usage:
			from phopymnehelper.exporters.HDF5_Exporter import save_all_to_HDF5

			hdf5_out_path: Path = Path('E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/outputs').joinpath('2025-09-22_eegComputations.h5')
			hdf5_out_path
    """

    for idx, (a_raw, a_raw_outputs) in enumerate(zip(_active_only_out_eeg_raw, _active_all_outputs_dict)):
        # a_path: Path = Path(a_raw.filenames[0])
        # basename: str = a_path.stem
        # basename: str = a_raw.info.get('meas_date')
        src_file_path: Path = Path(a_raw.info.get('description'))
        basename: str = src_file_path.stem

        print(f'basename: {basename}')

        for an_output_key, an_output_dict in a_raw_outputs.items():
            for an_output_subkey, an_output_value in an_output_dict.items():
                final_data_key: str = '/'.join([basename, an_output_key, an_output_subkey])
                print(f'\tfinal_data_key: "{final_data_key}"')
                # all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)

        # spectogram_result_dict = a_raw_outputs['spectogram']['spectogram_result_dict']
        # fs = a_raw_outputs['spectogram']['fs']

        # for ch_idx, (a_ch, a_ch_spect_result_tuple) in enumerate(spectogram_result_dict.items()):
        #     all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)
        #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)

        #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)
