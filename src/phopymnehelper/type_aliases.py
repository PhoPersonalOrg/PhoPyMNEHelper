from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Literal
from typing_extensions import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
from typing import NewType

""" Usage:

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TypeVar
import phopymnehelper.type_aliases as types

e.g. Dict[types.FilterContextName, Dict[types.ComputationFunctionName: CapturedException]]
"""

""" from `neuropy.utils.type_aliases`
	aclu_index: TypeAlias = int # an integer index that is an aclu
	DecoderName = NewType('DecoderName', str)
"""
# FilterContextName: TypeAlias = str # a string identifier of a specific filtering context -- 'maze1_odd', 'maze2_odd', 'maze_odd' -- used in `.get_failed_computations(...)`
# ComputationFunctionName: TypeAlias = str # a string identifier of a computation function -- can be either the long or short name -- '_split_to_directional_laps' -- used in `.get_failed_computations(...)`


xdf_file_name: TypeAlias = str # a name of the xdf file corresponding to a given session
EEGComputationId: TypeAlias = str # a string identifier of a computation function --Literal["time_independent_bad_channels", "bad_epochs", "raw_data_topo", "cwt", "spectogram"]


# KnownNamedDecoderTrainedComputeEpochsType = Literal['laps', 'non_pbe'] # trained_compute_epochs
# KnownNamedDecodingEpochsType = Literal['laps', 'replay', 'ripple', 'pbe', 'non_pbe', 'non_pbe_endcaps', 'global'] # 'known_named_decoding_epochs_type'
# # Define a type that can only be one of these specific strings
# MaskedTimeBinFillType = Literal['ignore', 'last_valid', 'nan_filled', 'dropped'] ## used in `DecodedFilterEpochsResult.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(...)` to specify how invalid bins (due to too few spikes) are treated.
# DataTimeGrain = Literal['per_epoch', 'per_time_bin'] # 'data_grain'
# PrePostDeltaCategory = Literal['pre_delta', 'post_delta'] # 'pre_post_delta_category'


# epoch_index: TypeAlias = int # an integer index into a list of epochs
# time_bin_index: TypeAlias = int # an integer index into a list of time bins

# PastFutureCategory = Literal['past', 'future'] # 'past_futuret_category'

# type_to_name_mapping: Dict = dict(zip([KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, PrePostDeltaCategory], ['known_named_decoder_trained_compute_epochs_type', 'known_named_decoding_epochs_type', 'masked_time_bin_fill_type', 'data_time_grain', 'pre_post_delta_category']))
# name_to_short_name_dict: Dict = dict(zip(['known_named_decoder_trained_compute_epochs_type', 'known_named_decoding_epochs_type', 'masked_time_bin_fill_type', 'data_time_grain', 'pre_post_delta_category'], ['train', 'decode', 'mfill', 'grain', 'ppdelta']))

# type_to_df_column_name_mapping: Dict = dict(zip([KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, PrePostDeltaCategory], ['trained_compute_epochs', 'known_named_decoding_epochs_type', 'masked_time_bin_fill_type', 'data_grain', 'pre_post_delta_category']))
# # other columns: ['time_bin_size']

# # FilterContextName = NewType('FilterContextName', str) 
# # ComputationFunctionName = NewType('ComputationFunctionName', str) 
