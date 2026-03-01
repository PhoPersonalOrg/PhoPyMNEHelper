from collections import namedtuple
from copy import deepcopy
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import nptyping as ND
from nptyping import NDArray
import numpy as np
import pandas as pd

from dataclasses import dataclass
from attrs import define, field, Factory # used for Paginator

# from pyphocorehelpers.function_helpers import function_attributes
# ==================================================================================================================================================================================================================================================================================== #
# 2025-09-12 Copied from pyphocorehelpers                                                                                                                                                                                                                                              #
# ==================================================================================================================================================================================================================================================================================== #


# ==================================================================================================================== #
# List-Like and Iterators                                                                                              #
# ==================================================================================================================== #

def safe_get_if_not_None(a_list: Optional[List], index: int, fallback_value: Any):
    """Similar to dict's .get(key, fallback) function but for lists, and the lists don't even have to be non-None!. Returns a fallback/default value if the index is not valid for the list, otherwise returns the value at that index.
    Args:
        list (_type_): a list-like object
        index (_type_): an index into the list
        fallback_value (_type_): any value to be returned when the indexing fails

    Returns:
        _type_: the value in the list, or the fallback_value is the index is not valid for the list.
        
    Usage:
        from pyphocorehelpers.indexing_helpers import safe_get_if_not_None
            from phoofflineeeganalysis.helpers.indexing_helpers import safe_get_if_not_None
    """
    try:
        if a_list is None:
            return fallback_value # not a list or indexable, return the fallback
        ## otherwise try to de-reference it        
        return a_list[index]
    except TypeError:
        # TypeError: 'NoneType' object is not subscriptable
        return fallback_value
    except IndexError:
        return fallback_value
    
    
def safe_get(list, index, fallback_value):
    """Similar to dict's .get(key, fallback) function but for lists. Returns a fallback/default value if the index is not valid for the list, otherwise returns the value at that index.
    Args:
        list (_type_): a list-like object
        index (_type_): an index into the list
        fallback_value (_type_): any value to be returned when the indexing fails

    Returns:
        _type_: the value in the list, or the fallback_value is the index is not valid for the list.
    """
    try:
        return list[index]
    except IndexError:
        return fallback_value


def safe_len(v):
    """ 2023-05-08 - tries to return the length of v if possible, otherwise returns None """
    try:
        return len(v)
    except Exception as e:
        # raise e
        print(e)
        return None
    

def get_variable_shape(a_value, should_fail_when_cannot_determine:bool=True) -> Union[Tuple[int], int]:
    """ generally and safely tries several methods of determining a_value's shape 
    
    
    from pyphocorehelpers.indexing_helpers import get_variable_shape
    
    assert safe_get_variable_shape(active_one_step_decoder.time_bin_size) is None
    assert isinstance(safe_get_variable_shape(active_one_step_decoder.spikes_df), tuple)
    assert isinstance(safe_get_variable_shape(active_one_step_decoder.F), tuple)
    """
    try:
        value_shape = np.shape(a_value)
    except ValueError:
        # 'ipdb>  np.array(a_value) >>> *** ValueError: could not broadcast input array from shape (2,12) into shape (2,)' occurs when a_value is a list of differently shaped np.arrays
        value_shape = () # set value_shape to () to continue trying other size tests
    except Exception as e:
        raise
    
    if value_shape != ():
        # np.shape(...) worked
        return value_shape
    else:
        # empty shape:
        if hasattr(a_value, 'shape'):
            ## get the shape property
            value_shape = a_value.shape
            return value_shape
        else:
            # didn't work, try len(a_value):
            try:
                value_shape = len(a_value)
            except TypeError as e:
                # no length, no way to get shape
                if should_fail_when_cannot_determine:
                    raise ValueError(f'get_variable_shape(...): all methods for determining the shape of a_value failed! type(a_value): {type(a_value)}, a_value: {a_value}')
                else:
                    value_shape = None
                    return value_shape # value_shape = 'scalar'
                
            except Exception as e:
                raise

    return value_shape


def safe_get_variable_shape(a_value) -> Optional[Union[Tuple[int], int]]:
    """ generally and safely tries several methods of determining a_value's shape 
    
    
    from pyphocorehelpers.indexing_helpers import safe_get_variable_shape
    
    assert safe_get_variable_shape(active_one_step_decoder.time_bin_size) is None
    assert isinstance(safe_get_variable_shape(active_one_step_decoder.spikes_df), tuple)
    assert isinstance(safe_get_variable_shape(active_one_step_decoder.F), tuple)
    """
    return get_variable_shape(a_value=a_value, should_fail_when_cannot_determine=False)


def safe_find_index_in_list(a_list, a_search_obj):
    """ tries to find the index of `a_search_obj` in the list `a_list` 
    If found, returns the index
    If not found, returns None (instead of throwing a ValueError which is the default)
    
    Example:
        an_ax = plots.axs[2]
        safe_find_index_in_list(plots.axs, an_ax)
        # list(plots.axs).index(an_ax)
    """
    if not isinstance(a_list, list):
        a_list = list(a_list) # convert to list
    try:
        return a_list.index(a_search_obj)
    except ValueError as e:
        # Item not found
        return None
    except Exception as e:
        raise e


def is_consecutive_no_gaps(arr, enable_debug_print=False):
    """ Checks whether a passed array/list is a series of ascending indicies without gaps
    
    arr: listlike: checks if the series is from [0, ... , len(arr)-1]
    
    Usage:
        neuron_IDXs = extracted_neuron_IDXs
        is_consecutive_no_gaps(cell_ids, neuron_IDXs)
    """
    if enable_debug_print:
        print(f'is_consecutive_no_gaps(arr: {arr})')
    comparison_correct_sequence = np.arange(len(arr)) # build a series from [0, ... , N-1]
    differing_elements = np.setdiff1d(comparison_correct_sequence, arr)
    if (len(differing_elements) > 0):
        if enable_debug_print:
            print(f'\t differing_elements: {differing_elements}')
        return False
    else:
        return True
    

def bidirectional_setdiff1d(arr0, arr1):
    """ returns a tuple containing the bidirectional setdiff1D in each direction (they can differ) """
    return np.setdiff1d(arr0, arr1), np.setdiff1d(arr1, arr0)

def sorted_slice(a,l,r):
    start = np.searchsorted(a, l, 'left')
    end = np.searchsorted(a, r, 'right')
    return np.arange(start, end)


def chunks(iterable, size=10):
    """ Chunking

    Args:
        iterable ([type]): [description]
        size (int, optional): [description]. Defaults to 10.

    Usage:
        from pyphocorehelpers.indexing_helpers import chunks
        laps_pages = [list(chunk) for chunk in _chunks(sess.laps.lap_id, curr_num_subplots)]
    """
    iterator = iter(iterable)
    for first in iterator:    # stops when iterator is depleted
        def chunk():          # construct generator for next chunk
            yield first       # yield element from for loop
            for more in islice(iterator, size - 1):
                yield more    # yield more elements from the iterator
        yield chunk()         # in outer generator, yield next chunk


def build_pairwise_indicies(target_indicies, debug_print=False):
    """ Builds pairs of indicies from a simple list of indicies, for use in computing pairwise operations.
    
    Example:
        target_indicies = np.arange(5) # [0, 1, 2, 3, 4]
        out_pair_indicies = build_pairwise_indicies(target_indicies)
            > out_pair_indicies: [(0, 1), (1, 2), (2, 3), (3, 4)]
    
    Args:
        target_indicies ([type]): [description]
        debug_print (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
  
    Usage:
        target_indicies = np.arange(5)
        out_pair_indicies = build_pairwise_indicies(target_indicies)
        # out_pair_indicies = list(out_pair_indicies)
        # print(f'out_pair_indicies: {list(out_pair_indicies)}')


        print(f'out_pair_indicies: {list(out_pair_indicies)}')

        for i, pair in enumerate(list(out_pair_indicies)):
            # first_item_lap_idx, next_item_lap_idx
            print(f'i: {i}, pair: {pair}')
    """
    start_pairs = target_indicies[0:-1] # all but the last index
    end_pairs = target_indicies[1:] # from the second to the last index
    out_pair_indicies = list(zip(start_pairs, end_pairs)) # want to wrap in list so it isn't consumed
    if debug_print:
        print(f'target_indicies: {target_indicies}\nstart_pairs: {start_pairs}\nend_pairs: {end_pairs}')
    return out_pair_indicies


def interleave_elements(start_points, end_points, debug_print:bool=False):
    """ Given two equal sized arrays, produces an output array of double that size that contains elements of start_points interleaved with elements of end_points
    Example:
        a_starts = ['A','B','C','D']
        a_ends = ['a','b','c','d']
        a_interleaved = interleave_elements(a_starts, a_ends)
        >> a_interleaved: ['A','a','B','b','C','c','D','d']
    """
    if not isinstance(start_points, np.ndarray):
        start_points = np.array(start_points)
    if not isinstance(end_points, np.ndarray):
        end_points = np.array(end_points)
    assert np.shape(start_points) == np.shape(end_points), f"start_points and end_points must be the same shape. np.shape(start_points): {np.shape(start_points)}, np.shape(end_points): {np.shape(end_points)}"
    # Capture initial shapes to determine if np.atleast_2d changed the shapes
    start_points_initial_shape = np.shape(start_points)
    end_points_initial_shape = np.shape(end_points)
    
    # Capture initial datatypes for building the appropriate empty np.ndarray later:
    start_points_dtype = start_points.dtype # e.g. 'str32'
    end_points_dtype = end_points.dtype # e.g. 'str32'
    assert start_points_dtype == end_points_dtype, f"start_points and end_points must be the same datatype. start_points.dtype: {start_points.dtype.name}, end_points.dtype: {end_points.dtype.name}"
    start_points = np.atleast_2d(start_points)
    end_points = np.atleast_2d(end_points)
    if debug_print:
        print(f'start_points: {start_points}\nend_points: {end_points}')
        print(f'np.shape(start_points): {np.shape(start_points)}\tnp.shape(end_points): {np.shape(end_points)}') # np.shape(start_points): (1, 4)	np.shape(end_points): (1, 4)
        print(f'start_points_dtype.name: {start_points_dtype.name}\tend_points_dtype.name: {end_points_dtype.name}')
      
    if (np.shape(start_points) != start_points_initial_shape) and (np.shape(start_points)[0] == 1):
        # Shape changed after np.atleast_2d(...) which erroniously adds the newaxis to the 0th dimension. Fix by transposing:
        start_points = start_points.T
    if (np.shape(end_points) != end_points_initial_shape) and (np.shape(end_points)[0] == 1):
        # Shape changed after np.atleast_2d(...) which erroniously adds the newaxis to the 0th dimension. Fix by transposing:
        end_points = end_points.T
    if debug_print:
        print(f'POST-TRANSFORM: np.shape(start_points): {np.shape(start_points)}\tnp.shape(end_points): {np.shape(end_points)}') # POST-TRANSFORM: np.shape(start_points): (4, 1)	np.shape(end_points): (4, 1)
    all_points_shape = (np.shape(start_points)[0] * 2, np.shape(start_points)[1]) # it's double the length of the start_points
    if debug_print:
        print(f'all_points_shape: {all_points_shape}') # all_points_shape: (2, 4)
    # all_points = np.zeros(all_points_shape)
    all_points = np.empty(all_points_shape, dtype=start_points_dtype) # Create an empty array with the appropriate dtype to hold the objects
    all_points[np.arange(0, all_points_shape[0], 2), :] = start_points # fill the even elements
    all_points[np.arange(1, all_points_shape[0], 2), :] = end_points # fill the odd elements
    assert np.shape(all_points)[0] == (np.shape(start_points)[0] * 2), f"newly created all_points is not of corrrect size! np.shape(all_points): {np.shape(all_points)}"
    return np.squeeze(all_points)


def are_all_equal(arr) -> bool:
    """ returns True if arr is empty, or if all elements of arr are equal to each other """
    if len(arr) == 0:
        return True
    else:
        val = arr[0] # get first element
        return np.all([x == val for x in arr])
    

# ==================================================================================================================== #
# Dictionary and Maps                                                                                                  #
# ==================================================================================================================== #

def get_dict_subset(a_dict, included_keys=None, subset_excludelist=None, require_all_keys=False):
    """Gets a subset of a dictionary from a list of keys (included_keys)

    Args:
        a_dict ([type]): [description]
        included_keys ([type], optional): [description]. Defaults to None.
        require_all_keys: Bool, if True, requires all keys in included_keys to be in the dictionary (a_dict)

    Returns:
        [type]: [description]
    """
    if subset_excludelist is not None:
        assert included_keys is None, "included_keys must be None when a subset_excludelist is provided!"
        included_keys = [key for key in a_dict.keys() if key not in subset_excludelist]
        
    if included_keys is not None:
        if require_all_keys:
            return {included_key:a_dict[included_key] for included_key in included_keys} # filter the dictionary for only the keys specified
        else:
            out_dict = {}
            for included_key in included_keys:
                if included_key in a_dict.keys():
                    out_dict[included_key] = a_dict[included_key]
            return out_dict
    else:
        return a_dict

def validate_reverse_index_map(value_to_original_index_reverse_map, neuron_IDXs, cell_ids, debug_print=True):
    """
    Used to be called `validate_cell_IDs_to_CellIDXs_map`

    value_to_original_index_reverse_map: is a dictioanry that has any thing for its keys, but each
        Example:
            # Allows reverse indexing into the linear imported array using the original cell ID indicies:
            id_arr = [ 2  3  4  5  7  8  9 10 11 12 14 17 18 21 22 23 24 25 26 27 28 29 33 34 38 39 42 44 45 46 47 48 53 55 57 58 61 62 63 64]
            linear_flitered_ids = np.arange(len(id_arr)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
            value_to_original_index_reverse_map = dict(zip(id_arr, linear_flitered_ids))    
     
    
    Usage:
        cell_ids = extracted_cell_ids
        neuron_IDXs = extracted_neuron_IDXs
        reverse_cellID_index_map = ipcDataExplorer.active_session.neurons.reverse_cellID_index_map
        validate_reverse_index_map(reverse_cellID_index_map, cell_ids, neuron_IDXs)
    """
    if debug_print:
        print(f'\t cell_ids: {cell_ids}')
        print(f'\t neuron_IDXs: {neuron_IDXs}')
    if not is_consecutive_no_gaps(neuron_IDXs, enable_debug_print=debug_print):
        if debug_print:
            print('neuron_IDXs has gaps!')
        return False
    else:
        map_start_ids = list(value_to_original_index_reverse_map.keys()) # the cellIDs that can be mapped from
        differing_elements_ids = np.setdiff1d(map_start_ids, cell_ids)
        num_differing_ids = len(differing_elements_ids)
        map_destination_IDXs = list(value_to_original_index_reverse_map.values()) # the cellIDXs that can be mapped to.
        differing_elements_IDXs = np.setdiff1d(map_destination_IDXs, neuron_IDXs)
        num_differing_IDXs = len(differing_elements_IDXs)
        if (num_differing_IDXs > 0) or (num_differing_ids > 0):
            if debug_print:
                print(f'\t differing_elements_IDXs: {differing_elements_IDXs}')
                print(f'\t differing_elements_ids: {differing_elements_ids}')
            return False
        else:
            return True

def nested_dict_set(dic, key_list, value, create_missing=True):
    """ Allows setting the value of a nested dictionary hierarchy by drilling in with the keys in key_list, creating intermediate dictionaries if needed.
    
    Attribution and Credit:
        https://stackoverflow.com/a/49290758/9732163
    
    Usage:
        d = {}
        nested_set(d, ['person', 'address', 'city'], 'New York')
        d
        >> {'person': {'address': {'city': 'New York'}}}
    """
    d = dic
    for key in key_list[:-1]:
        if key in d:
            d = d[key]
        elif create_missing:
            d = d.setdefault(key, {})
        else:
            return dic
    if key_list[-1] in d or create_missing:
        d[key_list[-1]] = value
    return dic

def flatpaths_to_nested_dicts(flat_paths_form_dict, default_value_override='Test Value', flat_path_delimiter='.', debug_print=False):
    """ Reciprocal of nested_dicts_to_flatpaths(...)

    Args:
        flat_paths_list (_type_): _description_
        default_value_override (str, optional): _description_. Defaults to 'Test Value'.
        debug_print (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
        
    Usage:
        from pyphocorehelpers.indexing_helpers import nested_dicts_to_flatpaths, flatpaths_to_nested_dicts
        flatpaths_to_nested_dicts(['SpikeAnalysisComputations._perform_spike_burst_detection_computation',
        'ExtendedStatsComputations._perform_placefield_overlap_computation',
        'ExtendedStatsComputations._perform_firing_rate_trends_computation',
        'ExtendedStatsComputations._perform_extended_statistics_computation',
        'DefaultComputationFunctions._perform_velocity_vs_pf_density_computation',
        'DefaultComputationFunctions._perform_two_step_position_decoding_computation',
        'DefaultComputationFunctions._perform_position_decoding_computation',
        'PlacefieldComputations._perform_time_dependent_placefield_computation',
        'PlacefieldComputations._perform_baseline_placefield_computation'])


        flatpaths_to_nested_dicts(_temp_compuitations_flat_functions_list)
    
    """
    out_hierarchy_dict = {}
    if isinstance(flat_paths_form_dict, list):
        flat_paths_dict = {a_flat_path:default_value_override for a_flat_path in flat_paths_form_dict}
    else:
        flat_paths_dict = flat_paths_form_dict
        # Otherwise should already have an .items() method:
        
    # for a_flat_path in flat_paths_list:
    for a_flat_path, a_value in flat_paths_dict.items():
        key_hierarchy_list = a_flat_path.split(flat_path_delimiter)
        if debug_print:
            print(f'for item {a_flat_path}:')
        out_hierarchy_dict = nested_dict_set(out_hierarchy_dict, key_hierarchy_list, a_value, create_missing=True)
        if debug_print:
            print(f'\t out_hierarchy_dict: {out_hierarchy_dict}')
    return out_hierarchy_dict


_GLOBAL_MAX_DEPTH = 20
def nested_dicts_to_flatpaths(curr_key, curr_value, max_depth=20, depth=0, flat_path_delimiter='.', debug_print=False):
    """ Reciprocal of flatpaths_to_nested_dicts(...)

        curr_key: None to start
        curr_value: assumed to be nested_hierarchy_dict to start


    Usage:
        from pyphocorehelpers.indexing_helpers import nested_dicts_to_flatpaths, flatpaths_to_nested_dicts
        _out_flatpaths_dict = nested_dicts_to_flatpaths('', _temp_compuitations_functions_list)
        _out_flatpaths_dict

        {'SpikeAnalysisComputations._perform_spike_burst_detection_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis.SpikeAnalysisComputations._perform_spike_burst_detection_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'ExtendedStatsComputations._perform_placefield_overlap_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats.ExtendedStatsComputations._perform_placefield_overlap_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'ExtendedStatsComputations._perform_firing_rate_trends_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats.ExtendedStatsComputations._perform_firing_rate_trends_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'ExtendedStatsComputations._perform_extended_statistics_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats.ExtendedStatsComputations._perform_extended_statistics_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'DefaultComputationFunctions._perform_velocity_vs_pf_density_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions.DefaultComputationFunctions._perform_velocity_vs_pf_density_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'DefaultComputationFunctions._perform_two_step_position_decoding_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions.DefaultComputationFunctions._perform_two_step_position_decoding_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'DefaultComputationFunctions._perform_position_decoding_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions.DefaultComputationFunctions._perform_position_decoding_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult)>,
 'PlacefieldComputations._perform_time_dependent_placefield_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_time_dependent_placefield_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'PlacefieldComputations._perform_baseline_placefield_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_baseline_placefield_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>}
 
 
        _input_test_functions_list = _temp_compuitations_functions_list
        _out_flatpaths_dict = nested_dicts_to_flatpaths('', _input_test_functions_list)
        _original_hierarchical_test_dict = flatpaths_to_nested_dicts(_out_flatpaths_dict)
        assert _original_hierarchical_test_dict == _input_test_functions_list, "ERROR, flatpaths_to_nested_dicts(nested_dicts_to_flatpaths(INPUT)) should be identity transforms (should == INPUT), but they do not!"

        
    """
    if (depth >= _GLOBAL_MAX_DEPTH):
        print(f'OVERFLOW AT DEPTH {_GLOBAL_MAX_DEPTH}!')
        raise OverflowError
    elif (depth > max_depth):
        if debug_print:
            print(f'finished at DEPTH {depth} with max_depth: {max_depth}!')
        return None
        
    else:
        if debug_print:
            curr_value_type = type(curr_value)
            print(f'curr_value_type: {curr_value_type}')
            print(f'curr_value: {curr_value}')
        
        # See if the curr_value has .items() or not.
        try:    
            child_out_dict = {}
            for (curr_child_key, curr_child_value) in curr_value.items():
                # prints the current value:
                if debug_print:
                    print(f"\t {curr_child_key} - {type(curr_child_value)}")
                # process children keys
                if curr_key is None or curr_key == '':
                    child_key_path = f'{curr_child_key}' # the curr_child_key is the top-level item
                else:
                    child_key_path = f'{curr_key}{flat_path_delimiter}{curr_child_key}'    
                curr_out = nested_dicts_to_flatpaths(child_key_path, curr_child_value, max_depth=max_depth, depth=(depth+1))
                if curr_out is not None:
                    if debug_print:
                        print(f'\t curr_out: {curr_out}')
                    child_out_dict = child_out_dict | curr_out # merge in the new dict value
                    
            return child_out_dict
            # curr_value = child_out_dict
            # is_terminal_item = False   
            
        except AttributeError as e:                

            is_terminal_item = True
    
 
        if is_terminal_item:
            # A concrete item:
            if debug_print:
                print(f'TERMINAL ITEM: ({curr_key}, {curr_value})')
            return {curr_key: curr_value}
        else:
            if debug_print:
                print(f'NON-terminal item: ({curr_key}, {curr_value})')
            return None

def apply_to_dict_values(a_dict: dict, a_callable: Callable, include_condition: Callable = None) -> dict:
    """ applies the Callable a_callable to the values of a_dict 
    e.g. 

    from pyphocorehelpers.indexing_helpers import apply_to_dict_values
    """
    if include_condition is not None:
        return {k:a_callable(v) for k, v in a_dict.items() if include_condition(k,v)}
    else:
        return {k:a_callable(v) for k, v in a_dict.items()}


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for item in list_of_dicts:
        for key, value in item.items():
            if key in dict_of_lists:
                dict_of_lists[key].append(value)
            else:
                dict_of_lists[key] = [value]
    return dict_of_lists
    
        
def reorder_keys(a_dict: Dict, key_name_desired_index_dict: Dict[str, int]) -> Dict:
    """Reorders specified keys in a Dict while preserving other keys.
    
    based off of `reorder_columns`
                
    """
    # Validate column names
    missing_columns = set(key_name_desired_index_dict.keys()) - set(a_dict.keys())
    if missing_columns:
        raise ValueError(f"Keys {missing_columns} not found in the Dict.")

    # Ensure desired indices are unique and within range
    desired_indices = key_name_desired_index_dict.values()
    if len(set(desired_indices)) != len(desired_indices) or any(index < 0 or index >= len(list(a_dict.keys())) for index in desired_indices):
        raise ValueError("Desired indices must be unique and within the range of existing keys.")

    # Create a list of columns to reorder
    reordered_columns_desired_index_dict: Dict[str, int] = {column_name:desired_index for column_name, desired_index in sorted(key_name_desired_index_dict.items(), key=lambda item: item[1])}
    # print(reordered_columns_desired_index_dict)
    
    # # Reorder specified columns while preserving remaining columns
    remaining_columns = [col for col in list(a_dict.keys()) if col not in key_name_desired_index_dict]
    
    reordered_columns_list: List[str] = remaining_columns.copy()
    for item_to_insert, desired_index in reordered_columns_desired_index_dict.items():    
        reordered_columns_list.insert(desired_index, item_to_insert)
        
    # print(reordered_columns_list)
    reordered_dict = {k:a_dict[k] for k in reordered_columns_list}
    return reordered_dict

def reorder_keys_relative(a_dict: Dict, key_names: List[str], relative_mode='end') -> Dict:
    """Reorders specified keys in a Dict while preserving other keys.
    
    Based off of `reorder_columns_relative`
                
    Usage:
        from pyphocorehelpers.indexing_helpers import reorder_keys_relative

    """
    if relative_mode == 'end':
        existing_columns = list(a_dict.keys())
        return reorder_keys(a_dict, key_name_desired_index_dict=dict(zip(key_names, np.arange(len(existing_columns)-4, len(existing_columns))))) # -4 ???
    else:
        raise NotImplementedError
    

def set_if_none(d: Dict, key, default):
    """ similar to `setdefault(...)` for dict but even if the dictionary has the key, if its value is None it will replace it with the provided default
    
    Usage:
        from pyphocorehelpers.indexing_helpers import set_if_none
        set_if_none(a_config, key='dockAddLocationOpts', default=('bottom', ))
    
    """
    if d.get(key, None) is None:
        ## key either doesn't exist, or does exist but has a value of `None`. Set the default
        d[key] = default
    return d[key]
    


# ==================================================================================================================== #
# Numpy NDArrays                                                                                                       #
# ==================================================================================================================== #

# @function_attributes(short_name=None, tags=['numpy', 'safe', 'indexing','list'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-08 20:06', related_items=[])
def safe_numpy_index(arr, idxs: np.ndarray):
    """ tries to prevent errors when arr is a list and idxs is a numpy array. """
    try:
        return arr[idxs] # works for arr: np.ndarray and idxs: np.ndarray
    except TypeError as e:
        # "TypeError: only integer scalar arrays can be converted to a scalar index"
        # Occurs when arr is list and idxs: np.ndarray
        if isinstance(arr, list):
            return [arr[i] for i in idxs] # returns a list object
        else:
            print(f"NotImplementedError: type(arr): {type(arr)}, type(idxs): {type(idxs)}")
            raise NotImplementedError
    except Exception as e:
        raise e
        

def safe_np_vstack(arr):
    """ a version of np.vstack that doesn't throw a ValueError on empty lists
        from pyphocorehelpers.indexing_helpers import safe_np_vstack

    """
    if len(arr)>0:
        return np.vstack(arr) # without this check np.vstack throws `ValueError: need at least one array to concatenate` for empty lists
    else:
        return np.array(arr) # ChatGPT suggests returning np.empty((0, 0))  # Create an empty array with shape (0, 0)

def safe_np_hstack(arr):
    """ a version of np.hstack that doesn't throw a ValueError on empty lists
        from pyphocorehelpers.indexing_helpers import safe_np_hstack

    """
    if len(arr)>0:
        return np.hstack(arr) # without this check np.vstack throws `ValueError: need at least one array to concatenate` for empty lists
    else:
        return np.array(arr) # ChatGPT suggests returning np.empty((0, 0))  # Create an empty array with shape (0, 0)
    

def dict_to_full_array(a_dict: Dict, full_indicies: NDArray, fill_value=-1) -> NDArray:
    """ 
    a_dict: the dictionary of values you want to build into a NDArray
    full_indicies: NDArray - the completely list of all possible indicies that you want to build the array for. Any matching entries in `a_dict.keys()` will be filled with their corresponding value, otherwise `fill_value` will be used.
    Returns a NDArray of size: (len(full_indicies), )
    """
    keys_list =	list(a_dict.keys())
    values_list = list(a_dict.values())
    found_indicies = np.array([list(full_indicies).index(k) for k in keys_list])
    # print(f'found_indicies: {found_indicies}')
    key_to_index_map = dict(zip(keys_list, found_indicies))
    out_array: NDArray = np.full_like(full_indicies, fill_value=fill_value)
    out_array[found_indicies] = np.array(values_list)
    # print(f'values_list: {values_list}')
    # print(f'out_array: {out_array}')
    return out_array


class NumpyHelpers:
    """ various extensions and generalizations for numpy arrays 
    
    from pyphocorehelpers.indexing_helpers import NumpyHelpers


    """
    @classmethod
    def all_array_generic(cls, pairwise_numpy_fn, list_of_arrays: List[NDArray], **kwargs) -> bool:
        """ A n-element generalization of a specified pairwise numpy function such as `np.array_equiv`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_array_generic(list_of_arrays=list_of_arrays)

        """
        # Input type checking
        if not np.all(isinstance(arr, np.ndarray) for arr in list_of_arrays):
            raise ValueError("All elements in 'list_of_arrays' must be NumPy arrays.")        
    
        if len(list_of_arrays) == 0:
            return True # empty arrays are all equal
        elif len(list_of_arrays) == 1:
            # if only a single array, make sure it's not accidentally passed in incorrect
            reference_array = list_of_arrays[0] # Use the first array as a reference for comparison
            assert isinstance(reference_array, np.ndarray)
            return True # as long as imput is intended, always True
        
        else:
            ## It has more than two elements:
            reference_array = list_of_arrays[0] # Use the first array as a reference for comparison
            # Check equivalence for each array in the list
            return np.all([pairwise_numpy_fn(reference_array, an_arr, **kwargs) for an_arr in list_of_arrays[1:]]) # can be used without the list comprehension just as a generator if you use all(...) instead.
            # return all(np.all(np.array_equiv(reference_array, an_arr) for an_arr in list_of_arrays[1:])) # the outer 'all(...)' is required, otherwise it returns a generator object like: `<generator object NumpyHelpers.all_array_equiv.<locals>.<genexpr> at 0x00000128E0482AC0>`

    @classmethod
    def all_array_equal(cls, list_of_arrays: List[NDArray], equal_nan=True) -> bool:
        """ A n-element generalization of `np.array_equal`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_array_equal(list_of_arrays=list_of_arrays)

        """
        return cls.all_array_generic(np.array_equal, list_of_arrays=list_of_arrays, equal_nan=equal_nan)
    
    @classmethod
    def all_array_equiv(cls, list_of_arrays: List[NDArray]) -> bool:
        """ A n-element generalization of `np.array_equiv`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_array_equiv(list_of_arrays=list_of_arrays)

        """
        return cls.all_array_generic(np.array_equiv, list_of_arrays=list_of_arrays)


    @classmethod
    def all_allclose(cls, list_of_arrays: List[NDArray], rtol:float=1.e-5, atol:float=1.e-8, equal_nan:bool=True) -> bool:
        """ A n-element generalization of `np.allclose`
        Usage:
        
            list_of_arrays = list(xbins.values())
            NumpyHelpers.all_allclose(list_of_arrays=list_of_arrays)

        """
        return cls.all_array_generic(np.allclose, list_of_arrays=list_of_arrays, rtol=rtol, atol=atol, equal_nan=equal_nan)
    


# ==================================================================================================================== #
# Pandas Dataframes                                                                                                    #
# ==================================================================================================================== #


def safe_pandas_get_group(dataframe_group, key):
    """ returns an empty dataframe if the key isn't found in the group."""
    if key in dataframe_group.groups.keys():
        return dataframe_group.get_group(key)
    else:
        original_df = dataframe_group.obj
        return original_df.drop(original_df.index)
    

## Pandas DataFrame helpers:
def partition(df: pd.DataFrame, partitionColumn: str) -> Tuple[NDArray, NDArray]:
    """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.

    Usage:

    from pyphocorehelpers.indexing_helpers import partition, partition_df, partition_df_dict


    History: refactored from `pyphoplacecellanalysis.PhoPositionalData.analysis.helpers`
    """
    unique_values = np.unique(df[partitionColumn]) # array([ 0,  1,  2,  3,  4,  7, 11, 12, 13, 14])
    grouped_df = df.groupby([partitionColumn]) #  Groups on the specified column.
    return unique_values, np.array([grouped_df.get_group(aValue) for aValue in unique_values], dtype=object) # dataframes split for each unique value in the column

def partition_df(df: pd.DataFrame, partitionColumn: str)-> Tuple[NDArray, List[pd.DataFrame]]:
    """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.

    USEFUL NOTE: to get a dict, do `partitioned_dfs = dict(zip(*partition_df(spikes_df, partitionColumn='new_epoch_IDX')))`
    
    Usage:
        from pyphocorehelpers.indexing_helpers import partition_df
        
        partitioned_dfs = dict(zip(*partition_df(spikes_df, partitionColumn='new_epoch_IDX')))

        
        unique_values, partitioned_dfs_list = partition_df(spikes_df, partitionColumn='new_epoch_IDX')

    History: refactored from `pyphoplacecellanalysis.PhoPositionalData.analysis.helpers`
    """
    unique_values = np.unique(df[partitionColumn]) # array([ 0,  1,  2,  3,  4,  7, 11, 12, 13, 14])
    grouped_df = df.groupby([partitionColumn]) #  Groups on the specified column.
    return unique_values, [grouped_df.get_group(aValue) for aValue in unique_values] # dataframes split for each unique value in the column

def partition_df_dict(df: pd.DataFrame, partitionColumn: str)-> Dict[Any, pd.DataFrame]:
    """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.

    Usage:
        from pyphocorehelpers.indexing_helpers import partition_df_dict
        
        partitioned_dfs = partition_df_dict(spikes_df, partitionColumn='new_epoch_IDX')

    History: refactored from `pyphoplacecellanalysis.PhoPositionalData.analysis.helpers`
    """
    return dict(zip(*partition_df(df, partitionColumn=partitionColumn))) # dataframes split for each unique value in the column


def find_neighbours(value, df, colname):
    """Claims to be O(N)
    From https://stackoverflow.com/questions/30112202/how-do-i-find-the-closest-values-in-a-pandas-series-to-an-input-number
    
    Args:
        value ([type]): [description]
        df ([type]): [description]
        colname ([type]): [description]

    Returns:
        [type]: [description]
    """
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind] 
    
    
    #If the series is already sorted, an efficient method of finding the indexes is by using bisect functions.
 

# Concatenate dataframes
def simple_merge(*dfs_list, debug_print=False) -> pd.DataFrame:
    """ naievely merges several dataframes with an equal number of rows (and in the same order) into a single dataframe with all of the unique columns of the individual dfs. Any duplicate columns will be removed.

    Usage:
        dfs_list = (deepcopy(neuron_identities_table), deepcopy(long_short_fr_indicies_analysis_table), deepcopy(neuron_replay_stats_table))
        dfs_list = (deepcopy(neuron_identities_table), deepcopy(long_short_fr_indicies_analysis_table), deepcopy(neuron_replay_stats_table))
        df_combined, dropped_duplicate_columns = simple_merge(*dfs_list, debug_print=False)
        df_combined

    """
    assert are_all_equal([len(x) for x in dfs_list]), f"all dataframes must be the same length but [len(x) for x in dfs_list]: {[len(x) for x in dfs_list]}"
    df_combined = pd.concat(dfs_list, axis=1)
    # df_combined = pd.concat([df1, df2, df3], axis=1)

    # Remove duplicate columns if values are the same
    to_drop = []
    columns = df_combined.columns
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            if columns[i] == columns[j] and df_combined[columns[i]].equals(df_combined[columns[j]]):
                to_drop.append(columns[j])
    if debug_print:
        print(f"to_drop: {to_drop}")
    df_combined = df_combined.drop(columns=to_drop)
    # # Handle columns with the same name but conflicting values
    # # (Here, we're simply renaming them for clarity, you can handle them differently if needed)
    # for col in to_drop:
    #     df_combined[col + '_conflict'] = df_combined[col]
    # print(df_combined)
    return df_combined, to_drop


def join_on_index(*dfs, join_index='aclu', suffixes_list=None) -> pd.DataFrame:
    """ Joins a series of dataframes on a common index (`join_index`)
    from pyphocorehelpers.indexing_helpers import join_on_index
    
    suffixes_list = (('_lsfria', '_jfra'), ('_jfra', '_lspd'))
    joined_df = join_on_index(long_short_fr_indicies_df, neuron_replay_stats_df, rate_remapping_df, join_index='aclu', suffixes_list=suffixes_list)

    """
    if suffixes_list is not None:
        assert len(suffixes_list) == (len(dfs[1:])), f"{len(suffixes_list)} == {(len(dfs[1:]))}"
    else:
        suffixes_list = ('_x', '_y') * len(dfs[1:])
    joined_df: pd.DataFrame = dfs[0]
    for df, a_suffix_pair in zip(dfs[1:], suffixes_list):
        # joined_df = joined_df.join(df, how='inner')
        joined_df = joined_df.merge(df, on=join_index, how='inner', suffixes=a_suffix_pair)
    return joined_df



def reorder_columns(df: pd.DataFrame, column_name_desired_index_dict: Union[List[str], Dict[str, int]]) -> pd.DataFrame:
    """Reorders specified columns in a DataFrame while preserving other columns.
    
    Pure: Does not modify the df

    Args:
        df (pd.DataFrame): The DataFrame to reorder.
        column_name_desired_index_dict (Dict[str, int]): A dictionary where keys are column names
            to reorder and values are their desired indices in the reordered DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with specified columns reordered while preserving remaining columns.

    Raises:
        ValueError: If any column in the dictionary is not present in the DataFrame.
        
        
    Usage:
    
        from pyphocorehelpers.indexing_helpers import reorder_columns
        dict(zip(['Long_LR_evidence', 'Long_RL_evidence', 'Short_LR_evidence', 'Short_RL_evidence'], np.arange(4)+4))
        reorder_columns(merged_complete_epoch_stats_df, column_name_desired_index_dict=dict(zip(['Long_LR_evidence', 'Long_RL_evidence', 'Short_LR_evidence', 'Short_RL_evidence'], np.arange(4)+4)))
        
        ## Move the "height" columns to the end
        result_df = reorder_columns(result_df, column_name_desired_index_dict=dict(zip(list(filter(lambda column: column.endswith('_peak_heights'), result_df.columns)), np.arange(len(result_df.columns)-4, len(result_df.columns)))))
        result_df
                
    """
    if isinstance(column_name_desired_index_dict, (list, tuple)):
        # not a dict, assume the provided list specifies the order of the first elements
        column_name_desired_index_dict = dict(zip(column_name_desired_index_dict, np.arange(len(column_name_desired_index_dict))))

    # Validate column names
    missing_columns: bool = set(column_name_desired_index_dict.keys()) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    # Ensure desired indices are unique and within range
    desired_indices = column_name_desired_index_dict.values()
    if len(set(desired_indices)) != len(desired_indices) or any(index < 0 or index >= len(df.columns) for index in desired_indices):
        raise ValueError("Desired indices must be unique and within the range of existing columns.")

    # Create a list of columns to reorder
    reordered_columns_desired_index_dict: Dict[str, int] = {column_name:desired_index for column_name, desired_index in sorted(column_name_desired_index_dict.items(), key=lambda item: item[1])}
    # print(reordered_columns_desired_index_dict)
    
    # # Reorder specified columns while preserving remaining columns
    remaining_columns = [col for col in df.columns if col not in column_name_desired_index_dict]
    
    reordered_columns_list: List[str] = remaining_columns.copy()
    for item_to_insert, desired_index in reordered_columns_desired_index_dict.items():    
        reordered_columns_list.insert(desired_index, item_to_insert)
        
    # print(reordered_columns_list)
    reordered_df = df[reordered_columns_list]
    return reordered_df


def reorder_columns_relative(df: pd.DataFrame, column_names: list[str], relative_mode='end') -> pd.DataFrame:
    """Reorders specified columns in a DataFrame while preserving other columns.
    
    Pure: Does not modify the df

    Args:
        df (pd.DataFrame): The DataFrame to reorder.
        column_name_desired_index_dict (Dict[str, int]): A dictionary where keys are column names
            to reorder and values are their desired indices in the reordered DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with specified columns reordered while preserving remaining columns.

    Raises:
        ValueError: If any column in the dictionary is not present in the DataFrame.
        
        
    Usage:
    
        from pyphocorehelpers.indexing_helpers import reorder_columns, reorder_columns_relative
        
        ## Move the "height" columns to the end
        result_df = reorder_columns_relative(result_df, column_names=list(filter(lambda column: column.endswith('_peak_heights'), existing_columns)), relative_mode='end')
        result_df
                
        
    Usage 2:
        # move the specified columns to the start of the df:
        neuron_replay_stats_table = reorder_columns_relative(neuron_replay_stats_table, column_names=['neuron_uid', 'format_name', 'animal', 'exper_name', 'session_name', 'neuron_type', 'aclu', 'session_uid', 'session_datetime'],
                                                    relative_mode='start')


    """
    existing_columns = list(df.columns)

    if relative_mode == 'end':    
        return reorder_columns(df, column_name_desired_index_dict=dict(zip(column_names, np.arange(len(existing_columns)-4, len(existing_columns)))))
    elif relative_mode == 'start':    
        return reorder_columns(df, column_name_desired_index_dict=column_names)    
    else:
        raise NotImplementedError(f'unexpected relative_mode, valid: ["start", "end"] provided: relative_mode: {relative_mode}')

             
# @function_attributes(short_name=None, tags=['UNFINISHED', 'AI', 'split', 'df'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-17 16:07', related_items=[])
def _do_partition_df_dict(df: pd.DataFrame, partitionColumn: Union[str, Tuple[str], List[str]], prev_keys=None) -> Tuple[Dict[Any, pd.DataFrame], List]:
    """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
    Recursively processes multiple partition columns to create a nested dictionary structure.
    
    Usage:
        from benedict import benedict
        from pyphocorehelpers.indexing_helpers import _do_partition_df_dict, partition_df_dict

        result_dict, keys_path = _do_partition_df_dict(df=deepcopy(df), partitionColumn=['known_named_decoding_epochs_type', 'trained_compute_epochs', 'masked_time_bin_fill_type']) #.pho.partition_df_dict(partitionColumn=curr_partition_column_name) for a_key, a_df in a_df_dict.items()}
        result_dict = benedict(result_dict)
        result_dict.keypaths()

    """
    if prev_keys is None:
        prev_keys = []
        
    if isinstance(partitionColumn, str):        
        return (partition_df_dict(df=df, partitionColumn=partitionColumn), [*prev_keys, partitionColumn])
    else:
        ## a list of columns to partition on, building a nested dictionary
        if len(partitionColumn) == 0:
            ## finished recursively expanding
            return (df, [*prev_keys])
        else:
            # Make a copy to avoid modifying the original list
            remaining_columns = partitionColumn.copy()
            # Get the current column to partition on
            curr_partition_column_name = remaining_columns.pop(0)
            
            # Get dictionary of dataframes partitioned by current column
            curr_df_dict = partition_df_dict(df=df, partitionColumn=curr_partition_column_name)
            
            # If no more columns to partition, return the current partition dictionary
            if len(remaining_columns) == 0:
                return (curr_df_dict, [*prev_keys, curr_partition_column_name])
            
            # Otherwise, recursively partition each dataframe in the current dictionary
            out_final_dict = {}
            for key, sub_df in curr_df_dict.items():
                # Recursively partition the sub-dataframe with remaining columns
                nested_dict, updated_keys = _do_partition_df_dict(
                    df=sub_df, 
                    partitionColumn=remaining_columns, 
                    prev_keys=[*prev_keys, curr_partition_column_name]
                )
                out_final_dict[key] = nested_dict
                
            return (out_final_dict, updated_keys)

# ## TESTING 2025-03-17 16:15 
# """ 
# partitionColumn = ['known_named_decoding_epochs_type', 'trained_compute_epochs', 'masked_time_bin_fill_type']
# print(f'partitionColumn: {partitionColumn}')

# curr_partition_column_name = partitionColumn.pop()
# print(f'curr_partition_column_name: "{curr_partition_column_name}"')
# print(f'partitionColumn: {partitionColumn}')
# a_df_dict: Dict[str, pd.DataFrame] = df.pho.partition_df_dict(partitionColumn=curr_partition_column_name)

# curr_partition_column_name = partitionColumn.pop()
# print(f'curr_partition_column_name: "{curr_partition_column_name}"')
# print(f'partitionColumn: {partitionColumn}')
# # a_df_dict: Dict[str, pd.DataFrame] = df.pho.partition_df_dict(partitionColumn=curr_partition_column_name)

# a_df_dict: Dict[str, pd.DataFrame] = {a_key:a_df.pho.partition_df_dict(partitionColumn=curr_partition_column_name) for a_key, a_df in a_df_dict.items()}

# a_df_dict

# """

# """ 
# # Using the pandas accessor shown in your codebase
# partition_columns = ['known_named_decoding_epochs_type', 'trained_compute_epochs', 'masked_time_bin_fill_type']

# # Use the recursive function implementation
# result_dict, keys_path = _do_partition_df_dict(df=df, partitionColumn=partition_columns)

# # Or you could use the manual approach shown in your code sample:
# manual_result = {}
# columns_to_process = partition_columns.copy()

# # First level partition
# curr_column = columns_to_process.pop()
# first_level_dict = df.pho.partition_df_dict(partitionColumn=curr_column)

# # Second level partition
# curr_column = columns_to_process.pop()
# second_level_dict = {key: sub_df.pho.partition_df_dict(partitionColumn=curr_column) 
#                      for key, sub_df in first_level_dict.items()}

# # Third level partition (if needed)
# if columns_to_process:
#     curr_column = columns_to_process.pop()
#     final_dict = {key1: {key2: sub_df2.pho.partition_df_dict(partitionColumn=curr_column)
#                          for key2, sub_df2 in sub_dict.items()}
#                  for key1, sub_dict in second_level_dict.items()}
#     print("Created 3-level nested dictionary structure")

# """
    
# ==================================================================================================================== #
# 2024-11-15 - `neuropy` dataframe helper                                                              #
# ==================================================================================================================== #

@pd.api.extensions.register_dataframe_accessor("pho")
class PhoDataframeAccessor:
    """ Describes a dataframe with at least a neuron_id (aclu) column. Provides functionality regarding building globally (across-sessions) unique neuron identifiers.
    

    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor
    from neuropy.utils.indexing_helpers import NeuroPyDataframeAccessor
    from pyphocorehelpers.indexing_helpers import PhoDataframeAccessor
    
    """
   
    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._df = pandas_obj

    @staticmethod
    def _validate(obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('neuron_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"object must be a pandas Dataframe but is of type: {type(obj)}!\nobj: {obj}")


    def constrain_df_cols(self, should_drop_constrained_columns: bool=True, **constraining_kwargs) -> pd.DataFrame:
        """ 
        from neuropy.utils.indexing_helpers import NeuroPyDataframeAccessor
        
        filtered_single_FAT_df: pd.DataFrame = single_FAT_df.pho.constrain_df_cols(data_grain='per_time_bin', decoder_identifier='pseudo2D', masked_time_bin_fill_type=['ignore'], trained_compute_epochs='laps', known_named_decoding_epochs_type=['laps']) # long_RL=0, short_LR=0, short_RL=0
        filtered_single_FAT_df

        """
        _out_df: pd.DataFrame = deepcopy(self._df)
        for col_name, val in constraining_kwargs.items():
            if isinstance(val, (list, tuple, set)):
                _out_df = _out_df[_out_df[col_name].isin(val)]
            else:
                _out_df = _out_df[_out_df[col_name] == val]
                if should_drop_constrained_columns:
                    ## only drop columns when the value was constrained to a single value
                    _out_df.drop(columns=[col_name], inplace=True)
                    
        # END for col_...
        return _out_df


    ## Pandas DataFrame helpers:
    def partition(self, partitionColumn: str) -> Tuple[NDArray, NDArray]:
        """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
        """
        return partition(df=self._df, partitionColumn=partitionColumn) 


    def partition_df(self, partitionColumn: str) -> Tuple[NDArray, List[pd.DataFrame]]:
        """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
        USEFUL NOTE: to get a dict, do `partitioned_dfs = dict(zip(*partition_df(spikes_df, partitionColumn='new_epoch_IDX')))`
        """
        return partition_df(df=self._df, partitionColumn=partitionColumn) 


    # def partition_df_dict(self, partitionColumn: str) -> Dict[Any, pd.DataFrame]:
    #     """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
    #     """
    #     return partition_df_dict(df=self._df, partitionColumn=partitionColumn) 
    

    def partition_df_dict(self, partitionColumn: Union[str, Tuple[str]]) -> Dict[Any, pd.DataFrame]:
        """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
        """
        if isinstance(partitionColumn, str):
            return partition_df_dict(df=self._df, partitionColumn=partitionColumn) 
        else:
            ## a list of columns to partition on, building a nested dictionary
            for a_partition_column in partitionColumn:
                a_df_dict: Dict[str, pd.DataFrame] = self._df.pho.partition_df_dict(partitionColumn='known_named_decoding_epochs_type')





            

# ==================================================================================================================== #
# Discrete Bins/Binning                                                                                                #
# ==================================================================================================================== #

def get_bin_centers(bin_edges):
    """ For a series of 1D bin edges given by bin_edges, returns the center of the bins. Output will have one less element than bin_edges. """
    return (bin_edges[:-1] + np.diff(bin_edges) / 2.0)
    
def get_bin_edges(bin_centers):
    """ For a series of 1D bin centers given by bin_centers, returns the edges of the bins. Output will have one more element than bin_centers
        Reciprocal of get_bin_centers(bin_edges)
    """
    bin_width = float((bin_centers[1] - bin_centers[0]))
    half_bin_width = bin_width / 2.0 # TODO: assumes fixed bin width
    bin_start_edges = bin_centers - half_bin_width
    last_edge_bin = bin_centers[-1] + half_bin_width # the last edge bin is one half_bin_width beyond the last bin_center
    out = bin_start_edges.tolist()
    out.append(last_edge_bin) # append the last_edge_bin to the bins.
    return np.array(out)

            
def compute_position_grid_size(*any_1d_series, num_bins:tuple):
    """  Computes the required bin_sizes from the required num_bins (for each dimension independently)
    Usage:
    out_grid_bin_size, out_bins, out_bins_infos = compute_position_grid_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64))
    active_grid_bin = tuple(out_grid_bin_size)
    print(f'active_grid_bin: {active_grid_bin}') # (3.776841861770752, 1.043326930905373)
    """
    from neuropy.utils.mixins.binning_helpers import compute_spanning_bins
    
    assert (len(any_1d_series)) == len(num_bins), f'(len(other_1d_series)) must be the same length as the num_bins tuple! But (len(other_1d_series)): {(len(any_1d_series))} and len(num_bins): {len(num_bins)}!'
    num_series = len(num_bins)
    out_bins = []
    out_bins_info = []
    out_bin_grid_step_size = np.zeros((num_series,))

    for i in np.arange(num_series):
        xbins, xbin_info = compute_spanning_bins(any_1d_series[i], num_bins=num_bins[i])
        out_bins.append(xbins)
        out_bins_info.append(xbin_info)
        out_bin_grid_step_size[i] = xbin_info.step

    return out_bin_grid_step_size, out_bins, out_bins_info


def debug_print_1D_bin_infos(bins, label='bins'):
    """ prints info about the 1D bins provided 
    Usage:
        debug_print_1D_bin_infos(time_window_centers, label='time_window_centers')
        >> time_window_centers: [1211.71 1211.96 1212.21 ... 2076.96 2077.21 2077.46], count: 3464, start: 1211.7133460667683, end: 2077.4633460667683, unique_steps: [0.25]
    """
    print(f'{label}: {bins}, count: {len(bins)}, start: {bins[0]}, end: {bins[-1]}, unique_steps: {np.unique(np.diff(bins))}')


# ==================================================================================================================== #
# 2D Grids/Gridding                                                                                                          #
# ==================================================================================================================== #
RowColTuple = namedtuple('RowColTuple', 'num_rows num_columns')
PaginatedGridIndexSpecifierTuple = namedtuple('PaginatedGridIndexSpecifierTuple', 'linear_idx row_idx col_idx data_idx')
RequiredSubplotsTuple = namedtuple('RequiredSubplotsTuple', 'num_required_subplots num_columns num_rows combined_indicies')

# @function_attributes(short_name='compute_paginated_grid_config', tags=['page','grid','config'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-30 16:47')
def compute_paginated_grid_config(num_required_subplots, max_num_columns, max_subplots_per_page=None, data_indicies=None, last_figure_subplots_same_layout=True, debug_print=False):
    """ Fills row-wise first, and constrains the subplots values to just those that you need
    Args:
        num_required_subplots ([type]): [description]
        max_num_columns ([type]): [description]
        max_subplots_per_page ([type]): If None, pagination is effectively disabled and all subplots will be on a single page.
        data_indicies ([type], optional): your indicies into your original data that will also be accessible in the main loop. Defaults to None.
        last_figure_subplots_same_layout (bool): if True, the last page has the same number of items (same # columns and # rows) as the previous (full/complete) pages.
        
        
    Example:
        from pyphocorehelpers.indexing_helpers import compute_paginated_grid_config
        subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=subplots.num_columns, max_subplots_per_page=max_subplots_per_page, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=last_figure_subplots_same_layout)
        num_pages = len(included_combined_indicies_pages)
    
    """
    def _compute_subplots_grid_layout(num_page_required_subplots: int, page_max_num_columns: int):
        """ For a single page """
        fixed_columns = min(page_max_num_columns, num_page_required_subplots) # if there aren't enough plots to even fill up a whole row, reduce the number of columns
        needed_rows = int(np.ceil(num_page_required_subplots / fixed_columns))
        return RowColTuple(needed_rows, fixed_columns)
    
    def _compute_num_subplots(num_required_subplots: int, max_num_columns: int, data_indicies=None):
        """Computes the RequiredSubplotsTuple from the required number of subplots and the max_num_columns. We start in row[0] and begin filling to the right until we exceed max_num_columns. To avoid going over, we add a new row and continue from there.
        """
        linear_indicies = np.arange(num_required_subplots)
        if data_indicies is None:
            data_indicies = np.arange(num_required_subplots) # the data_indicies are just the same as the lienar indicies unless otherwise specified
        (total_needed_rows, fixed_columns) = _compute_subplots_grid_layout(num_required_subplots, max_num_columns) # get the result for a single page before moving on
        all_row_column_indicies = np.unravel_index(linear_indicies, (total_needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
        all_combined_indicies = [PaginatedGridIndexSpecifierTuple(linear_indicies[i], all_row_column_indicies[0][i], all_row_column_indicies[1][i], data_indicies[i]) for i in np.arange(len(linear_indicies))]
        return RequiredSubplotsTuple(num_required_subplots, fixed_columns, total_needed_rows, all_combined_indicies)

    subplot_no_pagination_configuration = _compute_num_subplots(num_required_subplots, max_num_columns=max_num_columns, data_indicies=data_indicies)
    
    # once we have the result for a single page, we paginate it using the chunks function to easily separate it into pages.
    if max_subplots_per_page is None:
        max_subplots_per_page = num_required_subplots # all subplots must fit on a single page.
    included_combined_indicies_pages = [list(chunk) for chunk in chunks(subplot_no_pagination_configuration.combined_indicies, max_subplots_per_page)]
    
    if last_figure_subplots_same_layout:
        page_grid_sizes = [RowColTuple(subplot_no_pagination_configuration.num_rows, subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]
    else:
        # If it isn't required to have the same layout as the previous (full) pages, recompute the correct number of columns for just this page. This deals with the case when not even a full row is filled.
        page_grid_sizes = [_compute_subplots_grid_layout(len(a_page), subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]

    if debug_print:
        print(f'page_grid_sizes: {page_grid_sizes}')
    return subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes


# ==================================================================================================================== #
# Pages/Pagination                                                                                                     #
# ==================================================================================================================== #

@define(slots=False)
class Paginator:
    """ 2023-05-02 - helper that allows easily creating paginated data either for batch or realtime usage. 

    Independent of any plotting technology. Just meant to hold and paginate the data.
    
    
    TODO 2023-05-02 - See also:
    ## paginated outputs for shared cells
    included_unit_indicies_pages = [[curr_included_unit_index for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in v] for page_idx, v in enumerate(included_combined_indicies_pages)] # a list of length `num_pages` containing up to 10 items

    # Can build a list of keyword arguments that will be provided to the function of interest ahead of time
    paginated_shared_cells_kwarg_list = [dict(included_unit_neuron_IDs=curr_included_unit_indicies,
        active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test', display_fn_name='batch_plot_test', plot_result_set='shared', page=f'{page_idx+1}of{num_pages}', aclus=f"{curr_included_unit_indicies}"),
        fignum=f'shared_{page_idx}', fig_idx=page_idx, n_max_page_rows=n_max_page_rows) for page_idx, curr_included_unit_indicies in enumerate(included_unit_indicies_pages)]

    Example:
    
        from pyphocorehelpers.indexing_helpers import Paginator
        
        ## Provide a tuple or list containing equally sized sequences of items:
        sequencesToShow = (rr_aclus, rr_laps, rr_replays)
        a_paginator = Paginator.init_from_data(sequencesToShow, max_num_columns=1, max_subplots_per_page=20, data_indicies=None, last_figure_subplots_same_layout=False)
        # If a paginator was constructed with `sequencesToShow = (rr_aclus, rr_laps, rr_replays)`, then:
        included_page_data_indicies, included_page_data_items = a_paginator.get_page_data(page_idx=1)
        curr_page_rr_aclus, curr_page_rr_laps, curr_page_rr_replays = included_page_data_items

    Extended Example:
        from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget
        a_paginator_controller_widget = PaginationControlWidget(n_pages=a_paginator.num_pages)

    """
    sequencesToShow: tuple
    subplot_no_pagination_configuration: RequiredSubplotsTuple
    included_combined_indicies_pages: list[list[PaginatedGridIndexSpecifierTuple]]
    page_grid_sizes: list[RowColTuple]

    nItemsToShow: int = field()
    num_pages: int = field()

    ## Computed properties:
    @property
    def num_items_per_page(self):
        """The number of items displayed on each page (one number per page).
        e.g. array([20, 20, 20, 20, 20,  8])
        """
        return np.array([(num_rows * num_columns) for (num_rows, num_columns) in self.page_grid_sizes])

    @property
    def max_num_items_per_page(self):
        """The number of items on the page with the maximum number of items.
        e.g. 20
        """
        return np.max(self.num_items_per_page)

    @classmethod
    def init_from_data(cls, sequencesToShow, max_num_columns=1, max_subplots_per_page=20, data_indicies=None, last_figure_subplots_same_layout=False):
        """ creates a Paginator object from a tuple of equal length sequences using `compute_paginated_grid_config`"""
        nItemsToShow  = len(sequencesToShow[0])
        subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nItemsToShow, max_num_columns=max_num_columns, max_subplots_per_page=max_subplots_per_page, data_indicies=data_indicies, last_figure_subplots_same_layout=last_figure_subplots_same_layout)
        num_pages = len(included_combined_indicies_pages)
        
        # Build a reverse index from (page_idx: int, a_linear_index: int) -> data_index: int

        return cls(sequencesToShow=sequencesToShow, subplot_no_pagination_configuration=subplot_no_pagination_configuration, included_combined_indicies_pages=included_combined_indicies_pages, page_grid_sizes=page_grid_sizes, nItemsToShow=nItemsToShow, num_pages=num_pages)


    def get_page_data(self, page_idx: int):
        """ 
        Usage:
            # If a paginator was constructed with `sequencesToShow = (rr_aclus, rr_laps, rr_replays)`, then:
            included_page_data_indicies, included_page_data_items = a_paginator.get_page_data(page_idx=0)
            curr_page_rr_aclus, curr_page_rr_laps, curr_page_rr_replays = included_page_data_items

        """
        ## paginated outputs for shared cells
        included_page_data_indicies = np.array([curr_included_data_index for (a_linear_index, curr_row, curr_col, curr_included_data_index) in self.included_combined_indicies_pages[page_idx]]) # a list of the data indicies on this page
        included_page_data_items = tuple([safe_numpy_index(a_seq, included_page_data_indicies) for a_seq in self.sequencesToShow])
        # included_data_indicies_pages = [[curr_included_unit_index for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in v] for page_idx, v in enumerate(self.included_combined_indicies_pages)] # a list of length `num_pages` containing up to 10 items
        return included_page_data_indicies, included_page_data_items

    # def on_page_change(self, page_idx: int, page_contents_items):
    # 	""" called when the page changes. Iterates through page_contents_items to get all the appropriate contents for that page. """
    # 	sequences_subset = []

    # 	for (a_linear_index, curr_row, curr_col, curr_included_data_index) in page_contents_items:
    # 		for a_seq in sequencesToShow
    # 			a_seq[curr_included_data_index]


# ==================================================================================================================== #
# Filling sentinal values with their adjacent values                                                                   #
# ==================================================================================================================== #

def np_ffill_1D(arr: np.ndarray, debug_print=False):
    '''  'forward-fill' the nan values in array arr. 
    By that I mean replacing each nan value with the nearest valid value from the left.
    row-wise by default
    
    Example:

    Input:
        array([[  5.,  nan,  nan,   7.,   2.],
        [  3.,  nan,   1.,   8.,  nan],
        [  4.,   9.,   6.,  nan,  nan]])
       
       
    Desired Solution:
        array([[  5.,   5.,   5.,  7.,  2.],
        [  3.,   3.,   1.,  8.,  8.],
        [  4.,   9.,   6.,  6.,  6.]])
       
       
    Most efficient solution from StackOverflow as timed by author of question: https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    Solution provided by Divakar.
    
    '''
    did_pad_1D_array = False
    if (arr.ndim < 2):
        ## Pad a 1D (N,) array to (N,1) to work just like the 2D arrays.
        if debug_print:
            print(f'np_ffill_1D(arr): (arr.ndim: {arr.ndim} < 2), adding dimension...')
        # arr = arr[:, np.newaxis] # .shape: (12100, 1)
        arr = arr[np.newaxis,:] # .shape: (1, 12100) 
        did_pad_1D_array = True # indicate that we modified the 1D array
        if debug_print:
            print(f'\t new dim: {arr.ndim}, np.shape(arr): {np.shape(arr)}.')
        assert arr.ndim == 2
    mask = np.isnan(arr)
    if debug_print:
        print(f'\t np.shape(mask): {np.shape(mask)}.')

    idx = np.where(~mask, np.arange(mask.shape[1]), 0) # chooses values from the ascending value range `np.arange(mask.shape[1])` when arr is *not* np.nan, and zeros when it is nan (idx.shape: 1D: (12100,), 2D: )
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    # output should be the same shape as the input
    if debug_print:
        print(f'\t pre-squeezed output shape: {out.shape}.')

    if did_pad_1D_array:
        out = np.squeeze(out)
        if debug_print:
            print(f'\t final 1D array restored output shape: {out.shape}.')
    return out

def np_bfill_1D(arr: np.ndarray):
    """ backfills the np.nan values instead of forward filling them 
    Simple solution for bfill provided by financial_physician in comment below
    """
    return np_ffill_1D(arr[:, ::-1])[:, ::-1]
