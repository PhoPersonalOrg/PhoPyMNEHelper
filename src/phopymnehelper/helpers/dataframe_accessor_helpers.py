from collections import namedtuple
from copy import deepcopy
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
import nptyping as ND
from nptyping import NDArray
import numpy as np
import pandas as pd
import polars as pl

class CommonDataFrameAccessorMixin(object):
    """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals 
    
    from phopymnehelper.helpers.dataframe_accessor_helpers import CommonDataFrameAccessorMixin

    """

    # dt_col_names = ['recording_datetime', 'recording_day_date']
    # timestamp_column_names = ['created_at', 'first_timestamp', 'last_timestamp']
    # timestamp_dt_column_names = ['created_at_dt', 'first_timestamp_dt', 'last_timestamp_dt']
    # timestamp_rel_column_names = ['created_at_rel', 'first_timestamp_rel', 'last_timestamp_rel']

    _required_column_names = ['start', 'stop', 'label', 'duration']

    def __init__(self, pandas_obj):      
        pandas_obj = self._validate(pandas_obj)
        self._obj = pandas_obj

    @classmethod
    def _validate(cls, obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('neuron_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """       
        return obj # important! Must return the modified obj to be assigned (since its columns were altered by renaming


    @property
    def extra_data_column_names(self):
        """Any additional columns in the dataframe beyond those that exist by default. """
        return list(set(self._obj.columns) - set(self._required_column_names))

    @property
    def extra_data_dataframe(self) -> pd.DataFrame:
        """The subset of the dataframe containing additional information in its columns beyond that what is required. """
        return self._obj[self.extra_data_column_names]

    # def as_array(self) -> NDArray:
    #     return self._obj[["start", "stop"]].to_numpy()


    def adding_or_updating_metadata(self, **metadata_update_kwargs) -> pd.DataFrame:
        """ updates the dataframe's `df.attrs` dictionary metadata, building it as a new dict if it doesn't yet exist

        Usage:
            from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch

            maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
            maze_epochs_df = maze_epochs_df.epochs.adding_or_updating_metadata(train_test_period='train')
            maze_epochs_df

        """
        ## Add the metadata:
        if self._obj.attrs is None:
            self._obj.attrs = {} # create a new metadata dict on the dataframe
        self._obj.attrs.update(**metadata_update_kwargs)
        return self._obj



# uv add polars[all]


@pd.api.extensions.register_dataframe_accessor("masked_df")
class MaskedValidDataFrameAccessor:
    """DataFrame accessor: replace values with ``pd.NA`` in selected columns where a boolean mask column is False.

    Where ``mask_col`` is True, values in ``value_cols`` are kept. Where False, those cells become ``pd.NA``.
    Rows with NA in ``mask_col`` are treated as False (masked). Non-boolean ``mask_col`` dtypes are cast with
    ``astype(bool)``. All other columns are unchanged.

    Usage::

        import pandas as pd
        from phopymnehelper.helpers.dataframe_accessor_helpers import MaskedValidDataFrameAccessor

        out = df.masked_df.apply_mask("is_valid", ["x", "y"])
        out_shallow = df.masked_df.apply_mask("is_valid", ["x", "y"], copy=False)

    Args:
        copy: If True (default), use ``DataFrame.copy(deep=True)`` before applying the mask. If False, use
            ``copy(deep=False)`` so unmodified columns may share memory with the original until written.
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        if not isinstance(pandas_obj, pd.DataFrame):
            raise TypeError("masked_df accessor requires a pandas.DataFrame")
        self._obj = pandas_obj
        self._obj = self.adding_or_updating_metadata(mask_col_names=['is_valid'])

    @property
    def mask_col_names(self) -> List[str]:
        """Get or set the list of mask column names stored in DataFrame.attrs['mask_col_names']"""
        return self._obj.attrs.get('mask_col_names', None)
    @mask_col_names.setter
    def mask_col_names(self, value: List[str]):
        """Set the list of mask column names in DataFrame.attrs['mask_col_names']"""
        self._obj.attrs['mask_col_names'] = value


    def get_masked(self, copy: bool = True) -> pd.DataFrame:
        """ 
        Usage:
            from phopymnehelper.helpers.dataframe_accessor_helpers import MaskedValidDataFrameAccessor

            masked_detailed_eeg_df: pd.DataFrame = detailed_eeg_df.masked_df.apply_mask(mask_col='is_valid', value_cols=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
                                                                copy=True)
            masked_detailed_eeg_df

        """
        active_mask_column_names = self.mask_col_names
        
        df = self._obj
        available = set(df.columns)
        # Use all active mask columns as the mask; if none, return unchanged or fully masked
        if not active_mask_column_names or active_mask_column_names is None:
            # No mask columns found, treat all values as valid (masked = True)
            mask = pd.Series(True, index=df.index)
        else:
            # Ensure all mask columns exist
            missing_masks = [col for col in active_mask_column_names if col not in available]
            if missing_masks:
                raise KeyError(f"mask_col(s) {missing_masks!r} not found; available: {sorted(available)}")
            # Combine all mask columns with logical AND (if all True then valid)
            masks = [df[col].astype("boolean").fillna(False).astype(bool) if not pd.api.types.is_bool_dtype(df[col]) else df[col].fillna(False) for col in active_mask_column_names]
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m

        # Mask value_cols as usual
        value_cols_list = list(df.columns.difference(active_mask_column_names, sort=False))  # Mask all except mask cols by default
        missing = [c for c in value_cols_list if c not in available]
        if missing:
            raise KeyError(f"value_cols not in DataFrame: {missing}")
        if any(col in value_cols_list for col in (active_mask_column_names or [])):
            raise ValueError("mask_col must not appear in value_cols")
        if not value_cols_list:
            return df.copy(deep=copy)
        out = df.copy(deep=copy)
        out[value_cols_list] = df[value_cols_list].where(mask, pd.NA)
        return out



    def apply_mask(self, mask_col: str, value_cols: Sequence[str], *, copy: bool = True) -> pd.DataFrame:
        """ 
        Usage:
            from phopymnehelper.helpers.dataframe_accessor_helpers import MaskedValidDataFrameAccessor

            masked_detailed_eeg_df: pd.DataFrame = detailed_eeg_df.masked_df.apply_mask(mask_col='is_valid', value_cols=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
                                                                copy=True)
            masked_detailed_eeg_df

        """

        df = self._obj
        available = set(df.columns)
        if mask_col not in available:
            raise KeyError(f"mask_col {mask_col!r} not found; available: {sorted(available)}")
        value_cols_list = list(value_cols)
        missing = [c for c in value_cols_list if c not in available]
        if missing:
            raise KeyError(f"value_cols not in DataFrame: {missing}")
        if mask_col in value_cols_list:
            raise ValueError("mask_col must not appear in value_cols")
        if not value_cols_list:
            return df.copy(deep=copy)
        m = df[mask_col]
        if not pd.api.types.is_bool_dtype(m):
            m = m.astype("boolean").fillna(False).astype(bool)
        else:
            m = m.fillna(False)
        out = df.copy(deep=copy)
        out[value_cols_list] = df[value_cols_list].where(m, pd.NA)
        return out


    def masking_by_intervals(self, mask_bad_intervals_df: pd.DataFrame, time_col_name: str = 't', bool_mask_column_name: str = 'is_bad_motion', intervals_start_col_name: str = 'onset', intervals_end_col_name: str = 'onset_end') -> pd.DataFrame:
        """Flag rows whose ``time_col_name`` falls in any row of ``mask_bad_intervals_df`` (inclusive endpoints on start/end columns).

        Returns a new DataFrame with ``bool_mask_column_name`` (bool). Assign the result, for example
        ``df = df.masked_df.mask_by_intervals(intervals_df)``; the original frame is not updated.

        Args:
            mask_bad_intervals_df: DataFrame with interval bounds (default columns ``onset``, ``onset_end``).
            time_col_name: Column of point timestamps compared to each interval (closed interval).
            bool_mask_column_name: Name of the new boolean column; if it already exists on the input, it is dropped and recomputed.
            intervals_start_col_name / intervals_end_col_name: Column names for interval bounds on ``mask_bad_intervals_df``.

        Usage:
            from phopymnehelper.helpers.dataframe_accessor_helpers import MaskedValidDataFrameAccessor

            detailed_eeg_df = detailed_eeg_df.masked_df.masking_by_intervals(mask_bad_intervals_df=mask_bad_intervals_df, time_col_name='t', bool_mask_column_name='is_bad_motion',
                                                                        intervals_start_col_name='onset', intervals_end_col_name='onset_end')
            detailed_eeg_df['is_valid'] = np.logical_not(detailed_eeg_df['is_bad_motion'])
            detailed_eeg_df

        """
        base = self._obj.copy()
        available = set(base.columns)
        if time_col_name not in available:
            raise KeyError(f"time_col_name {time_col_name!r} not found; available: {sorted(available)}")
        if bool_mask_column_name in available:
            base = base.drop(columns=[bool_mask_column_name])
            available = set(base.columns)
        iv_cols = set(mask_bad_intervals_df.columns)
        if intervals_start_col_name not in iv_cols:
            raise KeyError(f"intervals_start_col_name {intervals_start_col_name!r} not found on mask_bad_intervals_df; have: {sorted(iv_cols)}")
        if intervals_end_col_name not in iv_cols:
            raise KeyError(f"intervals_end_col_name {intervals_end_col_name!r} not found on mask_bad_intervals_df; have: {sorted(iv_cols)}")
        if mask_bad_intervals_df.empty:
            out = base.copy()
            out[bool_mask_column_name] = False
            return out
        pandas_data_columns = list(base.columns)
        index_column_names = [c for c in pl.from_pandas(base, include_index=True).collect_schema().names() if c not in pandas_data_columns]
        lf = pl.from_pandas(base, include_index=True).lazy().with_row_index('__mask_row_id')
        iv = pl.from_pandas(mask_bad_intervals_df).select(intervals_start_col_name, intervals_end_col_name).lazy()
        bad_t = (
            lf.join(iv, how='cross')
            .filter((pl.col(time_col_name) >= pl.col(intervals_start_col_name)) & (pl.col(time_col_name) <= pl.col(intervals_end_col_name)))
            .select('__mask_row_id')
            .unique()
            .with_columns(pl.lit(True).alias(bool_mask_column_name))
        )
        out_pl = cast(pl.DataFrame, lf.join(bad_t, on='__mask_row_id', how='left').with_columns(pl.col(bool_mask_column_name).fill_null(False)).drop('__mask_row_id').collect())
        out_pd = out_pl.to_pandas()
        if index_column_names:
            out_pd = out_pd.set_index(index_column_names)
        return out_pd


    def adding_or_updating_metadata(self, **metadata_update_kwargs) -> pd.DataFrame:
        """ updates the dataframe's `df.attrs` dictionary metadata, building it as a new dict if it doesn't yet exist

        Usage:
            from neuropy.core.epoch import Epoch, EpochsAccessor, NamedTimerange, ensure_dataframe, ensure_Epoch

            maze_epochs_df = deepcopy(curr_active_pipeline.sess.epochs).to_dataframe()
            maze_epochs_df = maze_epochs_df.epochs.adding_or_updating_metadata(train_test_period='train')
            maze_epochs_df

        """
        ## Add the metadata:
        if self._obj.attrs is None:
            self._obj.attrs = {} # create a new metadata dict on the dataframe
        self._obj.attrs.update(**metadata_update_kwargs)
        return self._obj
