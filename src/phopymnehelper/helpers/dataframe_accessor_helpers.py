from collections import namedtuple
from copy import deepcopy
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
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
    """DataFrame accessor: replace values with ``pd.NA`` where boolean mask columns are False.

    Configuration is stored in ``DataFrame.attrs['mask_col_to_value_cols']``: a dict mapping each
    ``mask_col`` name to the list of ``value_cols`` masked by that column. Where the mask is True,
    values are kept; where False (or NA, treated as False), cells become ``pd.NA``. Non-boolean mask
    dtypes are converted via nullable boolean then ``astype(bool)``. Columns not listed for any mask
    are unchanged. Multiple entries affecting the same value column apply in sorted key order, equivalent
    to AND-ing the row masks.

    Usage::

        out = df.masked_df.add_masking_column("is_valid", ["x", "y"])
        out_shallow = df.masked_df.add_masking_column("is_valid", ["x", "y"], copy=False)
        out = df.masked_df.get_masked()

    Usage:

        from phopymnehelper.helpers.dataframe_accessor_helpers import MaskedValidDataFrameAccessor

        ## INPUTS: detailed_eeg_df, mask_bad_intervals_df
        detailed_eeg_df = detailed_eeg_df.masked_df.masking_by_intervals(mask_bad_intervals_df=mask_bad_intervals_df, time_col_name='t', bool_mask_column_name='is_bad_motion',
                                                                    intervals_start_col_name='onset', intervals_end_col_name='onset_end')
        detailed_eeg_df.masked_df.add_masking_column(mask_col='is_valid', value_cols=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'])
        masked_detailed_eeg_df: pd.DataFrame = detailed_eeg_df.masked_df.get_masked(copy=True)


    Legacy DataFrames may still have ``attrs['mask_col_names']``; on first read that is migrated to
    ``mask_col_to_value_cols`` (each mask key gets all non-mask columns, reproducing the former global AND).

    Args:
        copy: If True (default), use ``DataFrame.copy(deep=True)`` before applying masks. If False, use
            ``copy(deep=False)`` so unmodified columns may share memory with the original until written.



    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        if not isinstance(pandas_obj, pd.DataFrame):
            raise TypeError("masked_df accessor requires a pandas.DataFrame")
        self._obj = pandas_obj
        if self._obj.attrs is None:
            self._obj.attrs = {}
        if self._obj.attrs.get("mask_col_to_value_cols", None) is None:
            self._obj.attrs["mask_col_to_value_cols"] = {}


    @property
    def mask_col_to_value_cols(self) -> Dict[str, List[str]]:
        """Mapping ``mask_col -> value_cols`` in ``DataFrame.attrs['mask_col_to_value_cols']``. Migrates from legacy ``mask_col_names`` when that key is absent."""
        if self._obj.attrs is None:
            self._obj.attrs = {}
        attrs = self._obj.attrs
        if "mask_col_to_value_cols" not in attrs:
            old_names = attrs.get("mask_col_names") or []
            if old_names:
                df = self._obj
                mask_set = set(old_names)
                value_cols = [c for c in df.columns if c not in mask_set]
                attrs["mask_col_to_value_cols"] = {m: list(value_cols) for m in old_names}
            else:
                attrs["mask_col_to_value_cols"] = {}
        raw = attrs["mask_col_to_value_cols"]
        return raw if raw is not None else {}

    @mask_col_to_value_cols.setter
    def mask_col_to_value_cols(self, value: Dict[str, List[str]]) -> None:
        if self._obj.attrs is None:
            self._obj.attrs = {}
        self._obj.attrs["mask_col_to_value_cols"] = value


    def get_masked(self, copy: bool = True) -> pd.DataFrame:
        """Return a copy of the frame with ``pd.NA`` applied per ``mask_col_to_value_cols`` entries."""
        mapping = self.mask_col_to_value_cols
        df = self._obj
        if not mapping:
            return df.copy(deep=copy)
        available = set(df.columns)
        out = df.copy(deep=copy)
        for mask_col in sorted(mapping.keys()):
            value_cols_list = list(mapping[mask_col])
            if not value_cols_list:
                continue
            if mask_col not in available:
                raise KeyError(f"mask_col {mask_col!r} not found; available: {sorted(available)}")
            missing = [c for c in value_cols_list if c not in available]
            if missing:
                raise KeyError(f"value_cols not in DataFrame: {missing}")
            if mask_col in value_cols_list:
                raise ValueError("mask_col must not appear in value_cols")
            col_series = df[mask_col]
            if not pd.api.types.is_bool_dtype(col_series):
                mask = col_series.astype("boolean").fillna(False).astype(bool)
            else:
                mask = col_series.fillna(False)
            out[value_cols_list] = out[value_cols_list].where(mask, pd.NA)
        return out



    def add_masking_column(self, mask_col: str, value_cols: Sequence[str]): # *, copy: bool = True
        """Register ``mask_col`` -> ``value_cols`` in attrs and return ``get_masked()``."""
        df = self._obj
        available = set(df.columns)
        if mask_col not in available:
            print(f"mask_col '{mask_col}' not found; available columns: {sorted(available)}")
            df[mask_col] = True ## default to True
            # raise KeyError(f"mask_col {mask_col!r} not found; available: {sorted(available)}")
        value_cols_list = list(value_cols)
        missing = [c for c in value_cols_list if c not in available]
        if missing:
            raise KeyError(f"value_cols not in DataFrame: {missing}")
        if mask_col in value_cols_list:
            raise ValueError("mask_col must not appear in value_cols")
        new_map = dict(self.mask_col_to_value_cols)
        new_map[mask_col] = value_cols_list
        self.mask_col_to_value_cols = new_map
        # return self

        # return self.get_masked(copy=copy)


    def masking_by_intervals(self, mask_bad_intervals_df: pd.DataFrame, time_col_name: str = 't', bool_mask_column_name: str = 'is_bad_motion',
                 intervals_start_col_name: str = 'onset', intervals_end_col_name: str = 'onset_end',
                 add_final_valid_col_name: str = 'is_valid') -> pd.DataFrame:
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
        if (add_final_valid_col_name is not None):
            out_pd[add_final_valid_col_name] = np.logical_not(out_pd[bool_mask_column_name])

        # if update_masking_info:
            # out_pd: pd.DataFrame = out_pd.masked_df.add_masking_column(mask_col='is_valid', value_cols=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
            #                                                     copy=False)

            # masked_detailed_eeg_df.masked_df.get_masked()

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
