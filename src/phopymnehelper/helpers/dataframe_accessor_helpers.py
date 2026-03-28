from collections import namedtuple
from copy import deepcopy
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
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


    def apply_mask(self, mask_col: str, value_cols: Sequence[str], *, copy: bool = True) -> pd.DataFrame:
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


    def mask_by_intervals(self, mask_bad_intervals_df: pd.DataFrame, time_col_name: str = 't', bool_mask_column_name: str = 'is_bad_motion', 
            intervals_start_col_name: str='onset', intervals_end_col_name: str='onset_end',
            ):
        """ 
        time_col_name: point-like time column to be checked to see if overlaps the intervals in mask_bad_intervals_df
        bool_mask_column_name: column to be added
        """
        import polars as pl

        lf = pl.from_pandas(self._obj).lazy()
        iv = pl.from_pandas(mask_bad_intervals_df).select(intervals_start_col_name, intervals_end_col_name).lazy()

        bad_t = (
            lf.join(iv, how="cross")
            .filter((pl.col(time_col_name) >= pl.col(intervals_start_col_name)) & (pl.col(time_col_name) <= pl.col(intervals_end_col_name)))
            .select(time_col_name)
            .unique()
            .with_columns(pl.lit(True).alias(bool_mask_column_name))
        )

        df = lf.join(bad_t, on=time_col_name, how="left").with_columns(
            pl.col(bool_mask_column_name).fill_null(False)
        ).collect()

        self._obj = df.to_pandas()  # if you need pandas again
        return self._obj