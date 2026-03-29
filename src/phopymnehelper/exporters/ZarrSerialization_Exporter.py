from typing import List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import xarray as xr
import zarr
import numcodecs

# @function_attributes(short_name=None, tags=['serialization'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-10-01 16:58', related_items=[])
class ZarrSerialization:
    """ Exports to Zarr for XArrays

    import xarray as xr
    import zarr
    import numcodecs
    import numpy as np
    from pathlib import Path
    from typing import Union

    from phopymnehelper.exporters.ZarrSerialization_Exporter import ZarrSerialization
    

    cog_categorical_cats = ['cog_bad', 'cog_poor', 'cog_okay', 'cog_good', 'cog_great']

    day_status_dict = {
        '2025-09-19/01-54-39': 'cog_poor',
        '2025-09-18/15-23-08': 'cog_poor',
        '2025-09-22/21-35-47': 'cog_okay',
        '2025-09-19/20-51-18': 'cog_good',
        '2025-09-18/03-18-42': 'cog_great',
        '2025-09-22/21-35-47': 'cog_bad',
        '2025-09-21/08-55-41': 'cog_bad',
    }


    zarr_out_path = Path("2025-09-30_all_sessions.zarr")

    zarr_out_path = ZarrSerialization.save_sessions_as_zarr(active_only_out_eeg_raws=active_only_out_eeg_raws, results=results, day_status_dict=day_status_dict, out_path=zarr_out_path)

    ## Immediately re-load from file
    all_data = ZarrSerialization.load_sessions_from_zarr(zarr_out_path, verbose=True)

    """
    @classmethod
    def load_sessions_from_zarr(cls, zarr_path: Union[str, Path], verbose: bool = False) -> xr.Dataset:
        zarr_path = Path(zarr_path)
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found at {zarr_path}")

        store = zarr.DirectoryStore(zarr_path.as_posix())
        root = zarr.open(store, mode="r")
        session_keys = list(root.group_keys())
        if verbose:
            print("zarr groups:", session_keys)
        if not session_keys:
            raise ValueError(f"No session groups found in the Zarr store at {zarr_path}")

        da_list: List[xr.DataArray] = []
        for key in session_keys:
            try:
                ds = xr.open_zarr(store, group=key, consolidated=True)
            except Exception as e:
                if verbose: print(f"skipping group {key}: open_zarr error: {e}")
                continue

            # choose the DataArray: prefer 'Sxx' but fall back to the first data_var
            if "Sxx" in ds.data_vars:
                da = ds["Sxx"]
                picked_name = "Sxx"
            elif len(ds.data_vars) == 1:
                picked_name = list(ds.data_vars)[0]
                da = ds[picked_name]
                if verbose: print(f"group {key}: no 'Sxx' var, picked '{picked_name}'")
            else:
                if verbose: print(f"skipping group {key}: no Sxx and multiple data_vars: {list(ds.data_vars)}")
                continue

            # normalize name to 'Sxx' for consistency
            if da.name != "Sxx":
                da = da.rename("Sxx")

            session_key = ds.attrs.get("session_key", key)
            cognitive_status = da.attrs.get("cognitive_status", "unknown")

            # Expand with session dim labeled by session_key
            da = da.expand_dims({"session": [session_key]})

            # Assign cognitive_status as a coordinate on session dim
            da = da.assign_coords(cognitive_status=("session", [cognitive_status]))

            da_list.append(da)

        if not da_list:
            raise ValueError("No sessions with Sxx (or fallback data-var) found in Zarr store")

        combined = xr.concat(da_list, dim="session", join="outer")

        ds_combined = combined.to_dataset(name="Sxx")

        # session_key coord (same labels as session dim)
        sess_labels = [str(x) for x in ds_combined["session"].values.tolist()]
        ds_combined = ds_combined.assign_coords(session_key=("session", sess_labels))

        # ensure cognitive_status exists as session coord
        if "cognitive_status" not in ds_combined.coords:
            cs_vals = [da.coords.get("cognitive_status").values[0] if "cognitive_status" in da.coords else "unknown" for da in da_list]
            ds_combined = ds_combined.assign_coords(cognitive_status=("session", cs_vals))

        return ds_combined


    @classmethod
    def save_sessions_as_zarr(cls, active_only_out_eeg_raws, results, day_status_dict, out_path="all_sessions.zarr"):
        """ write out the result (spectogram) from each session in the `day_status_dict` to the out_path file

        Usage:

            cog_categorical_cats = ['cog_bad', 'cog_poor', 'cog_okay', 'cog_good', 'cog_great']

            day_status_dict = {
                '2025-09-19/01-54-39': 'cog_poor',
                '2025-09-18/15-23-08': 'cog_poor',
                '2025-09-22/21-35-47': 'cog_okay',
                '2025-09-19/20-51-18': 'cog_good',
                '2025-09-18/03-18-42': 'cog_great',
                '2025-09-22/21-35-47': 'cog_bad',
                '2025-09-21/08-55-41': 'cog_bad',
            }


            out_path = Path("2025-09-29_all_sessions.zarr")
            out_path = save_sessions_as_zarr(active_only_out_eeg_raws=active_only_out_eeg_raws, results=results, day_status_dict=day_status_dict, out_path=out_path)

        """
        compressor = numcodecs.Blosc()
        store = zarr.DirectoryStore(out_path)
        num_sessions: int = len(active_only_out_eeg_raws)

        for idx in np.arange(num_sessions):
            a_raw = active_only_out_eeg_raws[idx]
            a_meas_date = a_raw.info.get('meas_date')
            a_raw_key = a_meas_date.strftime("%Y-%m-%d/%H-%M-%S")
            a_status = day_status_dict.get(a_raw_key, None)
            if not a_status:
                continue

            a_result = results[idx]
            Sxx = a_result['spectogram']['Sxx']
            Sxx = Sxx.assign_attrs(cognitive_status=a_status) # xarray.DataArray - channels: 14, freqs: 513, times: 1116

            ds = Sxx.to_dataset(name='Sxx')
            ds.attrs['session_key'] = a_raw_key

            encoding = {v: {'compressor': compressor} for v in ds.data_vars}
            ds.to_zarr(store, group=a_raw_key, mode='a', encoding=encoding)

        zarr.consolidate_metadata(store)
        return out_path

