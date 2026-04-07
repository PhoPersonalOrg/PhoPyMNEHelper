import time
import hashlib
import re
from datetime import datetime, timezone


import uuid
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

from pathlib import Path
import numpy as np
import pandas as pd

import mne
from mne import set_log_level
from copy import deepcopy
import mne

mne.viz.set_browser_backend("Matplotlib")

# from ..EegProcessing import bandpower
from numpy.typing import NDArray
from phopymnehelper.helpers.dataframe_accessor_helpers import CommonDataFrameAccessorMixin

set_log_level("WARNING")


class MotionData:
    """ Methods related to processing of motion (gyro/accel/magnet/quaternion/etc) data
    from phopymnehelper.motion_data import MotionData
    
    (all_data_MOTION, all_times_MOTION), datasets_MOTION, df_MOTION = flat_data_modality_dict['MOTION']  ## Unpacking

    (active_motion_IDXs, analysis_results_MOTION) = MotionData.preprocess(datasets_MOTION, n_most_recent_sessions_to_preprocess=5)


    """
    @classmethod
    def _perform_process_motion_session(cls, raw_motion: mne.io.Raw):
        """ Called by `preprocess` on each rawMotion session
         
            motion_analysis_results = _perform_process_motion_session(raw_motion)
            motion_analysis_results['quaternion_df'].head()
        """
        motion_analysis_results = {}
        # # Compute quaternions from gyro and accelerometer data
        # motion_analysis_results['quaternion_df'] = cls.compute_quaternions(motion.to_data_frame())

        annots, an_is_moving_annots_df = MotionData.find_high_accel_periods(raw_motion, total_accel_threshold=0.5)
        motion_analysis_results['bad_periods_annotations'] = {'high_accel': annots, 'high_accel_df': an_is_moving_annots_df}
        return motion_analysis_results

    
    @classmethod
    def find_high_accel_periods(cls, a_ds: Union[mne.io.Raw, pd.DataFrame], total_accel_threshold: float = 0.5, should_set_bad_period_annotations: bool=True, minimum_bad_duration: Optional[float]=None, **set_annotations_kwargs) -> Tuple[mne.Annotations, pd.DataFrame]:
        """ finds periods of high acceleration in the dataset and returns annotations for those periods.

        ``minimum_bad_duration`` is in seconds when the epoch ``duration`` column is numeric or ``timedelta64`` (compared via total seconds).

        Usage:
            from phopymnehelper.motion_data import MotionData
            from phopymnehelper.MNE_helpers import MNEHelpers

            motion_widget, motion_track, motion_ds = timeline.get_track_tuple('MOTION_Epoc X Motion')

            total_accel_threshold: float = 0.5

            # a_motion_df: pd.DataFrame = motion_ds.detailed_df.copy() # a_ds.to_data_frame()       
            # a_motion_df: pd.DataFrame = MotionData.compute_rolling_motion_change_detection(a_df=a_motion_df, total_change_threshold=total_accel_threshold, enable_global_normalization=True)
            is_moving_annots, is_moving_annots_df = MotionData.find_high_accel_periods(a_ds=motion_ds.detailed_df, total_change_threshold=total_accel_threshold, should_set_bad_period_annotations=False, minimum_bad_duration=1e-3) # at least 1ms in duration to prevent tons of tiny intervals

            motion_ds.detailed_df

        """
        from phopymnehelper.MNE_helpers import MNEHelpers

        if not isinstance(a_ds, pd.DataFrame):
            meas_date = deepcopy(a_ds.info['meas_date'])
            # a_ds = motion_datasets[-1]
            a_motion_df: pd.DataFrame = a_ds.to_data_frame()
            time_col_names: str = 'time'

        else:
            assert (not should_set_bad_period_annotations), f"should_set_bad_period_annotations is True but a pd.DataFrame was passed instead of a mne.io.Raw object."
            meas_date = set_annotations_kwargs.pop('meas_date', None)

            a_motion_df: pd.DataFrame = a_ds
            time_col_names: str = 't'

        assert time_col_names in a_motion_df.columns

        a_motion_df: pd.DataFrame = cls.compute_rolling_motion_change_detection(a_df=a_motion_df, total_change_threshold=total_accel_threshold, enable_global_normalization=True)

        # annots: mne.Annotations = MNEHelpers.convert_df_with_boolean_col_to_epochs(deepcopy(a_motion_df), is_bad_col_name="is_moving", annotation_description_name="BAD_motion", time_col_names='time', meas_date=meas_date) # , **kwargs
        
        is_moving_annots_df: pd.DataFrame = MNEHelpers.convert_df_with_boolean_col_to_epochs(deepcopy(a_motion_df), is_bad_col_name="is_moving", annotation_description_name="BAD_motion", time_col_names=time_col_names)
        if (minimum_bad_duration is not None) and (minimum_bad_duration > 0.0):
            _dur = is_moving_annots_df['duration']
            _dur_cmp = _dur.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(_dur.dtype) else _dur
            is_moving_annots_df = is_moving_annots_df[_dur_cmp >= minimum_bad_duration].reset_index(drop=True)

        _dur_out = is_moving_annots_df['duration']
        _duration_np = _dur_out.dt.total_seconds().to_numpy(dtype=float) if pd.api.types.is_timedelta64_dtype(_dur_out.dtype) else _dur_out.to_numpy()
        is_moving_annots: mne.Annotations = mne.Annotations(onset=is_moving_annots_df['onset'].to_numpy(), duration=_duration_np, description=is_moving_annots_df['description'].to_numpy(), orig_time=meas_date)


        if should_set_bad_period_annotations:
            ## sets the annotations to the raw object
            a_ds.set_annotations(is_moving_annots, **set_annotations_kwargs)
        
        return is_moving_annots, is_moving_annots_df

        # annots: mne.Annotations = MNEHelpers.convert_df_with_boolean_col_to_epochs(a_rolling_motion_df, is_bad_col_name="is_moving", annotation_description_name="BAD_motion", time_col_names='time', meas_date=meas_date)
        # annots

        # return annots, a_motion_df


    @classmethod
    def compute_rolling_motion_change_detection(cls, a_df: pd.DataFrame, enable_global_normalization:bool=True, total_change_threshold: float=0.5, average_method = np.mean) -> pd.DataFrame:
        """ 

        # a_df: pd.DataFrame = self.o.data.copy()
        from phopymnehelper.motion_data import MotionData

        a_motion_ds = motion_datasets[-1]
        a_motion_df: pd.DataFrame = MotionData.compute_rolling_motion_change_detection(a_df=a_motion_ds.to_data_frame())
        a_motion_df

        
        """
        # active_norm_fn = (lambda x: (x - curr_total_col_global_min)/(curr_total_col_global_max - curr_total_col_global_min))
        def _build_active_norm_fn(curr_global_min: float, curr_global_max: float):
            if (curr_global_min == np.nan) or (curr_global_max == np.nan):
                active_norm_fn = (lambda x: np.abs(x))
            else:
                active_norm_fn = (lambda x: (np.abs(x) - curr_global_min)/(curr_global_max - curr_global_min)) ## normalize column between [0.0, +1.0]
                # active_norm_fn = (lambda x: (2.0*(x - curr_global_min)/(curr_global_max - curr_global_min)) - 1.0) ## normalize column between [-1.0, +1.0]
            return active_norm_fn
        
        col_names = ['AccX', 'AccY', 'AccZ']
        smoothed_col_names = [f'{k}_smooth' for k in col_names]
        diff_col_names = [f'{k}_diff' for k in col_names]
        # global_max_min_dict = {'AccX': (0.0, 1.0), 'AccY': (0.0, 1.0), 'AccZ': (0.0, 1.0)}
        global_max_min_dict = {}


        # _temp_df[smoothed_col_names] = np.zeros_like(_temp_df['AccX'].values()) ## initialize to np.NaN
        # _temp_df[diff_col_names] = np.zeros_like(_temp_df['AccX'].values())

        for a_col_name, a_smoothed_col_name, a_diff_col_name in zip(col_names, smoothed_col_names, diff_col_names):
            if enable_global_normalization:
                if a_col_name not in global_max_min_dict:
                    global_max_min_dict[a_col_name] = (a_df[a_col_name].min(skipna=True), a_df[a_col_name].max(skipna=True))
                else:
                    global_max_min_dict[a_col_name] = (min(global_max_min_dict[a_col_name][0], a_df[a_col_name].min(skipna=True)), max(global_max_min_dict[a_col_name][1], a_df[a_col_name].max(skipna=True)))

            a_df[a_smoothed_col_name] = (a_df[a_col_name] ** 2).apply(average_method)
            if enable_global_normalization:
                if a_smoothed_col_name not in global_max_min_dict:
                    global_max_min_dict[a_smoothed_col_name] = (a_df[a_smoothed_col_name].min(skipna=True), a_df[a_smoothed_col_name].max(skipna=True))
                else:
                    global_max_min_dict[a_smoothed_col_name] = (min(global_max_min_dict[a_smoothed_col_name][0], a_df[a_smoothed_col_name].min(skipna=True)), max(global_max_min_dict[a_smoothed_col_name][1], a_df[a_smoothed_col_name].max(skipna=True)))

                ## normalize column between [-1.0, +1.0]
                # a_df[active_accel_col_names] = a_df[active_accel_col_names].apply(lambda x: 2*(x - x.min())/(x.max() - x.min()) - 1)
                curr_smoothed_col_global_min, curr_smoothed_col_global_max = global_max_min_dict[a_smoothed_col_name]
                # _temp_df[a_smoothed_col_name] = _temp_df[a_smoothed_col_name].apply(lambda x: 2*(x - x.min())/(x.max() - x.min()) - 1)
                a_df[a_smoothed_col_name] = a_df[a_smoothed_col_name].apply(_build_active_norm_fn(curr_global_min=curr_smoothed_col_global_min, curr_global_max=curr_smoothed_col_global_max))                
                # if (curr_smoothed_col_global_min != np.nan) and (curr_smoothed_col_global_max != np.nan):
                #     # _temp_df[a_smoothed_col_name] = _temp_df[a_smoothed_col_name].apply(lambda x: (2*(x - curr_smoothed_col_global_min)/(curr_smoothed_col_global_max - curr_smoothed_col_global_min) - 1)).abs()
                #     a_df[a_smoothed_col_name] = a_df[a_smoothed_col_name].apply(lambda x: (x - curr_smoothed_col_global_min)/(curr_smoothed_col_global_max - curr_smoothed_col_global_min)).abs() ## normalize column between [0.0, +1.0]


            a_df[a_diff_col_name] = a_df[a_smoothed_col_name].diff().abs()
            if enable_global_normalization:
                if a_diff_col_name not in global_max_min_dict:
                    global_max_min_dict[a_diff_col_name] = (a_df[a_diff_col_name].min(skipna=True), a_df[a_diff_col_name].max(skipna=True))
                else:
                    global_max_min_dict[a_diff_col_name] = (min(global_max_min_dict[a_diff_col_name][0], a_df[a_diff_col_name].min(skipna=True)), max(global_max_min_dict[a_diff_col_name][1], a_df[a_diff_col_name].max(skipna=True)))

                ## normalize column between [0.0, +1.0]
                curr_diff_col_global_min, curr_diff_col_global_max = global_max_min_dict[a_diff_col_name]
                a_df[a_diff_col_name] = a_df[a_diff_col_name].apply(_build_active_norm_fn(curr_global_min=curr_diff_col_global_min, curr_global_max=curr_diff_col_global_max))
                # if (curr_diff_col_global_min != np.nan) and (curr_diff_col_global_max != np.nan):
                    # a_df[a_diff_col_name] = a_df[a_diff_col_name].apply(lambda x: (x - curr_diff_col_global_min)/(curr_diff_col_global_max - curr_diff_col_global_min)).abs()
                    
                    
                    
        ## END for a_col_name, a_smoothed_col_name in zip(se...
        
        a_df['total'] = a_df[diff_col_names].max(axis='columns', skipna=True)
        if enable_global_normalization:
            if 'total' not in global_max_min_dict:
                global_max_min_dict['total'] = (a_df['total'].min(skipna=True), a_df['total'].max(skipna=True))
            else:
                global_max_min_dict['total'] = (min(global_max_min_dict['total'][0], a_df['total'].min(skipna=True)), max(global_max_min_dict['total'][1], a_df['total'].max(skipna=True)))
            ## normalize column between [-1.0, +1.0]
            # a_df[active_accel_col_names] = a_df[active_accel_col_names].apply(lambda x: 2*(x - x.min())/(x.max() - x.min()) - 1)
            curr_total_col_global_min, curr_total_col_global_max = global_max_min_dict['total']
            # _temp_df[a_smoothed_col_name] = _temp_df[a_smoothed_col_name].apply(lambda x: 2*(x - x.min())/(x.max() - x.min()) - 1)
            a_df['total'] = a_df['total'].apply(_build_active_norm_fn(curr_global_min=curr_total_col_global_min, curr_global_max=curr_total_col_global_max))

            # if (curr_total_col_global_min != np.nan) and (curr_total_col_global_max != np.nan):
                # _temp_df['total'] = _temp_df['total'].apply(lambda x: 2*(x - curr_total_col_global_min)/(curr_total_col_global_max - curr_total_col_global_min) - 1) ## normalize column between [-1.0, +1.0]
                # a_df['total'] = a_df['total'].apply(lambda x: (x - curr_total_col_global_min)/(curr_total_col_global_max - curr_total_col_global_min)).abs() ## normalize column between [0.0, +1.0]
                
        # _temp_df = _temp_df.dropna(axis='index', how='any', subset=['total'])
        a_df['is_moving'] = (a_df['total'] > total_change_threshold)

        # moving_only_df = a_df[a_df['is_moving']]
        return a_df
    



    @classmethod
    def preprocess(cls, datasets_MOTION, n_most_recent_sessions_to_preprocess: Optional[int] = 5):
        """ 
        preprocessed_EEG_save_path: Path = eeg_analyzed_parent_export_path.joinpath('preprocessed_EEG').resolve()
        preprocessed_EEG_save_path.mkdir(exist_ok=True)

        ## INPUTS: flat_data_modality_dict
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = flat_data_modality_dict['EEG']  ## Unpacking


        """
        from mne.channels.montage import DigMontage
        from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper

        ## BEGIN ANALYSIS of EEG Data
        num_MOTION_files: int = len(datasets_MOTION)
        motion_session_IDXs = np.arange(num_MOTION_files)

        if (n_most_recent_sessions_to_preprocess is not None) and (n_most_recent_sessions_to_preprocess > 0):
            n_most_recent_sessions_to_preprocess = min(n_most_recent_sessions_to_preprocess, num_MOTION_files) ## don't process more than we have
            active_motion_IDXs = motion_session_IDXs[-n_most_recent_sessions_to_preprocess:]
        else:
            ## ALL sessions
            active_motion_IDXs = deepcopy(motion_session_IDXs)

        analysis_results_MOTION = []
        valid_active_IDXs = []
        
        for i in active_motion_IDXs:
            motion_analysis_results = None ## start empty
            try:
                a_raw_motion = datasets_MOTION[i]
                good_channels = a_raw_motion.pick_types(eeg=True)
                # sampling_rate = a_raw_motion.info["sfreq"]  # Store the sampling rate
                # print(f'sampling_rate: {sampling_rate}')
                a_raw_motion.load_data()
                # datasets_MOTION[i] = a_raw_motion ## update it and put it back
                motion_analysis_results = cls._perform_process_motion_session(a_raw_motion)
                # a_raw_savepath: Path = preprocessed_EEG_save_path.joinpath(a_raw_motion.filenames[0].name).resolve()
                # a_raw_motion.save(a_raw_savepath, overwrite=True)
                
            except ValueError as e:
                print(f'Encountered value error: {e} while trying to processing MOTION file {i}/{len(active_motion_IDXs)}: {a_raw_motion}. Skipping')
                datasets_MOTION[i] = None ## drop result
                motion_analysis_results = None ## no analysis result
                pass
            except Exception as e:
                raise e

            if motion_analysis_results is not None:
                analysis_results_MOTION.append(motion_analysis_results)
                valid_active_IDXs.append(i)



        valid_active_IDXs = np.array(valid_active_IDXs)
                
        ## OUTPUTS: analysis_results_EEG
        ## UPDATES: eeg_session_IDXs
        return (valid_active_IDXs, analysis_results_MOTION)
    

    # ==================================================================================================================================================================================================================================================================================== #
    # Computation Helpers                                                                                                                                                                                                                                                                  #
    # ==================================================================================================================================================================================================================================================================================== #
    @classmethod
    def compute_quaternions(cls, df: pd.DataFrame, beta=0.1, time_col_sec_name: str='timestamp') -> pd.DataFrame:
        """ function that will take your historical time series DataFrame and output a quaternion for each row using a Madgwick-style fusion of Gyro + Acc

        adds the quaternion channels: ['qw', 'qx', 'qy', 'qz'] to the dataframe

        Usage:
            raw_df = compute_quaternions(raw_df)
            raw_df
        
        """
        def normalize(v):
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        quats = []
        q = np.array([1.0, 0.0, 0.0, 0.0])

        for i in range(len(df)):
            if i == 0:
                quats.append(q.copy())
                continue

            dt = df[time_col_sec_name].iloc[i] - df[time_col_sec_name].iloc[i-1]
            gx, gy, gz = np.radians([df['GyroX'].iloc[i], df['GyroY'].iloc[i], df['GyroZ'].iloc[i]])
            ax, ay, az = normalize([df['AccX'].iloc[i], df['AccY'].iloc[i], df['AccZ'].iloc[i]])
            q1, q2, q3, q4 = q

            # Gradient descent correction
            f1 = 2*(q2*q4 - q1*q3) - ax
            f2 = 2*(q1*q2 + q3*q4) - ay
            f3 = 2*(0.5 - q2*q2 - q3*q3) - az
            J = np.array([[-2*q3,  2*q4, -2*q1,  2*q2],
                            [ 2*q2,  2*q1,  2*q4,  2*q3],
                            [ 0,    -4*q2, -4*q3,  0]])
            step = normalize(J.T @ np.array([f1, f2, f3]))

            gx -= beta * step[0]
            gy -= beta * step[1]
            gz -= beta * step[2]

            # Quaternion derivative
            q_dot = 0.5 * np.array([
                -q2*gx - q3*gy - q4*gz,
                    q1*gx + q3*gz - q4*gy,
                    q1*gy - q2*gz + q4*gx,
                    q1*gz + q2*gy - q3*gx
            ])

            q += q_dot * dt
            q = normalize(q)
            quats.append(q.copy())

        quat_df: pd.DataFrame = pd.DataFrame(quats, columns=['qw','qx','qy','qz'])
        return pd.concat([df.reset_index(drop=True), quat_df], axis=1)

    @classmethod
    def quaternion_to_rot_matrix(cls, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])

    # ==================================================================================================================================================================================================================================================================================== #
    # Real-time/Dynamic Updating Quaternion Estimation                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #
    @classmethod
    def normalize(cls, v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @classmethod
    def update_quaternion(cls, q, gyro_deg_s, acc_g, dt, beta=0.1):
        """ between-time-step quaternion updating 

        Usage:
            ## Quaternion Based: ['qw', 'qx', 'qy', 'qz']
            self._orientation['qw'], self._orientation['qx'], self._orientation['qy'], self._orientation['qz'] = MotionData.update_quaternion(q=(self._orientation['qw'], self._orientation['qx'], self._orientation['qy'], self._orientation['qz']),
                                                                                                                                acc_g=latest_sample[:3], ## [0,1,2]
                                                                                                                                gyro_deg_s=latest_sample[3:5], # [3,4,5]
                                                                                                        )                                                                                                    
        """
        gx, gy, gz = np.radians(gyro_deg_s)
        ax, ay, az = cls.normalize(acc_g)
        q1, q2, q3, q4 = q

        # Gradient descent correction from accelerometer
        f1 = 2*(q2*q4 - q1*q3) - ax
        f2 = 2*(q1*q2 + q3*q4) - ay
        f3 = 2*(0.5 - q2*q2 - q3*q3) - az
        J = np.array([[-2*q3,  2*q4, -2*q1,  2*q2],
                        [ 2*q2,  2*q1,  2*q4,  2*q3],
                        [ 0,    -4*q2, -4*q3,  0]])
        step = cls.normalize(J.T @ np.array([f1, f2, f3]))

        # Apply feedback
        gx -= beta * step[0]
        gy -= beta * step[1]
        gz -= beta * step[2]

        # Quaternion derivative from gyro
        q_dot = 0.5 * np.array([
            -q2*gx - q3*gy - q4*gz,
                q1*gx + q3*gz - q4*gy,
                q1*gy - q2*gz + q4*gx,
                q1*gz + q2*gy - q3*gx
        ])

        q += q_dot * dt
        return cls.normalize(q)



    ## Plotting/Visualization
    @classmethod
    def plot_3d_orientation(cls, df, step=10):
        """ Plots the df containing Quaternion data's orientation as a function of time (kinda, it generates a snapshot every `step` frames
        

            # Usage:
            plot_3d_orientation(raw_df, step=20)

        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D        

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        origin = np.zeros(3)

        # Draw coordinate frames for quaternions every `step` rows
        for i in range(0, len(df), step):
            # q = df.loc[i, ['qw','qx','qy','qz']].to_numpy()
            q = df.iloc[i][['qw','qx','qy','qz']].to_numpy()

            R = cls.quaternion_to_rot_matrix(q)

            # Plot axes: red=x, green=y, blue=z
            ax.quiver(*origin, *R[:,0], length=0.5, color='r', arrow_length_ratio=0.2)
            ax.quiver(*origin, *R[:,1], length=0.5, color='g', arrow_length_ratio=0.2)
            ax.quiver(*origin, *R[:,2], length=0.5, color='b', arrow_length_ratio=0.2)

        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Orientation from Quaternion')
        plt.show()







@pd.api.extensions.register_dataframe_accessor("motion_df")
class MotionDataFrame(CommonDataFrameAccessorMixin):
    """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals 
    
    from phopymnehelper.motion_data import MotionDataFrame

    """
    timestamp_dt_column_names = ['t_start', 't_duration']
    timestamp_rel_column_names = ['onset', 'duration']

    _required_column_names = ['start', 'stop', 'label', 'duration']

    def __init__(self, pandas_obj):      
        pandas_obj = self._validate(pandas_obj)
        self._obj = pandas_obj



try:
    from pypho_timeline.utils.datetime_helpers import datetime_to_unix_timestamp

    @pd.api.extensions.register_dataframe_accessor("bad_motion_epochs_df")
    class BadMotionDataFrame(CommonDataFrameAccessorMixin):
        """ A Pandas pd.DataFrame representation of [start, stop, label] epoch intervals 
        
        from phopymnehelper.motion_data import MotionDataFrame, BadMotionDataFrame

        """
        timestamp_dt_column_names = ['t_start', 't_duration']
        timestamp_rel_column_names = ['onset', 'duration']

        _required_column_names = ['start', 'stop', 'label', 'duration']

        def __init__(self, pandas_obj):      
            pandas_obj = self._validate(pandas_obj)
            self._obj = pandas_obj

        def is_timestamp_format(self) -> bool:
            return np.all([v in self._obj.columns for v in self.timestamp_rel_column_names])

        @classmethod
        def _motion_bad_cell_to_unix_seconds(cls, x: Any) -> float:
            if isinstance(x, (datetime, pd.Timestamp)):
                return float(datetime_to_unix_timestamp(x))
            return float(x)

        @classmethod
        def _detail_t_column_to_unix_numpy(cls, t_col: pd.Series) -> np.ndarray:
            if pd.api.types.is_datetime64_any_dtype(t_col):
                return np.asarray(datetime_to_unix_timestamp(t_col.tolist()), dtype=np.float64)
            return np.asarray(t_col.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)


        @classmethod
        def motion_bad_intervals_key_suffix(cls, bad_unix_df: pd.DataFrame) -> str:
            if bad_unix_df is None or len(bad_unix_df) == 0:
                return ''
            is_timestamp_format: bool = np.all([v in bad_unix_df.columns for v in cls.timestamp_rel_column_names])
            if is_timestamp_format:
                rows = sorted((round(cls._motion_bad_cell_to_unix_seconds(r.onset), 6), round(float(r.duration.total_seconds()), 6)) for r in bad_unix_df.itertuples(index=False))
            else:
                rows = sorted((round(float(r.t_start), 6), round(float(r.t_duration), 6)) for r in bad_unix_df.itertuples(index=False))
            return hashlib.md5(repr(rows).encode('utf-8')).hexdigest()[:12]


        @classmethod
        def normalize_motion_bad_intervals_df(cls, bad_df: Optional[pd.DataFrame], time_origin_unix: Optional[float] = None) -> pd.DataFrame:
            """Build a DataFrame with float Unix ``t_start`` and ``t_duration`` for motion bad/exclusion intervals.

            Accepts either timeline-style columns ``t_start``/``t_duration`` (Unix seconds or datetimes), or MNE-style
            ``onset``/``duration`` (seconds relative to recording start) plus ``time_origin_unix`` (recording start, Unix s).
            """
            if bad_df is None or len(bad_df) == 0:
                return pd.DataFrame(columns=cls.timestamp_dt_column_names)
            df = bad_df.copy()
            
            if 'onset' in df.columns and 'duration' in df.columns:
                # if time_origin_unix is None:
                #     raise ValueError("bad_intervals_df uses MNE columns 'onset' and 'duration'; pass bad_intervals_time_origin_unix (recording start as Unix seconds).")
                # return pd.DataFrame({'t_start': time_origin_unix + df['onset'].astype(float), 't_duration': df['duration'].astype(float)}).reset_index(drop=True)
                return df

            elif 't_start' in df.columns and 't_duration' in df.columns:
                t_st = df['t_start'].map(cls._motion_bad_cell_to_unix_seconds)
                t_du = df['t_duration'].astype(float)
                return pd.DataFrame({'t_start': t_st, 't_duration': t_du}).reset_index(drop=True)
            else:
                raise ValueError("bad_intervals_df must have columns ('t_start', 't_duration') or ('onset', 'duration').")


        def adding_unix_float_columns(self, inplace: bool=True) -> pd.DataFrame:
            """ creates the unix_float columns 

            Adds columns: `timestamp_dt_column_names = ['t_start', 't_duration']`
            Requires existing columns: `timestamp_rel_column_names = ['onset', 'duration']`

            """
            assert self.is_timestamp_format(), f"expected timestamp columns: {self.timestamp_rel_column_names} but had self._obj.columns: {self._obj.columns}" #_obj
            if inplace:
                # self._obj['t_start'] = self._obj['onset'].map(cls._motion_bad_cell_to_unix_seconds)
                self._obj['t_start'] = self._detail_t_column_to_unix_numpy(self._obj['onset']) #.map(cls._motion_bad_cell_to_unix_seconds)
                self._obj['t_duration'] = self._obj['duration'].dt.total_seconds()
                ## add the optional end/stop columns:
                self._obj['onset_end'] = self._obj['onset'] + self._obj['duration']
                self._obj['t_stop'] = self._obj['t_start'] + self._obj['t_duration']
                return self._obj
            else:
                df: pd.DataFrame = self._obj.copy()
                df['t_start'] = self._detail_t_column_to_unix_numpy(df['onset']) #.map(cls._motion_bad_cell_to_unix_seconds)
                df['t_duration'] = df['duration'].dt.total_seconds()
                ## add the optional end/stop columns:
                df['onset_end'] = df['onset'] + df['duration']
                df['t_stop'] = df['t_start'] + df['t_duration']
                return df

except Exception as e:
    BadMotionDataFrame = None
    # raise e

