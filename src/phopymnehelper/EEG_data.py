import time
import re
from datetime import datetime, timezone

import uuid
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import h5py
from matplotlib import pyplot as plt

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.signal import spectrogram
import xarray as xr

import mne
from mne import set_log_level
from copy import deepcopy
from mne.filter import filter_data  # Slightly faster for large data as direct fn

from mne.io import read_raw
# import pyedflib ## for creating single EDF+ files containing channel with different sampling rates (e.g. EEG and MOTION data)

mne.viz.set_browser_backend("Matplotlib")

# from ..EegProcessing import bandpower
import autoreject # apply_autoreject_filter

set_log_level("WARNING")


class EEGData:
    """ Methods related to processing of motion (gyro/accel/magnet/quaternion/etc) data

    from phopymnehelper.EEG_data import EEGData

    (all_data_MOTION, all_times_MOTION), datasets_MOTION, df_MOTION = flat_data_modality_dict['MOTION']  ## Unpacking
    df_MOTION

    """

    @classmethod
    def _perform_process_EEG_session(cls, raw_eeg):
        """ Called by `preprocess` on each rawEEG session
         
            eeg_analysis_results = _perform_process_EEG_session(raw_eeg)
            eeg_analysis_results['eog_epochs'].plot_image(combine="mean")
            eeg_analysis_results['eog_epochs'].average().plot_joint()

            # Visualize the extracted microstates
            nk.microstates_plot(eeg_analysis_results['microstates'], epoch=(0, 500)) # RuntimeError: No digitization points found.

        """
        eeg_analysis_results = {}
        # Apply band-pass filter (1-58Hz) and re-reference the raw signal
        raw_eeg = raw_eeg.copy().filter(1, 58, verbose=False)
        # eeg = nk.eeg_rereference(eeg, "average") ## do not do this, it apparently reduces quality

        # Extract microstates
        # eeg_analysis_results['microstates'] = nk.microstates_segment(raw_eeg, n_microstates=4)

        ## Try EOG -- fials due to lack of digitization points -- # RuntimeError: No digitization points found.
        eeg_analysis_results['eog_epochs'] = mne.preprocessing.create_eog_epochs(raw_eeg, ch_name=['AF3', 'AF4']) ## , baseline=(-0.5, -0.2) use the EEG channels closest to the eyes - eog_epochs 

        return eeg_analysis_results
    

    @classmethod
    def set_montage(cls, datasets_EEG: List):
        """ 
        preprocessed_EEG_save_path: Path = eeg_analyzed_parent_export_path.joinpath('preprocessed_EEG').resolve()
        preprocessed_EEG_save_path.mkdir(exist_ok=True)

        ## INPUTS: flat_data_modality_dict
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = flat_data_modality_dict['EEG']  ## Unpacking


        """
        from mne.channels.montage import DigMontage
        from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper

        active_electrode_man: ElectrodeHelper = ElectrodeHelper.init_EpocX_montage()
        emotiv_epocX_montage: DigMontage = active_electrode_man.active_montage
        
        if isinstance(datasets_EEG, (mne.io.BaseRaw, mne.io.Raw, mne.io.RawArray)):
            # datasets_EEG = [datasets_EEG] # single element list
            datasets_EEG.set_montage(emotiv_epocX_montage)
        else:
            for i, raw_eeg in enumerate(datasets_EEG):
                # raw_eeg = raw_eeg.pick(["eeg"], verbose=False)
                # raw_eeg.load_data()
                # sampling_rate = raw_eeg.info["sfreq"]  # Store the sampling rate
                raw_eeg.set_montage(emotiv_epocX_montage)


    @classmethod
    def preprocess(cls, datasets_EEG: List, preprocessed_EEG_save_path: Optional[Path]=None, n_most_recent_sessions_to_preprocess: Optional[int] = 5):
        """ 
        preprocessed_EEG_save_path: Path = eeg_analyzed_parent_export_path.joinpath('preprocessed_EEG').resolve()
        preprocessed_EEG_save_path.mkdir(exist_ok=True)
        
        ## INPUTS: flat_data_modality_dict
        (all_data_EEG, all_times_EEG), datasets_EEG, df_EEG = flat_data_modality_dict['EEG']  ## Unpacking
        
        
        """
        from mne.channels.montage import DigMontage
        from phopymnehelper.anatomy_and_electrodes import ElectrodeHelper

        active_electrode_man: ElectrodeHelper = ElectrodeHelper.init_EpocX_montage()
        emotiv_epocX_montage: DigMontage = active_electrode_man.active_montage

        ## BEGIN ANALYSIS of EEG Data
        num_EEG_files: int = len(datasets_EEG)
        eeg_session_IDXs = np.arange(num_EEG_files)

        if (n_most_recent_sessions_to_preprocess is not None) and (n_most_recent_sessions_to_preprocess > 0):
            n_most_recent_sessions_to_preprocess = min(n_most_recent_sessions_to_preprocess, num_EEG_files) ## don't process more than we have
            active_eeg_rec_IDXs = eeg_session_IDXs[-n_most_recent_sessions_to_preprocess:]
        else:
            ## ALL sessions
            active_eeg_rec_IDXs = deepcopy(eeg_session_IDXs)

        analysis_results_EEG = []
        valid_active_IDXs = []

        for i in active_eeg_rec_IDXs:
            eeg_analysis_results = None
            try:
                raw_eeg = datasets_EEG[i]
                raw_eeg = raw_eeg.pick(["eeg"], verbose=False)
                raw_eeg.load_data()
                sampling_rate = raw_eeg.info["sfreq"]  # Store the sampling rate
                raw_eeg.set_montage(emotiv_epocX_montage)
                datasets_EEG[i] = raw_eeg ## update it and put it back
                eeg_analysis_results = cls._perform_process_EEG_session(raw_eeg)
            except ValueError as e:
                print(f'Encountered value error: {e} while trying to processing EEG file {i}/{len(datasets_EEG)}: {raw_eeg}. Skipping')
                datasets_EEG[i] = None ## drop result
                eeg_analysis_results = None ## no analysis result
                pass
            except Exception as e:
                raise e
            
            if eeg_analysis_results is not None:
                analysis_results_EEG.append(eeg_analysis_results)
                valid_active_IDXs.append(i)
                
                if preprocessed_EEG_save_path is not None:
                    if not preprocessed_EEG_save_path.exists():
                        preprocessed_EEG_save_path.mkdir(parents=True, exist_ok=True)
                    a_raw_savepath: Path = preprocessed_EEG_save_path.joinpath(raw_eeg.filenames[0].name).resolve()
                    raw_eeg.save(a_raw_savepath, overwrite=True)
                    
            else:
                # valid_active_IDXs
                print(f'EEG dataset {i} is invalid. Skipping.')
                pass
            

        valid_active_IDXs = np.array(valid_active_IDXs)
        
        ## OUTPUTS: analysis_results_EEG
        ## UPDATES: eeg_session_IDXs
        # return (active_session_IDXs, analysis_results_EEG)
        return (valid_active_IDXs, analysis_results_EEG)
    


class EEGComputations:
    """ 
    
    from phopymnehelper.EEG_data import EEGComputations, EEGData
    
        
        
    _all_outputs = EEGComputations.run_all(raw=raw)
    """
    @classmethod
    def all_fcns_dict(cls):
        return {
            'time_independent_bad_channels': cls.time_independent_bad_channels,
            # 'time_dependent_bad_channels': cls.time_dependent_bad_channels,
            'raw_data_topo': cls.raw_data_topo,
            'cwt': cls.raw_morlet_cwt, 
            'spectogram': cls.raw_spectogram_working,
        }


    @classmethod
    def run_all(cls, raw, should_suppress_exceptions: bool=True, **kwargs):
        _all_outputs = {}
        _all_fns = cls.all_fcns_dict()
        for a_fn_name, a_fn in _all_fns.items():
            print(f'running {a_fn_name}...')
            try:
                _all_outputs[a_fn_name] = a_fn(raw, **kwargs)
                print(f'\tdone.')
            except Exception as e:
                print(f'\terror occured in {a_fn_name}: error {e}.')
                if not should_suppress_exceptions:
                    raise
            
        return _all_outputs


    @classmethod
    def run_all_with_graph(cls, raw, should_suppress_exceptions: bool = True, use_cache: bool = False, cache_root: Optional[Path] = None, session_path: Optional[Path] = None, session_mtime: Optional[float] = None, parallel: bool = False, max_workers: int = 4, **kwargs):
        """Run the same computations as ``all_fcns_dict`` via the DAG executor (ordered dict compatible with ``run_all``). Optional per-node disk cache under *cache_root*."""
        from phopymnehelper.analysis.computations.cache import DiskComputationCache
        from phopymnehelper.analysis.computations.eeg_registry import run_eeg_graph_legacy_ordered, session_fingerprint_for_raw_or_path
        try:
            session = session_fingerprint_for_raw_or_path(raw, path=session_path, mtime=session_mtime)
            cache = DiskComputationCache(Path(cache_root)) if (use_cache and cache_root is not None) else None
            return run_eeg_graph_legacy_ordered(raw=raw, session=session, global_params=kwargs, cache=cache, use_cache=use_cache, parallel=parallel, max_workers=max_workers)
        except Exception as e:
            print(f'run_all_with_graph error: {e}.')
            if not should_suppress_exceptions:
                raise
            return {}


    @classmethod
    def raw_morlet_cwt(cls, raw: mne.io.Raw, picks=None, wavelet_param=4, num_freq=60, fmax=50, spacing=12.5, **kwargs):
        """Compute continuous Morlet wavelet transform for MNE Raw EEG.

        raw: mne.io.Raw
        picks: channels to use (default: all EEG)
        wavelet_param: number of cycles
        num_freq: number of frequencies
        fmax: highest frequency (Hz)
        spacing: frequency step (Hz) if <1, treated as log spacing factor
        """
        if picks is None:
            picks = mne.pick_types(raw.info, eeg=True, meg=False)

        fs = raw.info["sfreq"]
        data = raw.get_data(picks=picks)

        if spacing < 1:  # logarithmic spacing
            freqs = np.geomspace(fmax/num_freq, fmax, num=num_freq)
        else:  # linear spacing
            freqs = np.arange(spacing, fmax+spacing, spacing)[:num_freq]

        power = mne.time_frequency.tfr_array_morlet(data[np.newaxis], sfreq=fs, freqs=freqs, n_cycles=wavelet_param, output="power")

        return dict(freqs=freqs, power=power[0])
    

    @classmethod
    def raw_data_topo(cls, raw: mne.io.Raw, l_freq=1, h_freq=58, epoch_dur=4,
                       epoch_step: float = 0.250,
                       moving_avg_epochs: int = 32,
                       **kwargs):
        """Compute continuous Morlet wavelet transform for MNE Raw EEG.

        raw: mne.io.Raw
        picks: channels to use (default: all EEG)
        wavelet_param: number of cycles
        num_freq: number of frequencies
        fmax: highest frequency (Hz)
        spacing: frequency step (Hz) if <1, treated as log spacing factor
        
        l_freq=1
        h_freq=58
        epoch_dur=4
        epoch_step=0.025
        moving_avg_epochs=32
        
        epoch_avg = _all_outputs['raw_data_topo']['epoch_avg']
                
        epochs = _all_outputs['raw_data_topo']['epochs']
        
        
        """
        # out_dict = dict(raw_filtered=out_dict['raw_filtered'], topo=out_dict['data_topo'], epoch_avg=out_dict['epoch_avg'], mov_avg=out_dict['mov_avg'], epochs=out_dict['epochs'])
        out_dict = dict(raw_filtered=None, topo=None, epoch_avg=None, mov_avg=None, epochs=None)
        
        # 1. Temporal filter: filter modifies in-place if not copying
        out_dict['raw_filtered'] = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', n_jobs='cuda')

        # 2. Epoching (sliding window)
        sfreq = raw.info['sfreq']
        step_samples: int = int(epoch_step * sfreq)
        window_samples: int  = int(epoch_dur * sfreq)
        data = out_dict['raw_filtered'].get_data()
        n_ch, n_times = data.shape

        print(f'for INPUT PARAMS: epoch_dur: {epoch_dur}, epoch_step: {epoch_step}, moving_avg_epochs: {moving_avg_epochs}')
        print(f'\tstep_samples: {step_samples}, window_samples: {window_samples}\n\tn_ch: {n_ch}, n_times: {n_times}')
        # Use strided view for efficient sliding windows (no explicit python loops)
        def epoch_strided(data, window_samples, step_samples):
            n_epochs = (n_times - window_samples) // step_samples + 1
            s0, s1 = data.strides
            return np.lib.stride_tricks.as_strided(
                data,
                shape=(n_epochs, n_ch, window_samples),
                strides=(step_samples * s1, s0, s1),
                writeable=False
            )

        out_dict['epochs'] = epoch_strided(data, window_samples, step_samples)
        # shape: (n_epochs, n_ch, n_samples)

        # 3. x²
        epochs_squared = np.square(out_dict['epochs'])

        # 4. Moving epoch average (vectorized via convolution)
        # kernel = np.ones(moving_avg_epochs) / moving_avg_epochs
        # pad so output is same shape (len=n_epochs)
        # mov_avg = np.apply_along_axis(
        #     lambda m: np.vstack([np.convolve(m[:, ch, samp], kernel, mode='full')[:len(m)] 
        #                         for ch in range(n_ch) for samp in range(window_samples)]
        #                     ).reshape((n_ch, window_samples, len(m))).transpose(2, 0, 1),
        #     axis=0, arr=epochs_squared[None,...]).squeeze()

        # Alternatively, for moving window average over epochs axis, use:
        # import scipy.ndimage
        out_dict['mov_avg'] = scipy.ndimage.uniform_filter1d(epochs_squared, size=moving_avg_epochs, axis=0, origin=-(moving_avg_epochs//2)).squeeze()

        # 5. Epoch average (over all epochs)
        out_dict['epoch_avg'] = out_dict['mov_avg'].mean(axis=0)  # (n_ch, n_samples)
        out_dict['data_topo'] = out_dict['epoch_avg'].mean(axis=1)  # (n_ch,)
        
        return out_dict


    @classmethod
    def raw_spectogram_working(cls, raw: mne.io.Raw, picks=None, nperseg=1024, noverlap=512, mask_bad_annotated_times: bool=True):
        """Compute continuous spectrogram for MNE Raw EEG.

        BAD channels (e.g. from ``time_independent_bad_channels`` or ``raw.info['bads']``)
        are excluded from the computation.  When *mask_bad_annotated_times* is True,
        time segments covered by ``BAD_*`` annotations are NaN-ed out before the FFT.

        raw: mne.io.Raw
        picks: channels to use (default: all EEG, excluding bads)
        wavelet_param: number of cycles
        num_freq: number of frequencies
        fmax: highest frequency (Hz)
        spacing: frequency step (Hz) if <1, treated as log spacing factor


        plt.figure(num='spectrogram', clear=True)
        plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='auto'); plt.ylim([1,40])


        Unpack like:

            a_spectogram_result: Dict = a_result['spectogram'] 

            ch_names = a_spectogram_result['ch_names']
            fs = a_spectogram_result['fs']
            a_spectogram_result_dict = a_spectogram_result['spectogram_result_dict'] # Dict[channel: Tuple]
            Sxx = a_spectogram_result['Sxx']
            Sxx_avg = a_spectogram_result['Sxx_avg']
            
            for a_ch, a_tuple in a_spectogram_result_dict.items():
                f, t, Sxx = a_tuple ## unpack the tuple
                
        """
        # from scipy.signal import spectrogram        

        # Exclude BAD channels (e.g. from time_independent_bad_channels) so they are not included in spectrograms.
        bads = set(raw.info.get("bads") or [])
        if picks is None:
            picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        else:
            picks = [p for p in picks if raw.info.ch_names[p] not in bads]

        fs: float = raw.info["sfreq"]

        if mask_bad_annotated_times:
            ## NaN out the bad annoted times (motion epochs, blink artifacts, etc):
            data, times = raw.get_data(picks=picks, return_times=True)
            mask = np.ones_like(times, dtype=bool)
            for ann in raw.annotations:
                if ann['description'].startswith('BAD_'):
                    start = ann['onset']
                    stop = start + ann['duration']
                    mask &= ~((times >= start) & (times < stop))
            data[:, ~mask] = np.nan
        else:
            data = raw.get_data(picks=picks)

        ch_names = [raw.info.ch_names[i] for i in picks]
        Sxx_list = []
        Sxx_avg_list = []
        
        spectogram_result_dict = {}
        for row_idx, a_ch in enumerate(ch_names):
            f, t, Sxx = spectrogram(data[row_idx], fs=fs, nperseg=nperseg, noverlap=noverlap) # #TODO 2025-09-28 13:25: - [ ] Convert to newer `ShortTimeFFT.spectrogram`
            spectogram_result_dict[a_ch] = (f, t, Sxx) ## a tuple
            Sxx_list.append(Sxx) # np.shape(Sxx) # (513, 1116) - (n_freqs, n_times)
            Sxx_avg = np.nanmean(Sxx, axis=-1) ## average over all time to get one per session
            Sxx_avg_list.append(Sxx_avg)

        Sxx_avg_list = np.stack(Sxx_avg_list) # (14, 513) - (n_channels, n_freqs)
        Sxx_list = np.stack(Sxx_list) # (14, 513, 1116) - (n_channels, n_freqs, n_times)
            
        Sxx_avg_list = xr.DataArray(Sxx_avg_list, dims=("channels", "freqs"), coords={"channels": ch_names, "freqs": f})
        Sxx_list = xr.DataArray(Sxx_list, dims=("channels", "freqs", "times"), coords={"channels": ch_names, "freqs": f, "times": t})

        # return dict(fs=fs, spectogram_result_dict=spectogram_result_dict)

        return dict(t=t, freqs=f, fs=fs, ch_names=ch_names,
                    spectogram_result_dict=spectogram_result_dict,
                                        Sxx_avg=Sxx_avg_list,
                                        Sxx=Sxx_list,
                                        )


    # ==================================================================================================================================================================================================================================================================================== #
    # Channel Quality Determination/Bad Channels                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #


    @classmethod
    def detect_bad_channels_sliding_window(cls, raw: mne.io.Raw, *, eeg_reference='average', projection=False, l_freq=1.0, h_freq=40.0, fir_design='firwin', picks='eeg', window_sec=10.0, step_sec=5.0,
                                        z_rms_threshold=3.0, mean_corr_threshold=0.4, bad_fraction_threshold=0.3, copy_raw=True) -> Dict:
        """Sliding-window RMS z-score + mean |correlation| heuristic for bad EEG channels.
        _out_bad_channels_dict = EEGComputations.detect_bad_channels_sliding_window(raw=a_raw)
        bad_chs = _out_bad_channels_dict['bad_chs']
        _out_bad_channels_dict['bad_mask'] ## for whole data
        _out_bad_channels_dict['good_chs']

        ## actually set on object
        a_raw.info['bads'] = bad_chs

        """
        def compute_rms(x):
            return np.sqrt(np.mean(x**2, axis=1))

        def mean_channel_corr(x):
            C = np.corrcoef(x)
            return np.nanmean(np.abs(C), axis=1)

        a_raw = raw.copy() if copy_raw else raw
        a_raw.set_eeg_reference(eeg_reference, projection=projection)
        a_raw.filter(l_freq, h_freq, fir_design=fir_design)

        sfreq = float(a_raw.info['sfreq'])
        data = a_raw.get_data(picks=picks)

        win = int(window_sec * sfreq)
        step = int(step_sec * sfreq)
        n_ch, n_t = data.shape

        bad_mask = []
        for start in range(0, n_t - win, step):
            w = data[:, start : start + win]
            rms = compute_rms(w)
            z_rms = (rms - rms.mean()) / (rms.std() if rms.std() > 0 else np.nan)
            mcorr = mean_channel_corr(w)
            bad = (np.abs(z_rms) > z_rms_threshold) | (mcorr < mean_corr_threshold)
            bad_mask.append(bad)

        bad_mask = np.array(bad_mask)
        bad_fraction = bad_mask.mean(axis=0)
        ch_names = np.array(a_raw.ch_names)
        bad_chs = ch_names[bad_fraction > bad_fraction_threshold].tolist()
        good_chs = sorted(set(a_raw.ch_names) - set(bad_chs))

        return {'bad_chs': bad_chs, 'good_chs': good_chs, 'bad_fraction': bad_fraction, 'bad_mask': bad_mask}


    @classmethod
    def apply_autoreject_filter(cls, a_raw, epoch_fixed_duration=3, should_plot: bool = False):
        """ computes the bad epochs via the autoreject package, using global/local amplitude thresholds determined dynamically

        Usage:
            from phopymnehelper.EEG_data import EEGComputations

            epochs_cleaned, (epochs, reject_log), ica = EEGComputations.apply_autoreject_filter(a_raw, epoch_fixed_duration=3)
            epochs
            epochs_cleaned


        """
        a_raw.copy().filter(l_freq=1, h_freq=None)
        epochs = mne.make_fixed_length_epochs(a_raw, duration=epoch_fixed_duration, preload=True)

        # plot the data
        # epochs.average().detrend().plot_joint()


        # picks = mne.pick_types(a_raw.info, meg=True, eeg=True, stim=False,
        #                        eog=True, include=include, exclude='bads')
        # epochs = mne.Epochs(a_raw, events, event_id, tmin, tmax,
        #                     picks=picks, baseline=(None, 0), preload=True,
        #                     reject=None, verbose=False, detrend=1)

        ar = autoreject.AutoReject(n_interpolate=[1, 2, 3], random_state=1337, n_jobs=1, verbose=True)
        ar.fit(epochs[:20])  # fit on a few epochs to save time
        epochs_cleaned, reject_log = ar.transform(epochs, return_log=True)

        if should_plot:
            epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
            reject_log.plot('horizontal')

        # epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

        # compute ICA
        ica = mne.preprocessing.ICA(random_state=1337)
        ica.fit(epochs[~reject_log.bad_epochs])
        exclude = [0,  # blinks
                2  # saccades
                ]
        if should_plot:
            ica.plot_components(exclude)
        ica.exclude = exclude
        if should_plot:
            ica.plot_overlay(epochs.average(), exclude=ica.exclude)
        ica.apply(epochs, exclude=ica.exclude)

        return epochs_cleaned, (epochs, reject_log), ica



    @classmethod
    def time_independent_bad_channels(cls, raw: mne.io.Raw, skip_trying_PREP: bool=True, debug_print: bool=True, **kwargs) -> dict:
        """Detect low-quality EEG channels across the entire recording/session using pyprep.

        Uses ``pyprep.prep_pipeline`` on the entire EEG recording without gaps

        """
        did_find_bad_channels: bool = False

        _out_dict = dict(interpolated_channels=[], bad_channels_original=[], still_noisy_channels=[], noisy_channels_after_interpolation={}, all_bad_channels=[])

        # PREP Pipeline
        if not skip_trying_PREP:
            # _out_dict = dict(interpolated_channels=[], bad_channels_original=[], still_noisy_channels=[], noisy_channels_after_interpolation={}, all_bad_channels=[])

            try:
                from pyprep.prep_pipeline import PrepPipeline
            except ImportError:
                import warnings
                warnings.warn("pyprep is not installed; skipping time-independent bad-channel detection. Install with: uv add pyprep", RuntimeWarning, stacklevel=2)
                # return _out_dict

            try:
                import warnings as _warnings
                _prep_keys = frozenset({"ransac", "channel_wise", "max_chunk_size", "random_state", "filter_kwargs", "reject_by_annotation", "matlab_strict"})
                prep_extra = {k: v for k, v in kwargs.items() if k in _prep_keys}
                sample_rate = float(raw.info["sfreq"])
                montage = raw.get_montage()
                n_eeg = len(mne.pick_types(raw.info, meg=False, eeg=True, exclude=[]))
                line_upper = int(np.floor(sample_rate / 2.0))
                prep_params = {"ref_chs": "eeg", "reref_chs": "eeg", "line_freqs": np.arange(60, line_upper, 60)}
                use_ransac = bool(prep_extra["ransac"]) if "ransac" in prep_extra else (n_eeg >= 16)

                def _fit_prep(*, with_ransac: bool):
                    rc = raw.copy()
                    p = PrepPipeline(rc, prep_params, montage, **{**prep_extra, "ransac": with_ransac})
                    p.fit()
                    return p

                try:
                    prep = _fit_prep(with_ransac=use_ransac)
                except IndexError as _idx_err:
                    if not use_ransac:
                        raise
                    _warnings.warn(f"PrepPipeline RANSAC raised {_idx_err!r}; retrying with ransac=False.", RuntimeWarning, stacklevel=2)
                    prep = _fit_prep(with_ransac=False)

                interpolated_channels = list(prep.interpolated_channels)
                noisy_channels_original = dict(prep.noisy_channels_original)
                bad_channels_original = list(noisy_channels_original.get("bad_all", []))
                still_noisy_channels = list(prep.still_noisy_channels)
                noisy_channels_after_interpolation = dict(prep.noisy_channels_after_interpolation)

                if debug_print:
                    print(f"\tBad channels: {interpolated_channels}")
                    print(f"\tBad channels original: {bad_channels_original}")
                    print(f"\tBad channels after interpolation: {still_noisy_channels}")

                # Union of all detected bad channels so downstream stages can skip them.
                all_bad_channels = sorted(set(bad_channels_original) | set(still_noisy_channels))

                # Persist bad channels onto the original Raw object for subsequent computations.
                raw.info["bads"] = sorted(set(list(raw.info.get("bads") or []) + all_bad_channels))

                did_find_bad_channels = True
                _out_dict = dict(interpolated_channels=interpolated_channels, bad_channels_original=bad_channels_original, still_noisy_channels=still_noisy_channels, noisy_channels_after_interpolation=noisy_channels_after_interpolation,
                            all_bad_channels=all_bad_channels)

            except Exception as e:
                import warnings
                warnings.warn(f"PrepPipeline failed ({type(e).__name__}: {e}); no channels marked bad.", RuntimeWarning, stacklevel=2)
                did_find_bad_channels = False
                # return _empty_out



        if not did_find_bad_channels:
            try:
                ## try custom output
                print(f'trying EEGComputations.detect_bad_channels_sliding_window(...) to detect bad channels...')
                _out_bad_channels_dict = EEGComputations.detect_bad_channels_sliding_window(raw=raw)
                all_bad_channels = _out_bad_channels_dict['bad_chs']
                # _out_bad_channels_dict['bad_mask'] ## for whole data
                good_chs = _out_bad_channels_dict['good_chs']

                did_find_bad_channels = True
                _out_dict = _out_dict | dict(all_bad_channels=all_bad_channels, bad_fraction=_out_bad_channels_dict.get('bad_fraction', None),  
                                            bad_mask=_out_bad_channels_dict.get('bad_mask', None), good_chs=_out_bad_channels_dict.get('good_chs', None),
                )

                if debug_print:
                    print(f"\tBad channels: {all_bad_channels}")
                    print(f"\tGood channels: {good_chs}")

                ## actually set on object
                # raw.info['bads'] = all_bad_channels
                # Persist bad channels onto the original Raw object for subsequent computations.
                raw.info["bads"] = sorted(set(list(raw.info.get("bads") or []) + all_bad_channels))

            except Exception as e:
                import warnings
                warnings.warn(f"EEGComputations.detect_bad_channels_sliding_window(...) failed ({type(e).__name__}: {e}); no channels marked bad.", RuntimeWarning, stacklevel=2)
                did_find_bad_channels = False
                # raise e

        return _out_dict



    @classmethod
    def time_dependent_bad_channels(cls, raw: mne.io.Raw, window_sec: float = 3.0, picks=None, n_neighbors: int = 20, threshold: float = 1.5, return_scores: bool = False, **kwargs) -> dict:
        """Detect low-quality EEG channels in fixed-duration windows using LOF.

        Uses ``mne.preprocessing.find_bad_channels_lof`` on each non-overlapping
        window of ``window_sec`` length.
        """
        if window_sec <= 0:
            raise ValueError(f"window_sec must be > 0, got {window_sec}")

        if picks is None:
            picks = mne.pick_types(raw.info, eeg=True, meg=False)
        elif isinstance(picks, (list, tuple, np.ndarray)) and len(picks) > 0 and isinstance(picks[0], str):
            picks = mne.pick_channels(raw.info["ch_names"], include=list(picks), exclude=[])
        else:
            picks = np.atleast_1d(np.asarray(picks, dtype=int).ravel())

        n_channels = int(picks.size)
        if n_channels == 0:
            empty_df = pd.DataFrame(columns=["t_start", "t_end", "n_bad", "bad_channels"])
            out = dict(window_sec=float(window_sec), intervals=[], bad_channels_per_interval=[], df=empty_df)
            if return_scores:
                out["scores_per_interval"] = []
            return out

        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")

        effective_n_neighbors = max(1, min(int(n_neighbors), n_channels - 1)) if n_channels > 1 else 1
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        samples_per_window = max(1, int(round(window_sec * sfreq)))

        intervals = []
        bad_channels_per_interval = []
        scores_per_interval = []

        for start_idx in range(0, n_times, samples_per_window):
            end_idx = min(start_idx + samples_per_window, n_times)  # exclusive
            if end_idx <= start_idx:
                continue

            t_start = start_idx / sfreq
            t_end = end_idx / sfreq
            crop_tmax = (end_idx - 1) / sfreq
            raw_seg = raw.copy().crop(tmin=t_start, tmax=crop_tmax, include_tmax=True)

            if n_channels < 3:
                bads = []
                scores = np.full(n_channels, np.nan, dtype=float)
            else:
                try:
                    if return_scores:
                        bads, scores = mne.preprocessing.find_bad_channels_lof(raw_seg, n_neighbors=effective_n_neighbors, picks=picks, threshold=threshold, return_scores=True)
                    else:
                        bads = mne.preprocessing.find_bad_channels_lof(raw_seg, n_neighbors=effective_n_neighbors, picks=picks, threshold=threshold, return_scores=False)
                        scores = None
                except Exception as e:
                    print(f'\tLOF error in window [{t_start:.3f}, {t_end:.3f}] sec: {e}')
                    bads = []
                    scores = np.full(n_channels, np.nan, dtype=float) if return_scores else None

            intervals.append((t_start, t_end))
            bad_channels_per_interval.append(list(bads))
            if return_scores:
                score_arr = np.asarray(scores, dtype=float).ravel()
                if score_arr.shape[0] != n_channels:
                    fixed_scores = np.full(n_channels, np.nan, dtype=float)
                    fixed_scores[:min(n_channels, score_arr.shape[0])] = score_arr[:min(n_channels, score_arr.shape[0])]
                    score_arr = fixed_scores
                scores_per_interval.append(score_arr)

        rows = []
        for (t_start, t_end), bad_channels in zip(intervals, bad_channels_per_interval):
            rows.append(dict(t_start=t_start, t_end=t_end, n_bad=len(bad_channels), bad_channels=bad_channels))
        out_df = pd.DataFrame.from_records(rows, columns=["t_start", "t_end", "n_bad", "bad_channels"])

        out = dict(window_sec=float(window_sec), intervals=intervals, bad_channels_per_interval=bad_channels_per_interval, df=out_df)
        if return_scores:
            out["scores_per_interval"] = scores_per_interval
        return out


    # ==================================================================================================================================================================================================================================================================================== #
    # Read/Write                                                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #
    @classmethod
    def perform_write_to_hdf(cls, a_result, f, root_key: str='/', debug_print=True):
        """ 
        EEGComputations.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

        from phopymnehelper.EEG_data import EEGComputations

        # EEGComputations.to_hdf(a_result=results[0], file_path="")
        hdf5_out_path: Path = Path('E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/outputs').joinpath('2025-09-23_eegComputations.h5').resolve()
        hdf5_out_path

        for idx, (a_raw, a_raw_outputs) in enumerate(zip(active_only_out_eeg_raws, results)):
            # a_path: Path = Path(a_raw.filenames[0])
            # basename: str = a_path.stem
            # basename: str = a_raw.info.get('meas_date')
            src_file_path: Path = Path(a_raw.info.get('description')).resolve()
            basename: str = src_file_path.stem

            print(f'basename: {basename}')
            EEGComputations.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

            # EEGComputations.to_hdf(a_result=results[0], file_path="", root_key=f"/{basename}/")

            # for an_output_key, an_output_dict in a_raw_outputs.items():
            #     for an_output_subkey, an_output_value in an_output_dict.items():
            #         final_data_key: str = '/'.join([basename, an_output_key, an_output_subkey])
            #         print(f'\tfinal_data_key: "{final_data_key}"')
            #         # all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)

            # spectogram_result_dict = a_raw_outputs['spectogram']['spectogram_result_dict']
            # fs = a_raw_outputs['spectogram']['fs']

            # for ch_idx, (a_ch, a_ch_spect_result_tuple) in enumerate(spectogram_result_dict.items()):
            #     all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)
            #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)

            #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)


        # E:\Dropbox (Personal)\Databases\AnalysisData\MNE_preprocessed\outputs\


        """
        def _perform_write_dict_recurrsively(attribute, a_value):
            if debug_print:
                print(f'attribute: {attribute}')
            if isinstance(a_value, pd.DataFrame):
                a_value.to_hdf(f, key=attribute, append=True)
            elif isinstance(a_value, (xr.DataArray, xr.Dataset)):
                # xr.open_dataset("/path/to/my/file.h5", group="/my/group")
                # f.create_dataset(attribute, data=a_value.values)
                f.create_dataset(attribute, data=a_value)
                # xr.open_dataset("/path/to/my/file.h5", group="/my/group")
            elif isinstance(a_value, np.ndarray):
                f.create_dataset(attribute, data=a_value)
            elif isinstance(a_value, (str, float, int)):
                # f.attrs.create(
                print(f'cannot yet write attributes. Skipping "{attribute}" of type {type(a_value)}')
            elif isinstance(a_value, dict):
                for a_sub_attribute, a_sub_value in a_value.items():
                    ## process each subattribute independently
                    _perform_write_dict_recurrsively(f"{attribute}/{a_sub_attribute}", a_sub_value)

            elif (Path(attribute).parts[-2] == 'spectogram_result_dict') and isinstance(a_value, tuple) and len(a_value) == 3:
                ## unpack tuple
                # freqs, t, Sxx = a_value
                ## convert to dict and pass attribute as-is
                # _perform_write_dict_recurrsively(attribute, {'f': freqs, 't': t, 'Sxx': Sxx})
                pass
            else:
                print(f'error: {attribute} of type {type(a_value)} cannot be written. Skipping')                

        _perform_write_dict_recurrsively(f'{root_key}', a_value=a_result)




    @classmethod
    def to_hdf(cls, a_result, file_path: Path, root_key: str='/', debug_print=True):
        """ 
        EEGComputations.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

        from phopymnehelper.EEG_data import EEGComputations

        # EEGComputations.to_hdf(a_result=results[0], file_path="")
        hdf5_out_path: Path = Path('E:/Dropbox (Personal)/Databases/AnalysisData/MNE_preprocessed/outputs').joinpath('2025-09-23_eegComputations.h5').resolve()
        hdf5_out_path

        for idx, (a_raw, a_raw_outputs) in enumerate(zip(active_only_out_eeg_raws, results)):
            # a_path: Path = Path(a_raw.filenames[0])
            # basename: str = a_path.stem
            # basename: str = a_raw.info.get('meas_date')
            src_file_path: Path = Path(a_raw.info.get('description')).resolve()
            basename: str = src_file_path.stem

            print(f'basename: {basename}')
            EEGComputations.to_hdf(a_result=a_raw_outputs, file_path=hdf5_out_path, root_key=f"/{basename}/")

            # EEGComputations.to_hdf(a_result=results[0], file_path="", root_key=f"/{basename}/")

            # for an_output_key, an_output_dict in a_raw_outputs.items():
            #     for an_output_subkey, an_output_value in an_output_dict.items():
            #         final_data_key: str = '/'.join([basename, an_output_key, an_output_subkey])
            #         print(f'\tfinal_data_key: "{final_data_key}"')
            #         # all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)

            # spectogram_result_dict = a_raw_outputs['spectogram']['spectogram_result_dict']
            # fs = a_raw_outputs['spectogram']['fs']

            # for ch_idx, (a_ch, a_ch_spect_result_tuple) in enumerate(spectogram_result_dict.items()):
            #     all_WHISPER_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/WHISPER/df', append=True)
            #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)

            #     all_pho_log_to_lsl_df.drop(columns=['filepath']).to_hdf(hdf5_out_path, key='modalities/PHO_LOG_TO_LSL/df', append=True)


        # E:\Dropbox (Personal)\Databases\AnalysisData\MNE_preprocessed\outputs\

        
        """
        write_mode = 'r+'
        if (not file_path.exists()):
            write_mode = 'w'

        with h5py.File(file_path, write_mode) as f:

            def _perform_write_dict_recurrsively(attribute, a_value):
                if debug_print:
                    print(f'attribute: {attribute}')
                if isinstance(a_value, pd.DataFrame):
                    a_value.to_hdf(file_path, key=attribute, append=True)
                elif isinstance(a_value, (xr.DataArray, xr.Dataset)):
                    # xr.open_dataset("/path/to/my/file.h5", group="/my/group")
                    # f.create_dataset(attribute, data=a_value.values)
                    f.create_dataset(attribute, data=a_value)
                    # xr.open_dataset("/path/to/my/file.h5", group="/my/group")
                elif isinstance(a_value, np.ndarray):
                    f.create_dataset(attribute, data=a_value)
                elif isinstance(a_value, (str, float, int)):
                    # f.attrs.create(
                    print(f'cannot yet write attributes. Skipping "{attribute}" of type {type(a_value)}')
                elif isinstance(a_value, dict):
                    for a_sub_attribute, a_sub_value in a_value.items():
                        ## process each subattribute independently
                        _perform_write_dict_recurrsively(f"{attribute}/{a_sub_attribute}", a_sub_value)
                        
                elif (Path(attribute).parts[-2] == 'spectogram_result_dict') and isinstance(a_value, tuple) and len(a_value) == 3:
                    ## unpack tuple
                    # freqs, t, Sxx = a_value
                    ## convert to dict and pass attribute as-is
                    # _perform_write_dict_recurrsively(attribute, {'f': freqs, 't': t, 'Sxx': Sxx})
                    pass
                else:
                    print(f'error: {attribute} of type {type(a_value)} cannot be written. Skipping')                

            _perform_write_dict_recurrsively(f'{root_key}', a_value=a_result)



    # ==================================================================================================================================================================================================================================================================================== #
    # PLOTTING/VISUALIZATION FUNCTIONS                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #

