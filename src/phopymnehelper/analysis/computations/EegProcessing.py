import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, fftpack
import mne
import matplotlib.mlab as mlab

## TODO: I broke this function for live values in favor of having it work for historical values loaded from 'EegFromMatFileSource.py'.
def process_eeg(data):
    # data: numpy array of shape (numSamples, 14) where 14 is the number of channels
    # print(data.shape)
    numVariables = data.shape[1]
    numSamples = data.shape[0]
    #print('numVariables: ', numVariables, ' numSamples: ', numSamples)

    # Iteration based method:
    # fig = plt.figure(constrained_layout=True)
    # fig.clf()
    #
    for aChannelIndex in range(0, (numVariables-1)):
        ch = signal.detrend(data[:, aChannelIndex].T)
        time_step = 1 / 128.0
        sample_freq = fftpack.fftfreq(ch.size, d=time_step)
        pidxs = np.where(sample_freq > 0)
        freqs = sample_freq[pidxs]
        max_freq = 128.0 / 2.0

        fft = fftpack.fft(ch)
        power = np.abs(fft)[pidxs]
        max_power_ind = power.argmax()
        max_args = power.argsort()[::-1][:5]
        max_power = power[max_args]
        # print('channel[', aChannelIndex, ']: ',[max_args], power[max_args])
        print('channel[', aChannelIndex, ']r: ', freqs, power)


        # Spectral Power Density:
        # power, freqs = matplotlib.mlab.psd(ch, 128, 128, detrend, window, noverlap=0, pad_to, sides=, scale_by_freq)
        psd_power, psd_freqs = mlab.psd(ch, 128, 128)
        print('channel[', aChannelIndex, ']: ',psd_freqs, psd_power)


        # Plots radial frequency/power figure.
        ## TODO: Replace scatterplot with different plot object to better visualize the data. A standard plot(...) should work.
        if aChannelIndex == 12:
            # Only plot the 5th one (arrbitrarily)
            # r = 2 * max_power
            r = 2 * power
            theta = 2 * np.pi * (freqs / max_freq)
            # area = 200 * r ** 2
            colors = theta
            fig = plt.figure()
            # ax = fig.add_subplot(111, projection='polar')
            ax = fig.add_subplot(111, polar=True)
            # c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
            # c = ax.scatter(theta, r, c=colors, cmap='hsv', alpha=0.75)
            c = ax.plot(theta, r, alpha=0.75)



def process_eeg_power(data):
    # data: numpy array of shape (numSamples, 14) where 14 is the number of channels
    # print(data.shape)
    Fs = 128 # Sampling rate (Hz)
    time_step = 1.0 / Fs # Sampling timestemp [sec]

    numVariables = data.shape[1]
    numSamples = data.shape[0]

    # Get the sampling frequencies:
    sample_freq = fftpack.fftfreq(numSamples, d=time_step)  # sample_freq returns the frequencies of the fft centered around 0. The first half is positive and the second half is negative, with zero in the middle
    # print('sample_freq:',sample_freq.shape) # sample_freq: (1152,)
    # print('sample_freq:',sample_freq)
    positive_sample_frequency_indicies = np.where(sample_freq > 0) # omits the negative frequencies and zero
    outputFreqs = sample_freq[positive_sample_frequency_indicies]

    numOutputSamples = len(outputFreqs)

    numPSD = Fs
    numOutputPSD = int((numPSD / 2.0) + 1)
    max_freq = numPSD / 2.0

    # outputFreqs = np.zeros((numOutputSamples,numVariables))
    outputPower = np.zeros((numOutputSamples, numVariables))
    outputPsdFreqs = np.zeros((numOutputPSD, numVariables))
    outputPsdPower = np.zeros((numOutputPSD, numVariables))

    # Define Blackman window to filter out excess noise at low frequencies
    # window = np.blackman(numSamples)

    ## TODO: for performance, can do fft = fftpack.fft(ch, axis=0)
    ch = signal.detrend(data.T, axis=1) # the data.T.shape: (14, numSamples) => ch.shape: (14, numSamples)
    fft = fftpack.fft(ch, axis=1)  # fft.shape: (14, 1152)

    # Iteration based method:
    for aChannelIndex in range(0, (numVariables-1)):
        # channelData = data[:, aChannelIndex].T * window
        # channelData = data[:, aChannelIndex].T
        # ch = signal.detrend(channelData)

        # NFFT: numSamples
        # fft = fftpack.fft(ch) # fft.shape: (1152,)
        # print('fft: ', fft.shape)
        outputPower[:,aChannelIndex] = np.abs(fft)[aChannelIndex, positive_sample_frequency_indicies]
        max_power_ind = outputPower[:,aChannelIndex].argmax()
        max_args = outputPower[:,aChannelIndex].argsort()[::-1][:5]
        max_power = outputPower[:,aChannelIndex][max_args]
        # print('channel[', aChannelIndex, ']: ',[max_args], power[max_args])
        # print('channel[', aChannelIndex, ']r: ', freqs, power)

        # Spectral Power Density:
        # power, freqs = matplotlib.mlab.psd(ch, 128, 128, detrend, window, noverlap=0, pad_to, sides=, scale_by_freq)
        # outputPsdPower[:,aChannelIndex], outputPsdFreqs[:,aChannelIndex] = mlab.psd(ch, numPSD, numPSD)
        outputPsdPower[:, aChannelIndex], outputPsdFreqs[:, aChannelIndex] = mlab.psd(ch, Fs=Fs, NFFT=Fs)
        # print('channel[', aChannelIndex, ']: ',psd_freqs, psd_power)
        # print('channel[', aChannelIndex, ']r: ', freqs.shape, power.shape, psd_freqs.shape, psd_power.shape) # (575,) (575,) (65,) (65,)


    return (outputFreqs, outputPower, outputPsdFreqs, outputPsdPower)



from mne.time_frequency import psd_array_multitaper
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.signal import periodogram, welch

def bandpower(data: NDArray[np.float64], fs: float, method: str, band: tuple[float, float], relative: bool = True, **kwargs) -> Union[NDArray[np.float64], Dict[str, NDArray[np.float64]]] :
    """Compute the bandpower of the individual channels.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        Data on which the the bandpower is estimated.
    fs : float
        Sampling frequency in Hz.
    method : 'periodogram' | 'welch' | 'multitaper'
        Method used to estimate the power spectral density.
    band : tuple of shape (2,)
        Frequency band of interest in Hz as 2 floats, e.g. ``(8, 13)``. The
        edges are included.
    relative : bool
        If True, the relative bandpower is returned instead of the absolute
        bandpower.
    **kwargs : dict
        Additional keyword arguments are provided to the power spectral density
        estimation function.
        * 'periodogram': scipy.signal.periodogram
        * 'welch'``: scipy.signal.welch
        * 'multitaper': mne.time_frequency.psd_array_multitaper

        The only provided arguments are the data array and the sampling
        frequency.

    Returns
    -------
    bandpower : array of shape (n_channels,)
        The bandpower of each channel.
        
    Usage:
    
         from phoofflineeeganalysis.EegProcessing import bandpower
         
         
        # let's explore some frequency bands
        iter_freqs = [
            ('Theta', 4, 8),
            ('Alpha', 8, 16),
            ('Beta', 16, 32),
            ('Gamma', 32, 45)
        ]

        iter_freqs_df: pd.DataFrame = pd.DataFrame(iter_freqs, columns=['band_name', 'low_Hz', 'high_Hz'])
        iter_freqs_df

        source_id = uuid.uuid4().hex
        with Player(raw, chunk_size=200, name="bandpower-example", source_id=source_id):
            stream = Stream(bufsize=4, name="bandpower-example", source_id=source_id)
            stream.connect(acquisition_delay=0.1, processing_flags="all")
            stream.pick("eeg").filter(1, 58)
            stream.get_data()  # reset the number of new samples after the filter is applied

            datapoints, times = [], []
            while stream.n_new_samples < stream.n_buffer:
                time.sleep(0.1)  # wait for the buffer to be entirely filled
            while len(datapoints) != 30:
                if stream.n_new_samples == 0:
                    continue  # wait for new samples
                data, ts = stream.get_data()
                bp = bandpower(data, stream.info["sfreq"], "periodogram", band=(8, 13))
                datapoints.append(bp)
                times.append(ts[-1])
            stream.disconnect()
         
    """
    # compute the power spectral density
    assert data.ndim == 2, (
        "The provided data must be a 2D array of shape (n_channels, n_samples)."
    )
    if method == "periodogram":
        freqs, psd = periodogram(data, fs, **kwargs)
    elif method == "welch":
        freqs, psd = welch(data, fs, **kwargs)
    elif method == "multitaper":
        psd, freqs = psd_array_multitaper(data, fs, verbose="ERROR", **kwargs)
    else:
        raise RuntimeError(f"The provided method '{method}' is not supported.")
    # compute the bandpower
    freq_res = freqs[1] - freqs[0]
    if relative:
        ## power across full spectrum to normalize to relative:
        total_bandpower = simpson(psd, dx=freq_res)  

    if not isinstance(band, pd.DataFrame):
        assert len(band) == 2, "The 'band' argument must be a 2-length tuple."
        assert band[0] <= band[1], (
            "The 'band' argument must be defined as (low, high) (in Hz)."
        )
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        bandpower = simpson(psd[:, idx_band], dx=freq_res)
        if relative:
            bandpower = bandpower / total_bandpower
    else:
        ## multiple named bands passed as dfs
        assert ('low_Hz' in band.columns), f"band.columns: {list(band.columns)}"
        assert ('high_Hz' in band.columns), f"band.columns: {list(band.columns)}"
        assert ('band_name' in band.columns), f"band.columns: {list(band.columns)}"
        band_names_list: List[str] = band['band_name'].to_list()
        idx_band = [np.logical_and(freqs >= row.low_Hz, freqs <= row.high_Hz) for row in band.itertuples()]
        
        # Check if any frequencies exist in the bands and handle edge cases
        bandpower = []
        for i, an_idx_band in enumerate(idx_band):
            if np.any(an_idx_band):  # Check if any frequencies fall in this band
                try:
                    bp = simpson(psd[:, an_idx_band], dx=freq_res)
                    bandpower.append(bp)
                except (IndexError, ValueError) as e:
                    # Handle edge cases with insufficient data
                    print(f"Warning: Could not compute bandpower for band {band_names_list[i]}: {e}")
                    bandpower.append(np.zeros(psd.shape[0]))  # Return zeros for this band
            else:
                print(f"Warning: No frequencies found in band {band_names_list[i]} ({band.iloc[i]['low_Hz']}-{band.iloc[i]['high_Hz']} Hz)")
                bandpower.append(np.zeros(psd.shape[0]))  # Return zeros for this band
        
        if relative:
            bandpower = [a_bandpower / total_bandpower for a_bandpower in bandpower] 
        bandpower = dict(zip(band_names_list, bandpower))
    return bandpower




def annotate_jaw_clench(
    raw,
    *,
    window_size=1.0,          # seconds
    step_size=0.1,            # seconds
    ptp_thresh=None,          # float or None → auto-estimate
    thresh_scale=8.0,         # only used when ptp_thresh is None
    min_channels=4,           # how many channels must be “loud” together
    merge_threshold=0.2,      # s; merge detections closer than this
    annot_description="jaw_clench",
    picks="eeg",
    return_thresholds=False,  # optionally return per-channel thresholds
):
    """
    Detect jaw-clench artefacts in a Raw object and add them as Annotations.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG data.
    window_size, step_size : float
        Sliding window parameters in seconds.
    ptp_thresh : float | None
        Fixed peak-to-peak threshold in Volts.  If None, an adaptive
        (channel-specific) threshold is derived from the data.
    thresh_scale : float
        Multiplier for MAD when deriving the adaptive threshold.
    min_channels : int
        Number of channels that must simultaneously exceed threshold.
    merge_threshold : float
        Merge detections that are closer than this gap (s).
    annot_description : str
        Description string stored in the resulting annotations.
    picks : str | list
        Channel selection passed to MNE (default “eeg”).
    return_thresholds : bool
        If True, the function returns a dict {channel: threshold} in addition
        to the Raw object.

    Returns
    -------
    raw_out : mne.io.Raw
        A *copy* of `raw` with added jaw-clench annotations.
    thresh_dict : dict  (only if return_thresholds=True)
        Per-channel thresholds used during detection.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("`raw` must be an instance of mne.io.Raw or its subclasses.")

    # ---------------------------------------------------------------------
    # Prepare data and indices
    # ---------------------------------------------------------------------
    picks_idx = (
        mne.pick_channels(raw.info["ch_names"], include=picks)
        if isinstance(picks, (list, tuple))
        else mne.pick_types(raw.info, meg=False, eeg=True)
    )
    ch_names = [raw.ch_names[i] for i in picks_idx]

    data = raw.get_data(picks=picks_idx, reject_by_annotation="omit")
    sfreq = raw.info["sfreq"]
    n_samples = data.shape[1]

    win_samp = int(round(window_size * sfreq))
    step_samp = int(round(step_size * sfreq))

    # ---------------------------------------------------------------------
    # 1) Compute sliding-window PTPs for every channel
    # ---------------------------------------------------------------------
    # We store results in a list first for memory efficiency on long files
    ptp_per_channel = [[] for _ in range(len(picks_idx))]

    for start in range(0, n_samples - win_samp + 1, step_samp):
        stop = start + win_samp
        window = data[:, start:stop]
        ptp_vals = window.max(axis=1) - window.min(axis=1)
        for ch_i, val in enumerate(ptp_vals):
            ptp_per_channel[ch_i].append(val)

    # convert lists to arrays, shape (n_channels, n_windows)
    ptp_per_channel = np.array([np.array(lst) for lst in ptp_per_channel])

    # ---------------------------------------------------------------------
    # 2) Determine thresholds (either fixed or adaptive)
    # ---------------------------------------------------------------------
    if ptp_thresh is not None:
        # Same scalar for every channel
        thresh_per_channel = np.full(len(picks_idx), float(ptp_thresh))
    else:
        # Adaptive: median + thresh_scale * MAD   (robust to outliers)
        med = np.median(ptp_per_channel, axis=1)
        mad = 1.4826 * np.median(np.abs(ptp_per_channel - med[:, None]), axis=1)
        thresh_per_channel = med + thresh_scale * mad
        # Guard against channels with zero MAD (flat signal)
        thresh_per_channel[mad == 0] = med[mad == 0] * 1.1

    thresh_dict = dict(zip(ch_names, thresh_per_channel))

    # ---------------------------------------------------------------------
    # 3) Identify candidate windows
    # ---------------------------------------------------------------------
    candidate_starts, candidate_ends = [], []

    # Re-use ptp_per_channel (already computed) to avoid another pass
    n_windows = ptp_per_channel.shape[1]
    for w in range(n_windows):
        ptp_this_window = ptp_per_channel[:, w]
        n_high = np.sum(ptp_this_window > thresh_per_channel)
        if n_high >= min_channels:
            start_time = (w * step_samp) / sfreq
            end_time = (w * step_samp + win_samp) / sfreq
            candidate_starts.append(start_time)
            candidate_ends.append(end_time)

    # ---------------------------------------------------------------------
    # 4) Merge neighbouring / overlapping detections
    # ---------------------------------------------------------------------
    merged_on, merged_off = [], []
    for on, off in zip(candidate_starts, candidate_ends):
        if not merged_on:
            merged_on.append(on)
            merged_off.append(off)
        elif on - merged_off[-1] <= merge_threshold:
            merged_off[-1] = off  # extend the previous segment
        else:
            merged_on.append(on)
            merged_off.append(off)

    durations = np.array(merged_off) - np.array(merged_on)

    # ---------------------------------------------------------------------
    # 5) Build and attach the annotations
    # ---------------------------------------------------------------------
    new_annots = mne.Annotations(
        onset=merged_on,
        duration=durations,
        description=[annot_description] * len(merged_on),
        orig_time=raw.info["meas_date"],
    )

    raw_out = raw.copy()
    raw_out.set_annotations(raw.annotations + new_annots)

    if return_thresholds:
        return raw_out, thresh_dict
    return raw_out



import mne
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

def analyze_eeg_trends(
        raw_list,
        subject_id="S01",
        bands=[(4, 8, "theta"), (8, 12, "alpha"), (12, 30, "beta")],
        epoch_length=4.0,       # seconds
        epoch_overlap=1.0,      # seconds
        filter_hp=1,            # Hz – high-pass
        filter_lp=40            # Hz – low-pass
):
    """
    Segment each Raw into fixed-length epochs, extract band-power,
    and model time-of-day trends with a mixed-effects model.

    Parameters
    ----------
    raw_list : list of mne.io.Raw
    subject_id : str
    bands : list of tuples (fmin, fmax, label)
    epoch_length : float
        Length of each epoch in seconds.
    epoch_overlap : float
        Overlap between successive epochs in seconds (0 = butt-joined).
    filter_hp, filter_lp : float
        High-pass and low-pass corner frequencies for a basic band-pass filter.

    Returns
    -------
    df_feat : pd.DataFrame
        One row per epoch with band-power and timing info.
    models  : dict
        label -> fitted statsmodels MixedLM (or error string).
    """
    from yasa import bandpower
    from statsmodels.formula.api import mixedlm

    feat_rows = []     # accumulate dictionaries, one per epoch

    for raw_idx, raw in enumerate(raw_list):
        if raw is None or len(raw) == 0:
            continue

        try:
            # 1) Copy & filter
            raw.load_data() ## load the data as needed
            raw_filt = raw.copy().filter(filter_hp, filter_lp, fir_design="firwin")
            sf = raw_filt.info["sfreq"]

            # 2) Make fixed-length epochs
            epochs = mne.make_fixed_length_epochs(
                raw_filt,
                duration=epoch_length,
                overlap=epoch_overlap,
                preload=True,
                reject_by_annotation=True,
                # picks="all",
            )

            if len(epochs) == 0:
                continue

            data_epochs = epochs.get_data(units="uV")  # (n_epochs, n_chans, n_samples)

            # 3) Absolute start-time of each epoch
            meas_date = raw_filt.info["meas_date"]
            if meas_date is None:
                # Set to "today" noon to keep downstream code happy
                meas_date = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

            event_samp = epochs.events[:, 0]          # sample index of each epoch start
            epoch_starts = [meas_date + timedelta(seconds=s / sf) for s in event_samp]

            # 4) Loop over epochs and compute band-power
            for ep_i, ep_data in enumerate(data_epochs):
                bp = bandpower(
                    ep_data, sf=sf,
                    ch_names=epochs.ch_names,
                    bands=bands,
                    relative=True
                ).mean()        # average across channels

                row = {lbl: bp.get(lbl, np.nan) for _, _, lbl in bands}

                # attach timing information
                start_time = epoch_starts[ep_i]
                row["datetime"] = start_time
                row["time_of_day"] = start_time.hour + start_time.minute / 60 + start_time.second / 3600
                row["day"] = start_time.date()
                row["subject"] = subject_id
                row["recording_idx"] = raw_idx
                row["epoch_idx"] = ep_i
                feat_rows.append(row)
            ## END for ep_i, ep_data in enumerate(data_epochs)...
        except ValueError as e:
            print(f'WARN: raw file: {raw} failed with error: {e}. Skipping.')
            continue
        
        except Exception as e:
            raise e

    ## END for raw_idx, raw in enumerate(raw_list)...
    
    df_feat = pd.DataFrame(feat_rows)

    # 5) Mixed-effects models: band ~ time_of_day  (day = random intercept)
    models = {}
    for _, _, lbl in bands:
        if lbl not in df_feat.columns:
            models[lbl] = f"Band {lbl} not present."
            continue
        try:
            md = mixedlm(f"{lbl} ~ time_of_day", df_feat, groups=df_feat["day"])
            models[lbl] = md.fit()
        except Exception as e:
            models[lbl] = str(e)

    return df_feat, models