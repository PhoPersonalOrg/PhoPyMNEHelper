import pytest
import mne
import numpy as np
from phopymnehelper.analysis.computations.specific.EEG_Spectograms import filter_eeg_spectrogram_params, eeg_spectrogram_params_fingerprint, compute_raw_eeg_spectrogram, EEGSpectrogramComputation

def test_filter_eeg_spectrogram_params():
    params = {"nperseg": 1024, "noverlap": 512, "extra": "discard"}
    filtered = filter_eeg_spectrogram_params(params)
    assert "nperseg" in filtered
    assert "noverlap" in filtered
    assert "extra" not in filtered

def test_eeg_spectrogram_params_fingerprint():
    params1 = {"nperseg": 1024, "noverlap": 512, "extra": "discard"}
    params2 = {"noverlap": 512, "nperseg": 1024}
    assert eeg_spectrogram_params_fingerprint(params1) == eeg_spectrogram_params_fingerprint(params2)

def test_compute_raw_eeg_spectrogram():
    info = mne.create_info(ch_names=['eeg1'], sfreq=256, ch_types=['eeg'])
    data = np.random.randn(1, 10000)
    raw = mne.io.RawArray(data, info)

    result = compute_raw_eeg_spectrogram(raw, nperseg=256, noverlap=128)
    assert result is not None
    assert 'Sxx' in result
    assert 't' in result
    assert 'freqs' in result
    assert result['fs'] == 256.0
