import pytest
import mne
import numpy as np
from phopymnehelper.analysis.computations.specific.bad_epochs import filter_bad_epochs_qc_params, bad_epochs_qc_params_fingerprint, BadEpochsQCComputation

def test_filter_bad_epochs_qc_params():
    params = {"l_freq": 1.0, "use_autoreject": True, "extra_param": 123}
    filtered = filter_bad_epochs_qc_params(params)
    assert "l_freq" in filtered
    assert "use_autoreject" in filtered
    assert "extra_param" not in filtered

def test_bad_epochs_qc_params_fingerprint():
    params1 = {"l_freq": 1.0, "use_autoreject": True, "extra_param": 123}
    params2 = {"use_autoreject": True, "l_freq": 1.0, "other": 456}
    assert bad_epochs_qc_params_fingerprint(params1) == bad_epochs_qc_params_fingerprint(params2)

def test_compute_bad_epochs_qc():
    info = mne.create_info(ch_names=['eeg1', 'eeg2'], sfreq=256, ch_types=['eeg', 'eeg'])
    data = np.random.randn(2, 256 * 10)  # 10 seconds
    raw = mne.io.RawArray(data, info)

    result = BadEpochsQCComputation.compute_bad_epochs_qc(raw, use_autoreject=False)

    assert result is not None
    assert 'bad_epoch_intervals_df' in result
