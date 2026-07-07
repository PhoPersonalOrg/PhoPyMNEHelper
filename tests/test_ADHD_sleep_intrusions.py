import pytest
import mne
import numpy as np
from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import compute_theta_delta_sleep_intrusion_series

def test_compute_theta_delta_sleep_intrusion_series():
    info = mne.create_info(ch_names=['eeg1'], sfreq=256, ch_types=['eeg'])
    data = np.random.randn(1, 256 * 10)  # 10 seconds
    raw = mne.io.RawArray(data, info)

    result = compute_theta_delta_sleep_intrusion_series(raw, window_sec=2.0, step_sec=1.0)

    assert result is not None
    assert 'times' in result
    assert 'theta_delta_ratio' in result
    assert 'session_mean_theta_delta' in result
