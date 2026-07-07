import pytest
import mne
import numpy as np
from phopymnehelper.analysis.computations.specific.fatigue_analysis import compute_fatigue_metrics, analyze_fatigue_trends

def test_compute_fatigue_metrics():
    info = mne.create_info(ch_names=['eeg1', 'eeg2'], sfreq=256, ch_types=['eeg', 'eeg'])
    data = np.random.randn(2, 256 * 120)  # 120 seconds of data
    raw = mne.io.RawArray(data, info)

    result = compute_fatigue_metrics(raw, duration=60, overlap=0.5)

    assert result is not None
    assert 'metrics' in result
    assert 'theta_alpha_ratio_global' in result['metrics']
    assert 'engagement_index' in result['metrics']
    assert 'theta_beta_ratio' in result['metrics']
    assert 'spectral_entropy' in result['metrics']

    assert len(result['metrics']['theta_alpha_ratio_global']) > 0

def test_analyze_fatigue_trends():
    info = mne.create_info(ch_names=['eeg1', 'eeg2'], sfreq=256, ch_types=['eeg', 'eeg'])
    data = np.random.randn(2, 256 * 120)  # 120 seconds of data
    raw = mne.io.RawArray(data, info)

    result = compute_fatigue_metrics(raw, duration=60, overlap=0.5)
    metrics_dict = result['metrics']
    time_points = result['epoch_times']

    trends = analyze_fatigue_trends(metrics_dict, time_points)

    assert trends is not None
    assert 'theta_alpha_ratio_global' in trends
    assert 'engagement_index' in trends
