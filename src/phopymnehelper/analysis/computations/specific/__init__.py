"""Session- or cohort-specific analysis helpers (fatigue, ADHD/sleep intrusions, etc.)."""

from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import compute_theta_delta_sleep_intrusion_series
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase
from phopymnehelper.analysis.computations.specific.EEG_Spectograms import DEFAULT_SPECTROGRAM_NOVERLAP, DEFAULT_SPECTROGRAM_NPERSEG, compute_raw_eeg_spectrogram

__all__ = ["SpecificComputationBase", "compute_theta_delta_sleep_intrusion_series", "compute_raw_eeg_spectrogram", "DEFAULT_SPECTROGRAM_NPERSEG", "DEFAULT_SPECTROGRAM_NOVERLAP"]
