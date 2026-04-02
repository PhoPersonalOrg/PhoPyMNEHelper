"""Session- or cohort-specific analysis helpers (fatigue, ADHD/sleep intrusions, etc.)."""

from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import THETA_DELTA_SLEEP_INTRUSION_PARAM_KEYS, ThetaDeltaSleepIntrusionComputation, compute_theta_delta_sleep_intrusion_series, filter_theta_delta_sleep_intrusion_params, theta_delta_sleep_intrusion_params_fingerprint
from phopymnehelper.analysis.computations.specific.bad_epochs import BAD_EPOCHS_QC_PARAM_KEYS, BadEpochsQCComputation, apply_bad_epochs_overlays_to_timeline, autoreject_bad_sample_mask, bad_epochs_qc_params_fingerprint, compute_bad_epochs_qc, filter_bad_epochs_qc_params, fit_autoreject_bad_sample_mask
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase
from phopymnehelper.analysis.computations.specific.EEG_Spectograms import DEFAULT_SPECTROGRAM_NOVERLAP, DEFAULT_SPECTROGRAM_NPERSEG, EEG_SPECTROGRAM_PARAM_KEYS, EEGSpectrogramComputation, compute_raw_eeg_spectrogram, eeg_spectrogram_params_fingerprint, filter_eeg_spectrogram_params

__all__ = [
    "BAD_EPOCHS_QC_PARAM_KEYS",
    "BadEpochsQCComputation",
    "DEFAULT_SPECTROGRAM_NOVERLAP",
    "DEFAULT_SPECTROGRAM_NPERSEG",
    "EEG_SPECTROGRAM_PARAM_KEYS",
    "EEGSpectrogramComputation",
    "SpecificComputationBase",
    "THETA_DELTA_SLEEP_INTRUSION_PARAM_KEYS",
    "ThetaDeltaSleepIntrusionComputation",
    "apply_bad_epochs_overlays_to_timeline",
    "autoreject_bad_sample_mask",
    "bad_epochs_qc_params_fingerprint",
    "compute_bad_epochs_qc",
    "compute_raw_eeg_spectrogram",
    "compute_theta_delta_sleep_intrusion_series",
    "eeg_spectrogram_params_fingerprint",
    "filter_bad_epochs_qc_params",
    "filter_eeg_spectrogram_params",
    "filter_theta_delta_sleep_intrusion_params",
    "fit_autoreject_bad_sample_mask",
    "theta_delta_sleep_intrusion_params_fingerprint",
]
