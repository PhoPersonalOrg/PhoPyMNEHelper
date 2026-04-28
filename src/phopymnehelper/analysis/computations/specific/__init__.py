"""Session- or cohort-specific analysis helpers (fatigue, ADHD/sleep intrusions, etc.)."""

from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import ThetaDeltaSleepIntrusionComputation, compute_theta_delta_sleep_intrusion_series
from phopymnehelper.analysis.computations.specific.bad_epochs import BAD_EPOCH_INTERVALS_TRACK_DEFAULT_NAME, BAD_EPOCHS_QC_PARAM_KEYS, BadEpochsQCComputation, apply_bad_epochs_overlays_to_timeline, autoreject_bad_sample_mask, bad_epochs_qc_params_fingerprint, ensure_bad_epochs_interval_track, filter_bad_epochs_qc_params, fit_autoreject_bad_sample_mask
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase
from phopymnehelper.analysis.computations.specific.EEG_Spectograms import DEFAULT_SPECTROGRAM_NOVERLAP, DEFAULT_SPECTROGRAM_NPERSEG, EEG_SPECTROGRAM_PARAM_KEYS, EEGSpectrogramComputation, compute_raw_eeg_spectrogram, eeg_spectrogram_params_fingerprint, filter_eeg_spectrogram_params

__all__ = [
    "BAD_EPOCH_INTERVALS_TRACK_DEFAULT_NAME",
    "BAD_EPOCHS_QC_PARAM_KEYS",
    "BadEpochsQCComputation",
    "DEFAULT_SPECTROGRAM_NOVERLAP",
    "DEFAULT_SPECTROGRAM_NPERSEG",
    "EEG_SPECTROGRAM_PARAM_KEYS",
    "EEGSpectrogramComputation",
    "SpecificComputationBase",
    "ThetaDeltaSleepIntrusionComputation",
    "apply_bad_epochs_overlays_to_timeline",
    "autoreject_bad_sample_mask",
    "bad_epochs_qc_params_fingerprint",
    "ensure_bad_epochs_interval_track",
    "compute_raw_eeg_spectrogram",
    "compute_theta_delta_sleep_intrusion_series",
    "eeg_spectrogram_params_fingerprint",
    "filter_bad_epochs_qc_params",
    "filter_eeg_spectrogram_params",
    "fit_autoreject_bad_sample_mask",
]
