"""Computation protocol: DAG execution, per-node disk cache, EEG node registry."""

from phopymnehelper.analysis.computations.cache import DiskComputationCache, compute_chained_cache_key
from phopymnehelper.analysis.computations.eeg_registry import EEG_COMPUTATION_IDS_ORDERED, ensure_default_eeg_registry, register_eeg_computation_nodes, run_eeg_computations_graph, run_eeg_graph_legacy_ordered, session_fingerprint_for_raw_or_path
from phopymnehelper.analysis.computations.engine import GraphExecutor, expand_required_nodes, topological_sort
from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import THETA_DELTA_SLEEP_INTRUSION_PARAM_KEYS, ThetaDeltaSleepIntrusionComputation, filter_theta_delta_sleep_intrusion_params, theta_delta_sleep_intrusion_params_fingerprint
from phopymnehelper.analysis.computations.specific.bad_epochs import BAD_EPOCH_INTERVALS_TRACK_DEFAULT_NAME, BAD_EPOCHS_QC_PARAM_KEYS, BadEpochsQCComputation, apply_bad_epochs_overlays_to_timeline, autoreject_bad_sample_mask, bad_epochs_qc_params_fingerprint, compute_bad_epochs_qc, ensure_bad_epochs_interval_track, filter_bad_epochs_qc_params, fit_autoreject_bad_sample_mask
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase
from phopymnehelper.analysis.computations.specific.EEG_Spectograms import EEG_SPECTROGRAM_PARAM_KEYS, EEGSpectrogramComputation, eeg_spectrogram_params_fingerprint, filter_eeg_spectrogram_params
from phopymnehelper.analysis.computations.protocol import DEFAULT_REGISTRY, ArtifactKind, ArtifactRef, ComputationNode, ComputationRegistry, PROTOCOL_VERSION, RunContext, SessionFingerprint, register_default

__all__ = [
    "ArtifactKind",
    "BAD_EPOCH_INTERVALS_TRACK_DEFAULT_NAME",
    "BAD_EPOCHS_QC_PARAM_KEYS",
    "BadEpochsQCComputation",
    "SpecificComputationBase",
    "THETA_DELTA_SLEEP_INTRUSION_PARAM_KEYS",
    "ThetaDeltaSleepIntrusionComputation",
    "apply_bad_epochs_overlays_to_timeline",
    "autoreject_bad_sample_mask",
    "bad_epochs_qc_params_fingerprint",
    "compute_bad_epochs_qc",
    "ensure_bad_epochs_interval_track",
    "ArtifactRef",
    "ComputationNode",
    "ComputationRegistry",
    "DEFAULT_REGISTRY",
    "DiskComputationCache",
    "EEG_COMPUTATION_IDS_ORDERED",
    "EEG_SPECTROGRAM_PARAM_KEYS",
    "EEGSpectrogramComputation",
    "GraphExecutor",
    "PROTOCOL_VERSION",
    "RunContext",
    "SessionFingerprint",
    "compute_chained_cache_key",
    "ensure_default_eeg_registry",
    "expand_required_nodes",
    "eeg_spectrogram_params_fingerprint",
    "filter_bad_epochs_qc_params",
    "filter_eeg_spectrogram_params",
    "filter_theta_delta_sleep_intrusion_params",
    "fit_autoreject_bad_sample_mask",
    "register_default",
    "register_eeg_computation_nodes",
    "run_eeg_computations_graph",
    "run_eeg_graph_legacy_ordered",
    "session_fingerprint_for_raw_or_path",
    "theta_delta_sleep_intrusion_params_fingerprint",
    "topological_sort",
]
