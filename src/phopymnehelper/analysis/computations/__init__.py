"""Computation protocol: DAG execution, per-node disk cache, EEG node registry."""

from phopymnehelper.analysis.computations.cache import DiskComputationCache, compute_chained_cache_key
from phopymnehelper.analysis.computations.eeg_registry import EEG_COMPUTATION_IDS_ORDERED, ensure_default_eeg_registry, register_eeg_computation_nodes, run_eeg_computations_graph, run_eeg_graph_legacy_ordered, session_fingerprint_for_raw_or_path
from phopymnehelper.analysis.computations.engine import GraphExecutor, expand_required_nodes, topological_sort
from phopymnehelper.analysis.computations.protocol import DEFAULT_REGISTRY, ArtifactKind, ArtifactRef, ComputationNode, ComputationRegistry, PROTOCOL_VERSION, RunContext, SessionFingerprint, register_default

__all__ = [
    "ArtifactKind",
    "ArtifactRef",
    "ComputationNode",
    "ComputationRegistry",
    "DEFAULT_REGISTRY",
    "DiskComputationCache",
    "EEG_COMPUTATION_IDS_ORDERED",
    "GraphExecutor",
    "PROTOCOL_VERSION",
    "RunContext",
    "SessionFingerprint",
    "compute_chained_cache_key",
    "ensure_default_eeg_registry",
    "expand_required_nodes",
    "register_default",
    "register_eeg_computation_nodes",
    "run_eeg_computations_graph",
    "run_eeg_graph_legacy_ordered",
    "session_fingerprint_for_raw_or_path",
    "topological_sort",
]
