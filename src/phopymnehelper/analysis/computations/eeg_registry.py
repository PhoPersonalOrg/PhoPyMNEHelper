"""Registered EEG computation nodes wrapping EEGComputations (DAG + shared params)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, TypeAlias

import phopymnehelper.type_aliases as types
from phopymnehelper.EEG_data import EEGComputations
from phopymnehelper.analysis.computations.cache import DiskComputationCache
from phopymnehelper.analysis.computations.engine import GraphExecutor
from phopymnehelper.analysis.computations.protocol import DEFAULT_REGISTRY, ArtifactKind, ComputationNode, ComputationRegistry, RunContext, SessionFingerprint
from phopymnehelper.analysis.computations.specific.ADHD_sleep_intrusions import ThetaDeltaSleepIntrusionComputation
from phopymnehelper.analysis.computations.specific.bad_epochs import BadEpochsQCComputation
from phopymnehelper.analysis.computations.specific.EEG_Spectograms import EEGSpectrogramComputation

# xdf_file_name: TypeAlias = str # a name of the xdf file corresponding to a given session
# EEGComputationId: TypeAlias = Literal["time_independent_bad_channels", "bad_epochs", "raw_data_topo", "cwt", "spectogram"]

EEG_COMPUTATION_IDS_ORDERED: Tuple[types.EEGComputationId, ...] = ("time_independent_bad_channels", "bad_epochs", "raw_data_topo", "cwt", "spectogram")


_EEG_NODES_REGISTERED = False


def _register_node_if_absent(registry: ComputationRegistry, node: ComputationNode) -> None:
    if not registry.has(node.id):
        registry.register(node)


def _bad_ch_run(ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
    return EEGComputations.time_independent_bad_channels(ctx.raw, **dict(params))


def _topo_run(ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
    return EEGComputations.raw_data_topo(ctx.raw, **dict(params))


def _cwt_run(ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
    return EEGComputations.raw_morlet_cwt(ctx.raw, **dict(params))


def ensure_default_eeg_registry() -> ComputationRegistry:
    global _EEG_NODES_REGISTERED
    if _EEG_NODES_REGISTERED:
        return DEFAULT_REGISTRY
    # Idempotent: DEFAULT_REGISTRY may already hold nodes (e.g. register_eeg_computation_nodes elsewhere or a partial prior run).
    register_eeg_computation_nodes(DEFAULT_REGISTRY)
    _EEG_NODES_REGISTERED = True
    return DEFAULT_REGISTRY


def register_eeg_computation_nodes(registry: ComputationRegistry) -> None:
    """Register the standard EEG nodes on a custom registry (for tests or isolated graphs)."""
    _register_node_if_absent(registry, ComputationNode(id="time_independent_bad_channels", version="1", deps=(), kind=ArtifactKind.summary, run=_bad_ch_run))
    _register_node_if_absent(registry, ComputationNode(id="raw_data_topo", version="1", deps=("time_independent_bad_channels",), kind=ArtifactKind.stream, run=_topo_run))
    _register_node_if_absent(registry, ComputationNode(id="cwt", version="1", deps=("time_independent_bad_channels",), kind=ArtifactKind.stream, run=_cwt_run))
    _register_node_if_absent(registry, EEGSpectrogramComputation().to_computation_node())
    _register_node_if_absent(registry, BadEpochsQCComputation().to_computation_node())
    _register_node_if_absent(registry, ThetaDeltaSleepIntrusionComputation().to_computation_node())


def session_fingerprint_for_raw_or_path(raw: Any, path: Optional[Path] = None, mtime: Optional[float] = None) -> SessionFingerprint:
    if path is not None:
        p = Path(path)
        mt = mtime if mtime is not None else (p.stat().st_mtime if p.exists() else None)
        return SessionFingerprint.from_path(p, mtime=mt)
    desc = getattr(raw, "filenames", None)
    if desc and len(desc) > 0:
        try:
            return SessionFingerprint.from_path(Path(desc[0]), mtime=None)
        except Exception:
            pass
    info_d = getattr(raw, "info", None)
    if info_d is not None:
        d = info_d.get("description", "") or "unknown_raw"
        return SessionFingerprint(canonical_path=str(d), mtime=mtime, extra=())
    return SessionFingerprint(canonical_path="unknown_session", mtime=mtime, extra=())


def run_eeg_computations_graph(raw: Any, session: SessionFingerprint, global_params: Optional[Mapping[str, Any]] = None, goals: Optional[Sequence[types.EEGComputationId]] = None, registry: Optional[ComputationRegistry] = None, cache: Optional[DiskComputationCache] = None, use_cache: bool = True, parallel: bool = False, max_workers: int = 4) -> Dict[str, Any]:
    reg = registry
    if reg is None:
        reg = ensure_default_eeg_registry()
    else:
        if not reg.has("spectogram") or not reg.has("bad_epochs") or not reg.has("theta_delta_sleep_intrusion"):
            register_eeg_computation_nodes(reg)
    g = tuple(goals) if goals is not None else EEG_COMPUTATION_IDS_ORDERED
    ctx = RunContext(session=session, raw=raw)
    ex = GraphExecutor(reg, cache)
    return ex.run(ctx, g, global_params=global_params, use_cache=use_cache and cache is not None, parallel=parallel, max_workers=max_workers)


def run_eeg_graph_legacy_ordered(**kwargs: Any) -> Dict[str, Any]:
    """Run graph and return a dict with keys in the same order as EEGComputations.all_fcns_dict()."""
    out = run_eeg_computations_graph(**kwargs)
    return {k: out[k] for k in EEG_COMPUTATION_IDS_ORDERED if k in out}
