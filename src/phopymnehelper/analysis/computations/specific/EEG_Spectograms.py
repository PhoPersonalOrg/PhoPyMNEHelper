"""Timeline-oriented continuous EEG spectrogram from ``mne.io.Raw``.

See ``phopymnehelper/analysis/COMPUTATIONS_README.md`` for the computations contract.
This module centralizes default STFT parameters used by pyPhoTimeline and delegates
FFT work to :class:`phopymnehelper.EEG_data.EEGComputations`.
"""

from __future__ import annotations

import json
from typing import Any, Callable, ClassVar, Dict, FrozenSet, Mapping, Optional, Tuple

import mne

from phopymnehelper.EEG_data import EEGComputations
from phopymnehelper.analysis.computations.protocol import ArtifactKind, RunContext
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase

DEFAULT_SPECTROGRAM_NPERSEG = 1024
DEFAULT_SPECTROGRAM_NOVERLAP = 512

EEG_SPECTROGRAM_PARAM_KEYS: FrozenSet[str] = frozenset({"nperseg", "noverlap", "picks", "mask_bad_annotated_times"})

__all__ = [
    "DEFAULT_SPECTROGRAM_NPERSEG",
    "DEFAULT_SPECTROGRAM_NOVERLAP",
    "EEG_SPECTROGRAM_PARAM_KEYS",
    "EEGSpectrogramComputation",
    "compute_raw_eeg_spectrogram",
    "eeg_spectrogram_params_fingerprint",
    "filter_eeg_spectrogram_params",
]


def filter_eeg_spectrogram_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: params[k] for k in EEG_SPECTROGRAM_PARAM_KEYS if k in params}


def eeg_spectrogram_params_fingerprint(params: Mapping[str, Any]) -> str:
    f = filter_eeg_spectrogram_params(params)
    return json.dumps({k: f[k] for k in sorted(f.keys())}, sort_keys=True, default=str)


def compute_raw_eeg_spectrogram(raw: mne.io.Raw, *, nperseg: int = DEFAULT_SPECTROGRAM_NPERSEG, noverlap: int = DEFAULT_SPECTROGRAM_NOVERLAP, picks: Any = None, mask_bad_annotated_times: bool = True) -> Dict[str, Any]:
    """Compute continuous per-channel spectrogram; same defaults as timeline XDF processing."""
    return EEGComputations.raw_spectogram_working(raw, picks=picks, nperseg=nperseg, noverlap=noverlap, mask_bad_annotated_times=mask_bad_annotated_times)


class EEGSpectrogramComputation(SpecificComputationBase):
    """Continuous per-channel EEG spectrogram (STFT) for :class:`mne.io.Raw`, wired as DAG node ``"spectogram"``.

    Depends on ``time_independent_bad_channels`` so ``raw.info['bads']`` is populated before spectrogram picks.
    Params merged into ``compute`` are filtered to :data:`EEG_SPECTROGRAM_PARAM_KEYS` (``nperseg``, ``noverlap``,
    ``picks``, ``mask_bad_annotated_times``); see :func:`compute_raw_eeg_spectrogram` / ``EEGComputations.raw_spectogram_working``.

    **Default EEG graph** — the registered node is the same id used by :func:`phopymnehelper.analysis.computations.eeg_registry.ensure_default_eeg_registry`::

        Usage:
            from phopymnehelper.analysis.computations.eeg_registry import run_eeg_computations_graph, session_fingerprint_for_raw_or_path

            out = run_eeg_computations_graph(raw, session=session_fingerprint_for_raw_or_path(raw), goals=("spectogram",))
            spec = out["spectogram"]

    **Custom registry** — build a :class:`~phopymnehelper.analysis.computations.protocol.ComputationNode` and register it::

        from phopymnehelper.analysis.computations.protocol import ComputationRegistry
        from phopymnehelper.analysis.computations.specific.EEG_Spectograms import EEGSpectrogramComputation

        reg = ComputationRegistry()
        reg.register(EEGSpectrogramComputation().to_computation_node())

    **Direct call** (same signature as :attr:`~phopymnehelper.analysis.computations.protocol.ComputationNode.run`)::

        node = EEGSpectrogramComputation().to_computation_node()
        result = node.run(ctx, {"nperseg": 1024, "noverlap": 512}, dep_outputs={"time_independent_bad_channels": ...})
    """
    computation_id: ClassVar[str] = "spectogram"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ("time_independent_bad_channels",)
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.stream
    params_fingerprint_fn: ClassVar[Optional[Callable[[Mapping[str, Any]], str]]] = eeg_spectrogram_params_fingerprint


    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        if ctx.raw is None:
            raise ValueError("EEGSpectrogramComputation requires ctx.raw")
        kw = filter_eeg_spectrogram_params(params)
        return compute_raw_eeg_spectrogram(ctx.raw, **kw)
