"""DAG node wrapping quality-gated EEG activity manifold (PCA / IncrementalPCA / UMAP).

Batch path defaults to PCA + band-power features for cross-session trend comparability.
See ``phopymnehelper.analysis.computations.eeg_manifold`` for the pure compute helpers.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from phopymnehelper.analysis.computations.eeg_manifold import (
    DEFAULT_FEATURE_MODE,
    DEFAULT_FIT_HISTORY_SEC,
    DEFAULT_HOP_SEC,
    DEFAULT_MIN_QUALITY_TO_KEEP,
    DEFAULT_N_COMPONENTS,
    DEFAULT_WINDOW_SEC,
    EEGManifoldConfig,
    compute_eeg_manifold_from_detailed_df,
)
from phopymnehelper.analysis.computations.protocol import ArtifactKind, ComputationNode, RunContext
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase

EEG_MANIFOLD_PARAM_KEYS: FrozenSet[str] = frozenset(
    {
        "n_components",
        "feature_mode",
        "method",
        "min_quality_to_keep",
        "mask_unknown_quality",
        "window_sec",
        "hop_sec",
        "fit_history_sec",
        "filter_order",
        "max_nan_fraction",
        "min_channels",
        "allow_umap",
        "umap_n_neighbors",
        "umap_min_dist",
        "live_mode",
        "random_state",
        "session_bad_channels",
    }
)


class EEGManifoldComputation(SpecificComputationBase):
    """Session-level EEG activity manifold embedding as DAG node ``"eeg_manifold"``.

    Depends on ``time_independent_bad_channels`` so session-level bads gate channel selection.
    """

    computation_id: ClassVar[str] = "eeg_manifold"
    version: ClassVar[str] = "1"
    deps: ClassVar[Tuple[str, ...]] = ("time_independent_bad_channels",)
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.stream


    @classmethod
    def filter_eeg_manifold_params(cls, params: Mapping[str, Any]) -> Dict[str, Any]:
        return {k: params[k] for k in EEG_MANIFOLD_PARAM_KEYS if k in params}


    @classmethod
    def eeg_manifold_params_fingerprint(cls, params: Mapping[str, Any]) -> str:
        f = cls.filter_eeg_manifold_params(params)
        return json.dumps({k: f[k] for k in sorted(f.keys())}, sort_keys=True, default=str)


    @classmethod
    def config_from_params(cls, params: Mapping[str, Any], *, session_bad_channels: Optional[List[str]] = None) -> EEGManifoldConfig:
        """Build a validated config; batch defaults are PCA / band_power / 3 components."""
        kw = cls.filter_eeg_manifold_params(params)
        cfg = EEGManifoldConfig(
            n_components=int(kw.get("n_components", DEFAULT_N_COMPONENTS)),
            feature_mode=kw.get("feature_mode", DEFAULT_FEATURE_MODE),
            method=kw.get("method", "pca"),
            min_quality_to_keep=float(kw.get("min_quality_to_keep", DEFAULT_MIN_QUALITY_TO_KEEP)),
            mask_unknown_quality=bool(kw.get("mask_unknown_quality", True)),
            window_sec=float(kw.get("window_sec", DEFAULT_WINDOW_SEC)),
            hop_sec=float(kw.get("hop_sec", DEFAULT_HOP_SEC)),
            fit_history_sec=float(kw.get("fit_history_sec", DEFAULT_FIT_HISTORY_SEC)),
            filter_order=int(kw.get("filter_order", 4)),
            max_nan_fraction=float(kw.get("max_nan_fraction", 0.35)),
            min_channels=int(kw.get("min_channels", 4)),
            session_bad_channels=list(session_bad_channels if session_bad_channels is not None else kw.get("session_bad_channels", []) or []),
            allow_umap=bool(kw.get("allow_umap", True)),
            umap_n_neighbors=int(kw.get("umap_n_neighbors", 15)),
            umap_min_dist=float(kw.get("umap_min_dist", 0.1)),
            live_mode=bool(kw.get("live_mode", False)),
            random_state=int(kw.get("random_state", 0)),
        )
        return cfg.validated()


    @classmethod
    def raw_to_eeg_df(cls, raw: mne.io.BaseRaw) -> Tuple[pd.DataFrame, List[str], float]:
        """Convert MNE Raw EEG channels to a manifold ``detailed_df`` (``t`` + channel columns)."""
        picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude=[])
        if picks.size == 0:
            raise ValueError("EEGManifoldComputation: no EEG channels in raw")
        channel_names = [raw.ch_names[i] for i in picks]
        data = np.asarray(raw.get_data(picks=picks), dtype=float)
        t = np.asarray(raw.times, dtype=float)
        eeg_df = pd.DataFrame({"t": t, **{name: data[i] for i, name in enumerate(channel_names)}})
        sfreq = float(raw.info.get("sfreq", 0.0) or 0.0)
        if sfreq <= 0:
            sfreq = float(1.0 / np.median(np.diff(t))) if t.size > 1 else 128.0
        return eeg_df, channel_names, sfreq


    @classmethod
    def session_bad_channels_from_dep(cls, dep_outputs: Mapping[str, Any]) -> List[str]:
        bad_result = dep_outputs.get("time_independent_bad_channels")
        if not isinstance(bad_result, Mapping):
            return []
        bads = bad_result.get("all_bad_channels") or []
        return [str(c) for c in bads]


    @classmethod
    def compute_from_raw(cls, raw: mne.io.BaseRaw, *, params: Optional[Mapping[str, Any]] = None, session_bad_channels: Optional[List[str]] = None, quality_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """End-to-end Raw → cache-safe manifold result dict."""
        params = params or {}
        cfg = cls.config_from_params(params, session_bad_channels=session_bad_channels)
        eeg_df, channel_names, sfreq = cls.raw_to_eeg_df(raw)
        manifold_df = compute_eeg_manifold_from_detailed_df(eeg_df, channel_names, sfreq, quality_df=quality_df, config=cfg)
        component_names = cfg.component_names()
        if manifold_df is None or len(manifold_df) == 0 or "t" not in manifold_df.columns:
            return {
                "times": np.asarray([], dtype=float),
                "embedding": np.zeros((0, cfg.n_components), dtype=float),
                "component_names": component_names,
                "config_dict": asdict(cfg),
                "n_windows": 0,
            }
        times = np.asarray(manifold_df["t"], dtype=float)
        embedding = np.column_stack([np.asarray(manifold_df[c], dtype=float) if c in manifold_df.columns else np.full(len(manifold_df), np.nan) for c in component_names])
        return {
            "times": times,
            "embedding": embedding,
            "component_names": component_names,
            "config_dict": asdict(cfg),
            "n_windows": int(len(times)),
        }


    def to_computation_node(self) -> ComputationNode:
        return ComputationNode(
            id=self.computation_id,
            version=self.version,
            deps=self.deps,
            kind=self.artifact_kind,
            run=self.run_fn(),
            params_fingerprint=self.eeg_manifold_params_fingerprint,
        )


    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        if ctx.raw is None:
            raise ValueError("EEGManifoldComputation requires ctx.raw")
        session_bads = self.session_bad_channels_from_dep(dep_outputs)
        return self.compute_from_raw(ctx.raw, params=params, session_bad_channels=session_bads, quality_df=None)


__all__ = [
    "EEG_MANIFOLD_PARAM_KEYS",
    "EEGManifoldComputation",
]
