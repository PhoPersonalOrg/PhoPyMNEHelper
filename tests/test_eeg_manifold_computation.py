"""Tests for EEGManifoldComputation DAG node (Raw → manifold result)."""

from __future__ import annotations

import numpy as np
import mne
import pytest

from phopymnehelper.analysis.computations.eeg_registry import EEG_COMPUTATION_IDS_ORDERED, EEGComputationRegistryHelper
from phopymnehelper.analysis.computations.protocol import RunContext, SessionFingerprint
from phopymnehelper.analysis.computations.specific.eeg_manifold_computation import EEGManifoldComputation


def _synth_raw(n_ch: int = 8, duration_s: float = 25.0, sfreq: float = 128.0, seed: int = 0) -> mne.io.RawArray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sfreq)
    data = rng.standard_normal((n_ch, n)) * 20e-6
    ch_names = [f"CH{i}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_ch)
    return mne.io.RawArray(data, info, verbose=False)


def test_compute_from_raw_pca_shape() -> None:
    raw = _synth_raw()
    result = EEGManifoldComputation.compute_from_raw(
        raw,
        params={"n_components": 3, "method": "pca", "feature_mode": "band_power", "window_sec": 2.0, "hop_sec": 1.0, "fit_history_sec": 10.0, "min_channels": 4},
        session_bad_channels=[],
    )
    assert result["n_windows"] > 0
    assert result["times"].size == result["n_windows"]
    assert result["embedding"].shape == (result["n_windows"], 3)
    assert result["component_names"] == ["PC1", "PC2", "PC3"]
    assert np.any(np.isfinite(result["embedding"]))


def test_compute_uses_session_bads_from_dep() -> None:
    raw = _synth_raw(n_ch=6)
    node = EEGManifoldComputation()
    ctx = RunContext(session=SessionFingerprint(canonical_path="test_manifold", mtime=None, extra=()), raw=raw, extras={})
    dep_outputs = {"time_independent_bad_channels": {"all_bad_channels": ["CH0", "CH5"]}}
    result = node.compute(ctx, {"n_components": 2, "method": "pca", "window_sec": 2.0, "hop_sec": 1.0, "min_channels": 3}, dep_outputs)
    assert result["n_windows"] > 0
    assert result["embedding"].shape[1] == 2
    assert result["config_dict"]["session_bad_channels"] == ["CH0", "CH5"]


def test_eeg_manifold_in_default_ordered_goals() -> None:
    assert "eeg_manifold" in EEG_COMPUTATION_IDS_ORDERED
    reg = EEGComputationRegistryHelper.ensure_default_eeg_registry()
    assert reg.has("eeg_manifold")
