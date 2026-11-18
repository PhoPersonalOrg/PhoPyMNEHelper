"""Tests for phopymnehelper.analysis.computations.eeg_manifold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phopymnehelper.analysis.computations.eeg_manifold import (
    EEGManifoldConfig,
    apply_quality_and_session_gates,
    build_feature_matrix,
    component_column_names,
    compute_eeg_manifold_from_detailed_df,
    resolve_usable_channels,
)


def _synth_eeg(n_ch: int = 8, duration_s: float = 30.0, sfreq: float = 128.0, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sfreq)
    t = np.arange(n, dtype=float) / sfreq
    names = [f"CH{i}" for i in range(n_ch)]
    data = {name: rng.standard_normal(n) * 20.0 for name in names}
    data["t"] = t
    return pd.DataFrame(data), names


def test_resolve_usable_channels_drops_session_bads() -> None:
    assert resolve_usable_channels(["AF3", "F7", "F3"], ["F7"]) == ["AF3", "F3"]


def test_component_column_names_clamped() -> None:
    assert component_column_names(3) == ["PC1", "PC2", "PC3"]
    assert component_column_names(1) == ["PC1", "PC2"]
    assert component_column_names(9) == ["PC1", "PC2", "PC3", "PC4", "PC5"]


def test_quality_gate_masks_low_quality_channel() -> None:
    eeg, names = _synth_eeg(n_ch=4, duration_s=5.0)
    q = eeg[["t"]].copy()
    for name in names:
        q[name] = 4.0
    q["CH1"] = 1.0
    cfg = EEGManifoldConfig(min_quality_to_keep=2.0, session_bad_channels=["CH3"])
    gated, usable = apply_quality_and_session_gates(eeg, q, names, cfg)
    assert "CH3" not in usable
    assert "CH1" in usable
    assert np.all(~np.isfinite(gated["CH1"].to_numpy()))
    assert np.any(np.isfinite(gated["CH0"].to_numpy()))


def test_build_feature_matrix_band_power_shape() -> None:
    eeg, names = _synth_eeg(n_ch=6, duration_s=20.0, sfreq=128.0)
    cfg = EEGManifoldConfig(feature_mode="band_power", window_sec=2.0, hop_sec=1.0, min_channels=4)
    X, centers, cols = build_feature_matrix(eeg, names, 128.0, cfg)
    assert X.ndim == 2
    assert centers.ndim == 1
    assert X.shape[0] == centers.shape[0]
    assert X.shape[0] > 0
    assert X.shape[1] == len(cols) * len(cfg.validated().bands)


def test_compute_manifold_pca_output_columns() -> None:
    eeg, names = _synth_eeg(n_ch=8, duration_s=25.0, sfreq=128.0)
    cfg = EEGManifoldConfig(n_components=3, method="pca", feature_mode="band_power", window_sec=2.0, hop_sec=1.0, fit_history_sec=10.0)
    out = compute_eeg_manifold_from_detailed_df(eeg, names, 128.0, config=cfg)
    assert "t" in out.columns
    assert list(out.columns[1:]) == ["PC1", "PC2", "PC3"]
    assert len(out) > 0
    assert np.any(np.isfinite(out["PC1"].to_numpy()))


def test_compute_manifold_excludes_session_bads_and_low_quality() -> None:
    eeg, names = _synth_eeg(n_ch=6, duration_s=20.0, sfreq=128.0)
    q = eeg[["t"]].copy()
    for name in names:
        q[name] = 4.0
    q["CH0"] = 1.0
    cfg = EEGManifoldConfig(n_components=2, method="pca", window_sec=2.0, hop_sec=1.0, session_bad_channels=["CH5"], min_quality_to_keep=2.0, min_channels=3)
    out = compute_eeg_manifold_from_detailed_df(eeg, names, 128.0, quality_df=q, config=cfg)
    assert len(out) > 0
    assert "PC1" in out.columns and "PC2" in out.columns


def test_live_mode_forces_umap_to_incremental_pca() -> None:
    cfg = EEGManifoldConfig(method="umap", live_mode=True).validated()
    assert cfg.method == "incremental_pca"


def test_raw_stats_feature_mode() -> None:
    eeg, names = _synth_eeg(n_ch=6, duration_s=15.0, sfreq=64.0)
    cfg = EEGManifoldConfig(feature_mode="raw_stats", method="pca", n_components=2, window_sec=1.5, hop_sec=0.75, min_channels=3)
    out = compute_eeg_manifold_from_detailed_df(eeg, names, 64.0, config=cfg)
    assert len(out) > 0
    assert out.shape[1] == 3


@pytest.mark.slow
def test_umap_offline_optional() -> None:
    umap = pytest.importorskip("umap")
    _ = umap
    eeg, names = _synth_eeg(n_ch=8, duration_s=20.0, sfreq=64.0)
    cfg = EEGManifoldConfig(feature_mode="raw_stats", method="umap", n_components=2, window_sec=1.0, hop_sec=0.5, min_channels=4, umap_n_neighbors=5)
    out = compute_eeg_manifold_from_detailed_df(eeg, names, 64.0, config=cfg)
    assert len(out) > 5
    assert np.any(np.isfinite(out["PC1"].to_numpy()))
