"""Tests for AmphPeriEventSpectrogramComputation DAG node."""

from __future__ import annotations

import numpy as np
import pytest

from phopymnehelper.analysis.computations.protocol import (
    ArtifactKind,
    ComputationRegistry,
    RunContext,
    SessionFingerprint,
)
from phopymnehelper.analysis.computations.specific.amph_peri_event_spectrogram import (
    AMPH_PERI_EVENT_PARAM_KEYS,
    AmphPeriEventSpectrogramComputation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_spectrogram(n_ch: int = 2, n_freqs: int = 64, n_times: int = 300, t_start_unix: float = 1_000_000.0, hop_s: float = 4.0):
    """Build a minimal spectrogram dict matching the upstream spectogram node output."""
    rng = np.random.default_rng(42)
    freqs = np.linspace(0.5, 60.0, n_freqs)
    t_unix = t_start_unix + np.arange(n_times) * hop_s
    Sxx = rng.random((n_ch, n_freqs, n_times), dtype=np.float32) + 0.01  # avoid zeros for dB
    ch_names = [f"ch{i}" for i in range(n_ch)]
    return dict(Sxx=Sxx, freqs=freqs, t_unix=t_unix, ch_names=ch_names)


def _make_ctx_with_dose_events(amph_t_unix, *, canonical_path: str = "/fake/session.xdf"):
    """Build a RunContext with dose event timestamps in extras."""
    session = SessionFingerprint(canonical_path=canonical_path)
    return RunContext(session=session, raw=None, extras={"amph_dose_t_unix": amph_t_unix})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_filter_params_extracts_only_relevant_keys():
    params = {"pre_s": 300.0, "post_s": 900.0, "bands": {}, "nperseg": 1024, "extra": "ignored"}
    filtered = AmphPeriEventSpectrogramComputation.filter_params(params)
    assert set(filtered.keys()) == {"pre_s", "post_s", "bands"}


def test_params_fingerprint_ignores_unrelated_keys():
    fp1 = AmphPeriEventSpectrogramComputation.amph_peri_event_params_fingerprint({"pre_s": 600, "post_s": 1800, "nperseg": 1024})
    fp2 = AmphPeriEventSpectrogramComputation.amph_peri_event_params_fingerprint({"post_s": 1800, "pre_s": 600})
    assert fp1 == fp2


def test_to_computation_node_metadata():
    comp = AmphPeriEventSpectrogramComputation()
    node = comp.to_computation_node()
    assert node.id == "amph_peri_event_spectrogram"
    assert node.version == "1"
    assert node.deps == ("spectogram",)
    assert node.kind == ArtifactKind.object
    assert callable(node.run)


def test_compute_with_synthetic_data():
    spec = _make_synthetic_spectrogram(n_ch=2, n_freqs=32, n_times=600, hop_s=4.0)
    t_unix = spec["t_unix"]
    # Place two dose events well inside the spectrogram span
    dose_ts = [float(t_unix[150]), float(t_unix[400])]
    ctx = _make_ctx_with_dose_events(dose_ts)
    dep_outputs = {"spectogram": spec}

    comp = AmphPeriEventSpectrogramComputation()
    result = comp.compute(ctx, {"pre_s": 60.0, "post_s": 120.0}, dep_outputs)

    assert isinstance(result, dict)
    assert result["n_events"] == 2
    assert len(result["peri_slices"]) == 2
    assert result["Sxx_pre_mean"].size > 0
    assert result["Sxx_post_mean"].size > 0

    # Each peri slice should have the expected keys
    for ps in result["peri_slices"]:
        assert "event_t_unix" in ps
        assert "peri" in ps
        assert "bands" in ps
        assert "coverage_pre_s" in ps
        assert "coverage_post_s" in ps
    ## END for ps in result["peri_slices"]...


def test_compute_no_events_in_window():
    spec = _make_synthetic_spectrogram(n_ch=1, n_freqs=16, n_times=100, hop_s=4.0)
    # Dose event far outside the spectrogram window
    dose_ts = [0.0]
    ctx = _make_ctx_with_dose_events(dose_ts)
    dep_outputs = {"spectogram": spec}

    comp = AmphPeriEventSpectrogramComputation()
    result = comp.compute(ctx, {}, dep_outputs)

    assert result["n_events"] == 0
    assert len(result["peri_slices"]) == 0
    assert result["Sxx_pre_mean"].size == 0


def test_compute_raises_without_dose_events_in_extras():
    spec = _make_synthetic_spectrogram()
    ctx = RunContext(session=SessionFingerprint(canonical_path="/fake/session.xdf"), raw=None, extras={})
    dep_outputs = {"spectogram": spec}

    comp = AmphPeriEventSpectrogramComputation()
    with pytest.raises(ValueError, match="amph_dose_t_unix"):
        comp.compute(ctx, {}, dep_outputs)


def test_compute_raises_without_spectrogram():
    ctx = _make_ctx_with_dose_events([1_000_100.0])
    dep_outputs = {"spectogram": None}

    comp = AmphPeriEventSpectrogramComputation()
    with pytest.raises(ValueError, match="upstream"):
        comp.compute(ctx, {}, dep_outputs)


def test_dag_registration_and_execution():
    """Register the node in an isolated registry, wire a fake spectogram dep, and run via GraphExecutor."""
    from phopymnehelper.analysis.computations.engine import GraphExecutor
    from phopymnehelper.analysis.computations.protocol import ComputationNode

    registry = ComputationRegistry()

    # Minimal fake upstream "spectogram" node
    spec_data = _make_synthetic_spectrogram(n_ch=2, n_freqs=32, n_times=400, hop_s=4.0)
    def fake_spec_run(ctx, params, dep_outputs):
        return spec_data
    registry.register(ComputationNode(id="spectogram", version="1", deps=(), kind=ArtifactKind.stream, run=fake_spec_run))

    # Also need "time_independent_bad_channels" since spectogram depends on it in real registry,
    # but our fake spectogram has no deps, so skip it.

    # Register the peri-event node
    comp = AmphPeriEventSpectrogramComputation()
    registry.register(comp.to_computation_node())

    # Build context with dose events inside the spectrogram window
    t_unix = spec_data["t_unix"]
    dose_ts = [float(t_unix[200])]
    ctx = _make_ctx_with_dose_events(dose_ts)

    executor = GraphExecutor(registry, cache=None)
    results = executor.run(ctx, goals=["amph_peri_event_spectrogram"], global_params={"pre_s": 40.0, "post_s": 80.0})

    assert "amph_peri_event_spectrogram" in results
    out = results["amph_peri_event_spectrogram"]
    assert out["n_events"] == 1
    assert len(out["peri_slices"]) == 1
