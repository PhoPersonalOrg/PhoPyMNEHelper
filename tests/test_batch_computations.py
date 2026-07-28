"""Unit and integration tests for session computations, batch execution, and result aggregation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from phopymnehelper.analysis.computations import (
    ArtifactKind,
    BatchResultsAggregator,
    BatchSessionExecutor,
    BatchSessionResult,
    ComputationRegistry,
    DiskComputationCache,
    LoadedSessionContext,
    SessionComputationBase,
    SessionComputationRegistry,
    SessionFingerprint,
    session_computation,
)


# Fixture for isolated test registry
@pytest.fixture
def isolated_registry() -> ComputationRegistry:
    return ComputationRegistry()


def test_session_computation_decorator(isolated_registry: ComputationRegistry) -> None:
    @session_computation(
        id="test_node_fn",
        version="1.0",
        deps=(),
        kind=ArtifactKind.summary,
        registry=isolated_registry,
    )
    def compute_summary_metric(ctx: LoadedSessionContext, params: Dict[str, Any], dep_outputs: Dict[str, Any]) -> Dict[str, float]:
        scale = params.get("scale", 1.0)
        return {"metric_value": 42.0 * scale}

    assert isolated_registry.has("test_node_fn")
    node = isolated_registry.get("test_node_fn")
    ctx = LoadedSessionContext(session=SessionFingerprint(canonical_path="/fake/path.fif"))
    result = node.run(ctx, {"scale": 2.0}, {})
    assert result == {"metric_value": 84.0}


def test_session_computation_class(isolated_registry: ComputationRegistry) -> None:
    class CustomAlphaComputation(SessionComputationBase):
        computation_id = "custom_alpha"
        version = "1.0"
        deps = ()
        artifact_kind = ArtifactKind.summary

        def compute(self, ctx: LoadedSessionContext, params: Dict[str, Any], dep_outputs: Dict[str, Any]) -> Dict[str, Any]:
            return {"alpha_power": 12.5, "peak_channel": "Cz"}

    comp_reg = SessionComputationRegistry(inner_registry=isolated_registry)
    node = comp_reg.register(CustomAlphaComputation)

    assert isolated_registry.has("custom_alpha")
    ctx = LoadedSessionContext(session=SessionFingerprint(canonical_path="/fake/session.fif"))
    out = node.run(ctx, {}, {})
    assert out["alpha_power"] == 12.5
    assert out["peak_channel"] == "Cz"


def test_batch_executor_parallel_and_isolation(isolated_registry: ComputationRegistry) -> None:
    @session_computation(
        id="node_a",
        version="1.0",
        kind=ArtifactKind.summary,
        registry=isolated_registry,
    )
    def compute_a(ctx: LoadedSessionContext, params: Dict[str, Any], dep_outputs: Dict[str, Any]) -> Dict[str, str]:
        if "fail" in ctx.session.canonical_path:
            raise RuntimeError("Simulated session failure")
        return {"session_name": ctx.session.canonical_path}

    contexts = [
        LoadedSessionContext(session=SessionFingerprint(canonical_path="session_1.fif")),
        LoadedSessionContext(session=SessionFingerprint(canonical_path="session_fail.fif")),
        LoadedSessionContext(session=SessionFingerprint(canonical_path="session_3.fif")),
    ]

    executor = BatchSessionExecutor(registry=isolated_registry)
    results = executor.run_batch(
        session_contexts=contexts,
        goals=["node_a"],
        parallel_sessions=True,
        max_session_workers=2,
    )

    assert len(results) == 3
    assert results[0].success is True
    assert results[0].outputs["node_a"] == {"session_name": "session_1.fif"}

    assert results[1].success is False
    assert isinstance(results[1].error, RuntimeError)

    assert results[2].success is True
    assert results[2].outputs["node_a"] == {"session_name": "session_3.fif"}


def test_batch_aggregator_export() -> None:
    results = [
        BatchSessionResult(
            session_id="s1",
            session_path="/data/s1.fif",
            success=True,
            outputs={"node_a": {"val": 10.0, "status": "ok"}},
            execution_time_sec=0.5,
        ),
        BatchSessionResult(
            session_id="s2",
            session_path="/data/s2.fif",
            success=True,
            outputs={"node_a": {"val": 20.0, "status": "ok"}},
            execution_time_sec=0.6,
        ),
        BatchSessionResult(
            session_id="s3",
            session_path="/data/s3.fif",
            success=False,
            outputs={},
            execution_time_sec=0.1,
            error=ValueError("Corrupted data"),
        ),
    ]

    agg = BatchResultsAggregator(results)
    df = agg.to_dataframe()
    assert len(df) == 3
    assert "node_a_val" in df.columns
    assert df.loc[df["session_id"] == "s1", "node_a_val"].iloc[0] == 10.0

    report = agg.summary_report()
    assert report["total_sessions"] == 3
    assert report["successful_sessions"] == 2
    assert report["failed_sessions"] == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "report.csv"
        json_path = Path(tmpdir) / "report.json"
        agg.export_csv(csv_path)
        agg.export_json(json_path)
        assert csv_path.exists()
        assert json_path.exists()
