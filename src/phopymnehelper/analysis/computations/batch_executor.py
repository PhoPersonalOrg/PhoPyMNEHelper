"""Batch computation executor for running session-level DAG computations across multiple sessions."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from phopymnehelper.analysis.computations.cache import DiskComputationCache
from phopymnehelper.analysis.computations.engine import GraphExecutor
from phopymnehelper.analysis.computations.protocol import DEFAULT_REGISTRY, ComputationRegistry, SessionFingerprint
from phopymnehelper.analysis.computations.session_computation import LoadedSessionContext


@dataclass
class BatchSessionResult:
    """Execution result for a single session in a batch run."""

    session_id: str
    session_path: Optional[str]
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    execution_time_sec: float = 0.0
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchSessionExecutor:
    """Orchestrates single-session computations across multiple sessions independently."""

    def __init__(
        self,
        registry: Optional[ComputationRegistry] = None,
        cache: Optional[DiskComputationCache] = None,
        use_cache: bool = True,
        on_session_start: Optional[Callable[[str, LoadedSessionContext], None]] = None,
        on_session_complete: Optional[Callable[[BatchSessionResult], None]] = None,
        on_session_failed: Optional[Callable[[str, Exception], None]] = None,
    ) -> None:
        self.registry = registry if registry is not None else DEFAULT_REGISTRY
        self.cache = cache
        self.use_cache = use_cache
        self.on_session_start = on_session_start
        self.on_session_complete = on_session_complete
        self.on_session_failed = on_session_failed


    def run_session(
        self,
        ctx: LoadedSessionContext,
        goals: Sequence[str],
        global_params: Optional[Mapping[str, Any]] = None,
        params_by_node: Optional[Mapping[str, Mapping[str, Any]]] = None,
        parallel_node_execution: bool = False,
        max_node_workers: int = 4,
    ) -> BatchSessionResult:
        """Run computations for a single session with error isolation and performance timing."""
        start_t = time.perf_counter()
        session_id = ctx.session.canonical_path

        if self.on_session_start is not None:
            try:
                self.on_session_start(session_id, ctx)
            except Exception:
                pass

        try:
            executor = GraphExecutor(self.registry, cache=self.cache)
            outputs = executor.run(
                ctx=ctx,
                goals=goals,
                global_params=global_params,
                params_by_node=params_by_node,
                use_cache=self.use_cache and self.cache is not None,
                parallel=parallel_node_execution,
                max_workers=max_node_workers,
            )
            elapsed = time.perf_counter() - start_t
            result = BatchSessionResult(
                session_id=session_id,
                session_path=ctx.session.canonical_path,
                success=True,
                outputs=outputs,
                execution_time_sec=elapsed,
                metadata=dict(ctx.metadata),
            )
            if self.on_session_complete is not None:
                try:
                    self.on_session_complete(result)
                except Exception:
                    pass
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - start_t
            result = BatchSessionResult(
                session_id=session_id,
                session_path=ctx.session.canonical_path,
                success=False,
                outputs={},
                execution_time_sec=elapsed,
                error=exc,
                metadata=dict(ctx.metadata),
            )
            if self.on_session_failed is not None:
                try:
                    self.on_session_failed(session_id, exc)
                except Exception:
                    pass
            if self.on_session_complete is not None:
                try:
                    self.on_session_complete(result)
                except Exception:
                    pass
            return result


    def run_batch(
        self,
        session_contexts: Sequence[LoadedSessionContext],
        goals: Sequence[str],
        global_params: Optional[Mapping[str, Any]] = None,
        params_by_node: Optional[Mapping[str, Mapping[str, Any]]] = None,
        parallel_sessions: bool = True,
        max_session_workers: int = 4,
        execution_mode: str = "thread",
    ) -> List[BatchSessionResult]:
        """Run batch computations across multiple loaded session contexts."""
        if not parallel_sessions or max_session_workers <= 1 or len(session_contexts) <= 1:
            results: List[BatchSessionResult] = []
            for ctx in session_contexts:
                res = self.run_session(
                    ctx=ctx,
                    goals=goals,
                    global_params=global_params,
                    params_by_node=params_by_node,
                )
                results.append(res)
            ## END for ctx in session_contexts...
            return results

        results_by_idx: Dict[int, BatchSessionResult] = {}
        n_workers = min(max_session_workers, len(session_contexts))

        if execution_mode.lower() == "process":
            pool_cls = ProcessPoolExecutor
        else:
            pool_cls = ThreadPoolExecutor

        with pool_cls(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(
                    self.run_session,
                    ctx,
                    goals,
                    global_params,
                    params_by_node,
                ): idx
                for idx, ctx in enumerate(session_contexts)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    ctx = session_contexts[idx]
                    res = BatchSessionResult(
                        session_id=ctx.session.canonical_path,
                        session_path=ctx.session.canonical_path,
                        success=False,
                        outputs={},
                        execution_time_sec=0.0,
                        error=exc,
                        metadata=dict(ctx.metadata),
                    )
                results_by_idx[idx] = res
            ## END for fut in as_completed...

        return [results_by_idx[i] for i in range(len(session_contexts))]


__all__ = [
    "BatchSessionResult",
    "BatchSessionExecutor",
]
