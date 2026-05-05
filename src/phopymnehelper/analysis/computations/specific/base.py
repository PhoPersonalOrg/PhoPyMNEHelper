"""Base class for specific computations matching the authoring protocol in COMPUTATIONS_README."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Mapping, Optional, Tuple

from phopymnehelper.analysis.computations.protocol import ArtifactKind, ComputationNode, RunContext, RunFn


class SpecificComputationBase(ABC):
    """Declarative metadata plus ``compute``, optional ``build_output_renders``, and lifecycle hooks.

    Register with the graph via :meth:`to_computation_node` or :meth:`run_fn`. Hook methods run only when
    the executor invokes ``run`` (not on cache hits).
    """

    computation_id: ClassVar[str] = ""
    version: ClassVar[str] = ""
    deps: ClassVar[Tuple[str, ...]] = ()
    artifact_kind: ClassVar[ArtifactKind] = ArtifactKind.object
    params_fingerprint_fn: ClassVar[Optional[Callable[[Mapping[str, Any]], str]]] = None


    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", frozenset()):
            return
        if not getattr(cls, "computation_id", "") or not getattr(cls, "version", ""):
            raise TypeError(f"{cls.__qualname__} must define non-empty class attributes computation_id and version")


    @abstractmethod
    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        """Core analysis step; same contract as :attr:`ComputationNode.run`."""


    def build_output_renders(self, result: Any, **kwargs: Any) -> Any:
        """Map a cache-safe ``compute`` result to UI or timeline artifacts; default is no rendering."""
        return None


    def on_computation_start(self, ctx: RunContext, params: Mapping[str, Any]) -> None:
        """Called immediately before :meth:`compute` when this node runs (not on cache hit)."""


    def on_computation_complete(self, result: Any, cache_key_or_none: Optional[str] = None, meta: Optional[Mapping[str, Any]] = None) -> None:
        """Called after :meth:`compute` returns successfully."""


    def on_computation_failed(self, exc: BaseException) -> None:
        """Called when :meth:`compute` raises; re-raises after this returns."""


    def _run_with_hooks(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        self.on_computation_start(ctx, params)
        try:
            result = self.compute(ctx, params, dep_outputs)
            self.on_computation_complete(result, None, None)
            return result
        except (BaseException, RuntimeError) as exc:
            self.on_computation_failed(exc)
            raise


    def run_fn(self) -> RunFn:
        """Callable suitable for :attr:`ComputationNode.run`, including lifecycle hooks."""
        return self._run_with_hooks


    def to_computation_node(self) -> ComputationNode:
        """Build a :class:`ComputationNode` for this instance (metadata from the subclass, ``run`` from hooks)."""
        cls = type(self)
        return ComputationNode(id=cls.computation_id, version=cls.version, deps=cls.deps, kind=cls.artifact_kind, run=self.run_fn(), params_fingerprint=cls.params_fingerprint_fn)


__all__ = ["SpecificComputationBase"]
