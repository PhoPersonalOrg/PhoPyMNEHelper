"""Session computation abstractions: LoadedSessionContext, SessionComputationBase, @session_computation decorator, and SessionComputationRegistry."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

from phopymnehelper.analysis.computations.protocol import (
    DEFAULT_REGISTRY,
    ArtifactKind,
    ComputationNode,
    ComputationRegistry,
    RunContext,
    SessionFingerprint,
)
from phopymnehelper.analysis.computations.specific.base import SpecificComputationBase


@dataclass
class LoadedSessionContext(RunContext):
    """Standardized context encapsulating loaded multi-modal data for a session."""

    eeg_raw: Any = None
    motion_raw: Optional[Any] = None
    stream_infos: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.raw is None and self.eeg_raw is not None:
            self.raw = self.eeg_raw
        elif self.eeg_raw is None and self.raw is not None:
            self.eeg_raw = self.raw


class SessionComputationBase(SpecificComputationBase):
    """Base class for session computations operating on LoadedSessionContext."""

    required_modalities: ClassVar[Tuple[str, ...]] = ()
    tags: ClassVar[Tuple[str, ...]] = ()

    @abstractmethod
    def compute(self, ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
        """Core analysis step."""


def session_computation(id: str, version: str = "1.0", deps: Tuple[str, ...] = (), kind: ArtifactKind = ArtifactKind.object, registry: Optional[ComputationRegistry] = None, params_fingerprint: Optional[Callable[[Mapping[str, Any]], str]] = None) -> Callable[[Callable[..., Any]], ComputationNode]:
    """Decorator to convert a pure function into a registered ComputationNode.

    Supports signatures accepting (ctx, params, dep_outputs) or (ctx, **kwargs).
    """

    def decorator(fn: Callable[..., Any]) -> ComputationNode:
        sig = inspect.signature(fn)
        accepts_three_args = len(sig.parameters) == 3

        def _run_wrapper(ctx: RunContext, params: Mapping[str, Any], dep_outputs: Mapping[str, Any]) -> Any:
            if not isinstance(ctx, LoadedSessionContext):
                session_ctx = LoadedSessionContext(session=ctx.session, raw=ctx.raw, extras=ctx.extras)
            else:
                session_ctx = ctx

            if accepts_three_args:
                return fn(session_ctx, params, dep_outputs)
            return fn(session_ctx, **dict(params))

        node = ComputationNode(
            id=id,
            version=version,
            deps=deps,
            kind=kind,
            run=_run_wrapper,
            params_fingerprint=params_fingerprint,
        )

        target_reg = registry if registry is not None else DEFAULT_REGISTRY
        if not target_reg.has(id):
            target_reg.register(node)
        return node

    return decorator


class SessionComputationRegistry:
    """Wrapper managing computation node registration, query, and package auto-discovery."""

    def __init__(self, inner_registry: Optional[ComputationRegistry] = None) -> None:
        self.inner_registry = inner_registry if inner_registry is not None else DEFAULT_REGISTRY


    def register(self, node_or_class: Union[ComputationNode, SpecificComputationBase, Type[SpecificComputationBase]]) -> ComputationNode:
        """Explicitly register a ComputationNode instance or SpecificComputationBase subclass/instance."""
        if isinstance(node_or_class, ComputationNode):
            node = node_or_class
        elif isinstance(node_or_class, SpecificComputationBase):
            node = node_or_class.to_computation_node()
        elif isinstance(node_or_class, type) and issubclass(node_or_class, SpecificComputationBase):
            node = node_or_class().to_computation_node()
        else:
            raise TypeError(f"Cannot register object of type {type(node_or_class).__name__}")

        if not self.inner_registry.has(node.id):
            self.inner_registry.register(node)
        return node


    def discover_modules(self, package_or_path: Union[str, Path]) -> List[ComputationNode]:
        """Auto-discover and register SpecificComputationBase classes within a package or module directory."""
        discovered: List[ComputationNode] = []

        if isinstance(package_or_path, Path) or (isinstance(package_or_path, str) and Path(package_or_path).is_dir()):
            dir_path = Path(package_or_path)
            for file_path in dir_path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    discovered.extend(self._extract_and_register_from_module(mod))
            ## END for file_path in dir_path.glob("*.py")...

        else:
            pkg = importlib.import_module(str(package_or_path))
            discovered.extend(self._extract_and_register_from_module(pkg))
            if hasattr(pkg, "__path__"):
                for _, sub_name, is_pkg in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
                    try:
                        sub_mod = importlib.import_module(sub_name)
                        discovered.extend(self._extract_and_register_from_module(sub_mod))
                    except Exception:
                        pass
                ## END for _, sub_name, is_pkg in pkgutil.walk_packages...

        return discovered


    def _extract_and_register_from_module(self, mod: Any) -> List[ComputationNode]:
        nodes: List[ComputationNode] = []
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, SpecificComputationBase) and obj is not SpecificComputationBase and obj is not SessionComputationBase:
                if getattr(obj, "computation_id", "") and getattr(obj, "version", ""):
                    try:
                        node = self.register(obj)
                        nodes.append(node)
                    except Exception:
                        pass
        ## END for name, obj in inspect.getmembers...

        return nodes


    def get(self, node_id: str) -> ComputationNode:
        return self.inner_registry.get(node_id)


    def has(self, node_id: str) -> bool:
        return self.inner_registry.has(node_id)


    def all_ids(self) -> Tuple[str, ...]:
        return self.inner_registry.all_ids()


__all__ = [
    "LoadedSessionContext",
    "SessionComputationBase",
    "session_computation",
    "SessionComputationRegistry",
]
