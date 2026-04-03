"""Computation protocol: node metadata, session fingerprint, run context, registry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple


PROTOCOL_VERSION = "1"


class ArtifactKind(str, Enum):
    stream = "stream"
    summary = "summary"
    object = "object"
    renderer = "renderer"


@dataclass(frozen=True)
class SessionFingerprint:
    """Cheap stable descriptor for session-scoped data (e.g. XDF/FIF identity)."""

    canonical_path: str
    mtime: Optional[float] = None
    extra: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def from_path(cls, path: Path, mtime: Optional[float] = None, extra: Optional[Mapping[str, str]] = None) -> "SessionFingerprint":
        canon = str(Path(path).resolve())
        ex: Tuple[Tuple[str, str], ...] = tuple(sorted((k, str(v)) for k, v in (extra or {}).items()))
        return cls(canonical_path=canon, mtime=mtime, extra=ex)

    def payload_for_hash(self) -> str:
        extra_json = json.dumps(list(self.extra))
        if self.mtime is not None:
            return f"{self.canonical_path}|{self.mtime}|{extra_json}"
        return f"{self.canonical_path}|{extra_json}"


@dataclass(frozen=True)
class ArtifactRef:
    """Light handle for cached or in-memory computation output (timeline / exporters)."""

    node_id: str
    cache_key_hex: str
    kind: ArtifactKind
    serializer: str = "pickle"


@dataclass
class RunContext:
    """Inputs available to every computation node."""

    session: SessionFingerprint
    raw: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)


RunFn = Callable[[RunContext, Mapping[str, Any], Mapping[str, Any]], Any]


@dataclass
class ComputationNode:
    """Declarative computation: explicit deps, versioned cache keys, single run entrypoint."""

    id: str
    version: str
    deps: Tuple[str, ...]
    kind: ArtifactKind
    run: RunFn
    params_fingerprint: Optional[Callable[[Mapping[str, Any]], str]] = None

    def effective_params_fingerprint(self, params: Mapping[str, Any]) -> str:
        if self.params_fingerprint is not None:
            return self.params_fingerprint(params)
        return json.dumps(dict(params), sort_keys=True, default=str)


class ComputationRegistry:
    def __init__(self) -> None:
        self._nodes: MutableMapping[str, ComputationNode] = {}

    def register(self, node: ComputationNode) -> ComputationNode:
        if node.id in self._nodes:
            raise ValueError(f"Duplicate computation id: {node.id}")
        self._nodes[node.id] = node
        return node

    def get(self, node_id: str) -> ComputationNode:
        return self._nodes[node_id]

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def all_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._nodes.keys()))

    def nodes(self) -> Tuple[ComputationNode, ...]:
        return tuple(self._nodes[k] for k in sorted(self._nodes.keys()))


DEFAULT_REGISTRY = ComputationRegistry()


def register_default(node: ComputationNode) -> ComputationNode:
    return DEFAULT_REGISTRY.register(node)


def canonical_json_hash_payload(parts: Tuple[str, ...]) -> str:
    return "|".join(parts)
