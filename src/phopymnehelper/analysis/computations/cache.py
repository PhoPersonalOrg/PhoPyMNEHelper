"""Per-node content-addressed cache (disk-backed pickle) and chained cache keys."""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple

import pandas as pd

from phopymnehelper.analysis.computations.protocol import ComputationNode, SessionFingerprint, PROTOCOL_VERSION, canonical_json_hash_payload


def sha256_hex_digest(payload: str, n: int = 16) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]


def compute_chained_cache_key(session: SessionFingerprint, node: ComputationNode, params_digest: str, dep_cache_keys: Tuple[str, ...]) -> str:
    dep_part = canonical_json_hash_payload(tuple(sorted(dep_cache_keys)))
    payload = canonical_json_hash_payload((PROTOCOL_VERSION, session.payload_for_hash(), node.id, node.version, params_digest, dep_part))
    return sha256_hex_digest(payload, 16)


NODE_MANIFEST_COLUMNS = ["cache_key_hex", "node_id", "session_path", "session_mtime", "params_json", "dep_keys_json", "protocol_version", "node_version", "computed_at"]


class ComputationCacheBackend(Protocol):
    def get(self, cache_key_hex: str) -> Optional[Any]: ...
    def put(self, cache_key_hex: str, value: Any, meta: Mapping[str, Any]) -> None: ...


@dataclass
class DiskComputationCache:
    """Pickle blobs under cache_root; optional manifest CSV for debugging/provenance."""

    cache_root: Path
    manifest_name: str = "node_manifest.csv"
    _lock: threading.Lock = threading.Lock()
    _manifest_df: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        self.cache_root = Path(self.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.cache_root / self.manifest_name

    def _load_manifest(self) -> pd.DataFrame:
        if self._manifest_df is not None:
            return self._manifest_df
        if not self.manifest_path.exists():
            self._manifest_df = pd.DataFrame(columns=list(NODE_MANIFEST_COLUMNS))
            return self._manifest_df
        try:
            df = pd.read_csv(self.manifest_path)
            if list(df.columns) != list(NODE_MANIFEST_COLUMNS):
                self._manifest_df = pd.DataFrame(columns=list(NODE_MANIFEST_COLUMNS))
            else:
                self._manifest_df = df
        except Exception:
            self._manifest_df = pd.DataFrame(columns=list(NODE_MANIFEST_COLUMNS))
        return self._manifest_df

    def blob_path(self, cache_key_hex: str) -> Path:
        return self.cache_root / f"{cache_key_hex}.pkl"

    def contains(self, cache_key_hex: str) -> bool:
        p = self.blob_path(cache_key_hex)
        return p.is_file()

    def get(self, cache_key_hex: str) -> Optional[Any]:
        p = self.blob_path(cache_key_hex)
        if not p.is_file():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def put(self, cache_key_hex: str, value: Any, meta: Mapping[str, Any]) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        p = self.blob_path(cache_key_hex)
        with open(p, "wb") as f:
            pickle.dump(value, f)
        row = {
            "cache_key_hex": cache_key_hex,
            "node_id": meta.get("node_id", ""),
            "session_path": meta.get("session_path", ""),
            "session_mtime": meta.get("session_mtime", ""),
            "params_json": meta.get("params_json", ""),
            "dep_keys_json": meta.get("dep_keys_json", ""),
            "protocol_version": meta.get("protocol_version", PROTOCOL_VERSION),
            "node_version": meta.get("node_version", ""),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        new_row = pd.DataFrame([row])
        with self._lock:
            df = self._load_manifest()
            if self.manifest_path.exists() and len(df.columns) > 0 and len(df) > 0:
                new_row.to_csv(self.manifest_path, mode="a", header=False, index=False)
            else:
                new_row.to_csv(self.manifest_path, mode="w", header=True, index=False)
            self._manifest_df = None

    def resolve_key_for_node(self, session: SessionFingerprint, node: ComputationNode, params: Mapping[str, Any], dep_keys: Tuple[str, ...]) -> str:
        params_digest = node.effective_params_fingerprint(params)
        return compute_chained_cache_key(session, node, params_digest, dep_keys)
