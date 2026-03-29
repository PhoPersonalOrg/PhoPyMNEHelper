"""Topological execution, optional parallelism, goal subset expansion."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from phopymnehelper.analysis.computations.cache import DiskComputationCache, compute_chained_cache_key
from phopymnehelper.analysis.computations.protocol import ComputationNode, ComputationRegistry, RunContext


def expand_required_nodes(registry: ComputationRegistry, goals: Set[str]) -> Set[str]:
    req: Set[str] = set(goals)
    stack: List[str] = list(goals)
    while stack:
        nid = stack.pop()
        node = registry.get(nid)
        for d in node.deps:
            if d not in registry:
                raise KeyError(f"Unknown dependency {d!r} required by {nid!r}")
            if d not in req:
                req.add(d)
                stack.append(d)
    return req


def assert_acyclic(registry: ComputationRegistry, node_ids: Set[str]) -> None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in node_ids}

    def visit(n: str) -> None:
        if color[n] == GRAY:
            raise ValueError(f"Computation dependency cycle involving {n!r}")
        if color[n] == BLACK:
            return
        color[n] = GRAY
        for d in registry.get(n).deps:
            if d in node_ids:
                visit(d)
        color[n] = BLACK

    for n in node_ids:
        if color[n] == WHITE:
            visit(n)


def topological_sort(registry: ComputationRegistry, node_ids: Set[str]) -> List[str]:
    assert_acyclic(registry, node_ids)
    ids = node_ids
    in_degree: Dict[str, int] = {n: sum(1 for d in registry.get(n).deps if d in ids) for n in ids}
    dependents: Dict[str, List[str]] = defaultdict(list)
    for n in ids:
        for d in registry.get(n).deps:
            if d in ids:
                dependents[d].append(n)
    q = deque([n for n in ids if in_degree[n] == 0])
    out: List[str] = []
    while q:
        n = q.popleft()
        out.append(n)
        for m in dependents[n]:
            in_degree[m] -= 1
            if in_degree[m] == 0:
                q.append(m)
    if len(out) != len(ids):
        raise ValueError("Cycle detected in computation graph")
    return out


def merge_params(global_params: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    return {**global_params, **overrides}


class GraphExecutor:
    def __init__(self, registry: ComputationRegistry, cache: Optional[DiskComputationCache] = None) -> None:
        self.registry = registry
        self.cache = cache


    def run(self, ctx: RunContext, goals: Iterable[str], global_params: Optional[Mapping[str, Any]] = None, params_by_node: Optional[Mapping[str, Mapping[str, Any]]] = None, use_cache: bool = True, parallel: bool = False, max_workers: int = 4) -> Dict[str, Any]:
        gset = set(goals)
        for gid in gset:
            if gid not in self.registry:
                raise KeyError(f"Unknown goal computation id: {gid!r}")
        expanded = expand_required_nodes(self.registry, gset)
        order = topological_sort(self.registry, expanded)
        global_params = dict(global_params or {})
        params_by_node = dict(params_by_node or {})
        dep_outputs: Dict[str, Any] = {}
        dep_keys: Dict[str, str] = {}
        if not parallel:
            for nid in order:
                self._execute_one(ctx, nid, global_params, params_by_node, dep_outputs, dep_keys, use_cache)
        else:
            self._run_parallel(ctx, order, global_params, params_by_node, dep_outputs, dep_keys, use_cache, max_workers)
        return {k: dep_outputs[k] for k in gset}


    def _execute_one(self, ctx: RunContext, nid: str, global_params: Mapping[str, Any], params_by_node: Mapping[str, Mapping[str, Any]], dep_outputs: Dict[str, Any], dep_keys: Dict[str, str], use_cache: bool) -> None:
        node = self.registry.get(nid)
        merged = merge_params(global_params, params_by_node.get(nid, {}))
        dep_key_tuple = tuple(dep_keys[d] for d in sorted(node.deps))
        params_digest = node.effective_params_fingerprint(merged)
        cache_key = compute_chained_cache_key(ctx.session, node, params_digest, dep_key_tuple)
        if use_cache and self.cache is not None and self.cache.contains(cache_key):
            dep_outputs[nid] = self.cache.get(cache_key)
            dep_keys[nid] = cache_key
            return
        inputs = {d: dep_outputs[d] for d in node.deps}
        out = node.run(ctx, merged, inputs)
        dep_outputs[nid] = out
        dep_keys[nid] = cache_key
        if use_cache and self.cache is not None:
            meta = {"node_id": nid, "session_path": ctx.session.canonical_path, "session_mtime": ctx.session.mtime if ctx.session.mtime is not None else "", "params_json": json.dumps(merged, sort_keys=True, default=str), "dep_keys_json": json.dumps(list(dep_key_tuple)), "node_version": node.version}
            self.cache.put(cache_key, out, meta)


    def _run_parallel(self, ctx: RunContext, order: List[str], global_params: Mapping[str, Any], params_by_node: Mapping[str, Mapping[str, Any]], dep_outputs: Dict[str, Any], dep_keys: Dict[str, str], use_cache: bool, max_workers: int) -> None:
        completed: Set[str] = set()
        remaining: Set[str] = set(order)
        while remaining:
            ready = [n for n in remaining if all(d in completed for d in self.registry.get(n).deps)]
            if not ready:
                raise RuntimeError("Deadlock in parallel computation schedule (cycle or missing dependency)")
            with ThreadPoolExecutor(max_workers=min(max_workers, len(ready))) as ex:
                futs = {}
                for nid in ready:
                    dep_out = {k: dep_outputs[k] for k in self.registry.get(nid).deps}
                    dep_k = {k: dep_keys[k] for k in self.registry.get(nid).deps}
                    futs[ex.submit(self._run_node_isolated, ctx, nid, global_params, params_by_node, dep_out, dep_k, use_cache)] = nid
                for fut in as_completed(futs):
                    nid = futs[fut]
                    out, cache_key = fut.result()
                    dep_outputs[nid] = out
                    dep_keys[nid] = cache_key
                    completed.add(nid)
                    remaining.discard(nid)


    def _run_node_isolated(self, ctx: RunContext, nid: str, global_params: Mapping[str, Any], params_by_node: Mapping[str, Mapping[str, Any]], dep_outputs_subset: Mapping[str, Any], dep_keys_subset: Mapping[str, str], use_cache: bool) -> Tuple[Any, str]:
        node = self.registry.get(nid)
        merged = merge_params(global_params, params_by_node.get(nid, {}))
        dep_key_tuple = tuple(dep_keys_subset[d] for d in sorted(node.deps))
        params_digest = node.effective_params_fingerprint(merged)
        cache_key = compute_chained_cache_key(ctx.session, node, params_digest, dep_key_tuple)
        if use_cache and self.cache is not None and self.cache.contains(cache_key):
            return (self.cache.get(cache_key), cache_key)
        inputs = {d: dep_outputs_subset[d] for d in node.deps}
        out = node.run(ctx, merged, inputs)
        if use_cache and self.cache is not None:
            meta = {"node_id": nid, "session_path": ctx.session.canonical_path, "session_mtime": ctx.session.mtime if ctx.session.mtime is not None else "", "params_json": json.dumps(merged, sort_keys=True, default=str), "dep_keys_json": json.dumps(list(dep_key_tuple)), "node_version": node.version}
            self.cache.put(cache_key, out, meta)
        return (out, cache_key)
