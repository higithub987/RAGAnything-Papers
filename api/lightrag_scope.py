"""Container-scoped querying via an app ContextVar + a startup monkeypatch.

Step 4 of the container feature. LightRAG is a pinned pip package under gitignored
`.venv`, so we do NOT edit it. Instead:

  - `query_scope` (a ContextVar, like rag_manager.synthesis_thinking) carries the
    resolved ContainerScope for the current async query, and
  - `apply_scope_patches()` (called once at startup) wraps LightRAG's retrieval
    helpers so they **post-filter** their results to the active scope.

This works because lightrag.operate calls these helpers via module globals resolved
at call time, so replacing the module attribute intercepts every internal caller.
QueryParam / MilvusVectorDBStorage are untouched; scoping reuses
ContainerScope.file_path_in_scope (chunks carry a single file_path; graph
nodes/edges may carry several joined by GRAPH_FIELD_SEP).

Post-filtering (drop out-of-scope results after retrieval) is the same trade
LightRAG's own graph retrieval already makes; correctness is identical, only
recall/efficiency differ.
"""

import functools
import logging
from contextvars import ContextVar
from typing import Optional

from .container_scope import ContainerScope

logger = logging.getLogger(__name__)

# Active scope for the current async query. Set by the query routes, read by the
# wrappers. Async-task-local, so concurrent queries stay isolated. None => the
# virtual "All" (no scoping).
query_scope: ContextVar[Optional[ContainerScope]] = ContextVar(
    "query_scope", default=None
)


def _active_scope() -> Optional[ContainerScope]:
    """The scope to apply, or None when there is nothing to filter (unset / All)."""
    scope = query_scope.get()
    if scope is None or scope.is_all:
        return None
    return scope


# --- pure filter helpers (the unit-tested core) --------------------------------
def scope_chunks(chunks, scope: Optional[ContainerScope]):
    """Drop chunks whose file_path is out of scope. No-op when scope is None/All."""
    if scope is None or scope.is_all or not chunks:
        return chunks
    return [c for c in chunks if scope.file_path_in_scope(c.get("file_path"))]


def scope_nodes(node_datas, scope: Optional[ContainerScope]):
    """Drop graph nodes whose file_path is out of scope. No-op when None/All."""
    if scope is None or scope.is_all or not node_datas:
        return node_datas
    return [n for n in node_datas if scope.file_path_in_scope(n.get("file_path"))]


def scope_edges(edge_datas, scope: Optional[ContainerScope]):
    """Drop graph edges whose file_path is out of scope. No-op when None/All."""
    if scope is None or scope.is_all or not edge_datas:
        return edge_datas
    return [e for e in edge_datas if scope.file_path_in_scope(e.get("file_path"))]


# --- result transforms, one per patched retrieval helper -----------------------
def _transform_vector_context(result):
    """_get_vector_context -> list[chunk]."""
    return scope_chunks(result, _active_scope())


def _transform_node_data(result):
    """_get_node_data -> (node_datas, use_relations)."""
    scope = _active_scope()
    if scope is None:
        return result
    node_datas, use_relations = result
    return scope_nodes(node_datas, scope), scope_edges(use_relations, scope)


def _transform_edge_data(result):
    """_get_edge_data -> (edge_datas, use_entities)."""
    scope = _active_scope()
    if scope is None:
        return result
    edge_datas, use_entities = result
    return scope_edges(edge_datas, scope), scope_nodes(use_entities, scope)


# name in lightrag.operate -> its result transform
_TARGETS = {
    "_get_vector_context": _transform_vector_context,  # 4a: chunk scoping
    "_get_node_data": _transform_node_data,  # 4b: local entities/relations
    "_get_edge_data": _transform_edge_data,  # 4b: global relations/entities
}

_patched = False
_originals: dict = {}


def _make_wrapper(orig, transform):
    @functools.wraps(orig)
    async def wrapper(*args, **kwargs):
        result = await orig(*args, **kwargs)
        try:
            return transform(result)
        except Exception:
            # A filter bug must never crash a query -- fail open (unscoped) + log.
            logger.exception("Container scope filter failed; returning unscoped result")
            return result

    wrapper.__container_scope_wrapped__ = True
    return wrapper


def _patch_module(operate_module) -> list[str]:
    """Wrap the target retrieval helpers on a module in place; return wrapped names.

    Idempotent per function (skips already-wrapped). A missing target is logged and
    skipped (that path stays unscoped) rather than raising. Separated from
    `apply_scope_patches` so it can be tested against a stub module.
    """
    wrapped: list[str] = []
    for name, transform in _TARGETS.items():
        orig = getattr(operate_module, name, None)
        if orig is None or not callable(orig):
            logger.warning(
                "Container scoping: %s not found on %s; that retrieval path will be "
                "UNSCOPED (LightRAG may have changed).",
                name,
                getattr(operate_module, "__name__", operate_module),
            )
            continue
        if getattr(orig, "__container_scope_wrapped__", False):
            wrapped.append(name)
            continue
        _originals[name] = orig
        setattr(operate_module, name, _make_wrapper(orig, transform))
        wrapped.append(name)
    return wrapped


def apply_scope_patches() -> None:
    """Install the scope wrappers on lightrag.operate. Idempotent + version-guarded.

    Called once from rag_manager.initialize_rag(). Missing targets (a LightRAG
    change) are logged and skipped -- that retrieval path is simply unscoped rather
    than crashing.
    """
    global _patched
    if _patched:
        return
    try:
        from lightrag import operate
    except Exception as exc:  # pragma: no cover - lightrag always present in prod
        logger.warning(
            "Container scoping disabled: could not import lightrag.operate (%s)", exc
        )
        return

    wrapped = _patch_module(operate)
    _patched = True
    logger.info("Container scope patches applied to: %s", sorted(wrapped))
