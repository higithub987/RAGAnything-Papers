"""Container query-scope resolver (Step 3 of the container feature, DERIVE strategy).

Turns a container selection into the exact filter material the query layer will
consume, without storing anything on Milvus or the graph. Scoping works by filtering
on identifiers the data *already carries*:
  - chunks: `full_doc_id` (== the LightRAG doc_id == the registry's membership key),
  - graph entities/relationships: `file_path` (multi-valued, GRAPH_FIELD_SEP-joined
    on merged entities).

The container -> doc_id mapping comes from the registry (Step 1); the doc_id ->
file_path mapping comes from LightRAG's on-disk doc_status. This module is pure
translation over those two sources -- no Milvus, LightRAG, or LLM imports -- so it is
hermetically unit-testable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .config import settings
from .container_store import get_registry

_log = logging.getLogger(__name__)

# Mirrors lightrag.constants.GRAPH_FIELD_SEP -- the separator LightRAG uses to join
# multiple source file_paths on a merged entity/relationship. Duplicated here (rather
# than imported) so this module stays lightweight and LightRAG-free.
GRAPH_FIELD_SEP = "<SEP>"

_DOC_STATUS_FILENAME = "kv_store_doc_status.json"

# A doc id that can never collide with a real LightRAG doc id (those are
# "doc-<md5hex>"). Used to build a valid-but-never-matching chunk filter for an
# empty (matches-nothing) scope.
_NO_MATCH_DOC_ID = "__container_scope_matches_nothing__"


@dataclass(frozen=True)
class ContainerScope:
    """A resolved container selection -> filter material for the query layer.

    Under the derive strategy nothing is tagged on Milvus/graph; scoping filters on
    identifiers the rows already carry (chunks by full_doc_id, graph by file_path).
    """

    is_all: bool
    doc_ids: frozenset[str]
    file_paths: frozenset[str]
    # Milvus boolean expression for the *chunks* vector search; None when is_all.
    chunk_filter_expr: Optional[str]

    @property
    def matches_nothing(self) -> bool:
        """True when a real (non-All) selection resolves to zero documents.

        Distinct from `is_all`: the query layer should short-circuit to empty results
        rather than fall through to the whole database.
        """
        return not self.is_all and not self.doc_ids

    def file_path_in_scope(self, field: Optional[str]) -> bool:
        """Graph-traversal prune predicate: is this node/edge in scope?

        A graph node/edge `file_path` may hold several source paths joined by
        GRAPH_FIELD_SEP (merged entities); in scope if any part is a member. Always
        true under the virtual "All".
        """
        if self.is_all:
            return True
        if not field:
            return False
        return any(part in self.file_paths for part in field.split(GRAPH_FIELD_SEP))


def _build_chunk_filter_expr(doc_ids: frozenset[str]) -> str:
    """Milvus expr selecting chunks whose full_doc_id is in the scope.

    Mirrors the IN-list quoting in MilvusVectorDBStorage.get_by_ids. Falls back to a
    never-matching sentinel for an empty scope so callers always get a valid, safe
    expression (rather than an empty `in []`).
    """
    ids = sorted(doc_ids) if doc_ids else [_NO_MATCH_DOC_ID]
    id_list = '", "'.join(ids)
    return f'full_doc_id in ["{id_list}"]'


def _load_doc_file_paths(working_dir: str) -> dict[str, str]:
    """{doc_id: file_path} from LightRAG's doc_status store.

    Tolerates a missing/empty/corrupt file (returns {}), matching the defense-in-depth
    in rag_manager.load_existing_documents.
    """
    path = Path(working_dir) / _DOC_STATUS_FILENAME
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text) if text.strip() else {}
    except (OSError, json.JSONDecodeError):
        _log.warning(
            "doc_status unreadable; container scope has no file_paths this call"
        )
        return {}
    result: dict[str, str] = {}
    for doc_id, entry in raw.items():
        file_path = (entry or {}).get("file_path")
        if file_path:
            result[doc_id] = file_path
    return result


def resolve_scope(
    container_ids: Optional[Iterable[str]],
    *,
    registry=None,
    working_dir: Optional[str] = None,
) -> ContainerScope:
    """Translate a container selection into a ContainerScope.

    - None / empty selection -> the virtual "All" (no scoping).
    - Otherwise: union the selected containers' member doc_ids (unknown container ids
      contribute nothing), map them to upload file_paths via doc_status, and build the
      chunks filter expression.

    `registry` and `working_dir` are injectable for hermetic tests (defaults: the
    process registry singleton and settings.working_dir).
    """
    if isinstance(container_ids, str):  # guard a common footgun (a bare id string)
        container_ids = [container_ids]
    selection = [cid for cid in (container_ids or []) if cid]
    if not selection:
        return ContainerScope(
            is_all=True,
            doc_ids=frozenset(),
            file_paths=frozenset(),
            chunk_filter_expr=None,
        )

    registry = registry or get_registry()
    doc_ids: set[str] = set()
    for cid in selection:
        doc_ids.update(registry.documents_in_container(cid))

    doc_file_paths = _load_doc_file_paths(working_dir or settings.working_dir)
    file_paths = {doc_file_paths[d] for d in doc_ids if d in doc_file_paths}

    frozen_docs = frozenset(doc_ids)
    return ContainerScope(
        is_all=False,
        doc_ids=frozen_docs,
        file_paths=frozenset(file_paths),
        chunk_filter_expr=_build_chunk_filter_expr(frozen_docs),
    )
