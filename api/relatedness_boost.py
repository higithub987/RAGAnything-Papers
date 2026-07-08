"""Query-time document boosting driven by user-committed relatedness edits.

The Visualize tab lets a user commit a single connection's relatedness (0-1)
with "Apply to search". Each committed pair raises the *importance* of both
endpoint documents; that importance becomes a per-document multiplier applied
at query time through LightRAG's rerank hook (which is otherwise a no-op in
this deployment, since no rerank model is configured).

`rerank()` reorders the retrieved chunk texts by
    (retrieval-order base score) * (per-document boost)
so chunks from boosted documents are more likely to survive the chunk_top_k
and token-budget cutoffs. Nothing here touches the query path itself or the
persisted knowledge graph -- with no committed overrides every boost is 1.0
and the rerank step preserves the original retrieval order exactly.

Overrides are in-memory only: they persist for the server process's lifetime
(so edits survive page navigation) but reset on restart, like chat sessions.
"""

import json
import threading
from pathlib import Path
from typing import Optional

from .config import settings

# Per-document boost = 1 + BOOST_STRENGTH * importance, importance in [0, 1].
# A committed relatedness of 1.0 therefore yields a 1.5x boost.
BOOST_STRENGTH = 0.5

_OVERRIDES_FILENAME = "relatedness_overrides.json"
_CHUNKS_FILENAME = "kv_store_text_chunks.json"


def _pair_key(a: str, b: str) -> str:
    """Order-independent key for a document pair."""
    return f"{a}::{b}" if a <= b else f"{b}::{a}"


class RelatednessBoost:
    """Holds committed relatedness overrides and the derived query-time boost."""

    def __init__(self, working_dir: Optional[str] = None):
        base = Path(working_dir or settings.working_dir)
        self._chunks_path = base / _CHUNKS_FILENAME
        self._lock = threading.Lock()
        # Overrides are intentionally in-memory only: they live for the server
        # process's lifetime (so edits survive page navigation) but reset on
        # restart, like chat sessions. Nothing is written to disk.
        self._overrides: dict[str, float] = {}
        self._doc_boost: dict[str, float] = {}
        # content -> full_doc_id, cached and invalidated by the chunks file mtime.
        self._content_to_doc: dict[str, str] = {}
        self._chunks_mtime: Optional[float] = None
        # Drop any override file left by an earlier persistent version so a stale
        # artifact doesn't imply edits survive a restart.
        try:
            (base / _OVERRIDES_FILENAME).unlink(missing_ok=True)
        except OSError:
            pass

    # --- mutations -------------------------------------------------------
    def set_override(self, doc_id: str, related_doc_id: str, score: float) -> None:
        score = max(0.0, min(1.0, float(score)))
        with self._lock:
            self._overrides[_pair_key(doc_id, related_doc_id)] = score
            self._recompute_boost()

    def clear(self) -> None:
        with self._lock:
            self._overrides = {}
            self._recompute_boost()

    # --- derived state ---------------------------------------------------
    def _recompute_boost(self) -> None:
        """A document's importance is the max committed relatedness of any
        connection it participates in; unedited documents stay neutral (1.0)."""
        importance: dict[str, float] = {}
        for key, score in self._overrides.items():
            a, _, b = key.partition("::")
            for doc_id in (a, b):
                if doc_id and score > importance.get(doc_id, 0.0):
                    importance[doc_id] = score
        self._doc_boost = {
            doc_id: 1.0 + BOOST_STRENGTH * imp for doc_id, imp in importance.items()
        }

    def overrides(self) -> dict[str, float]:
        with self._lock:
            return dict(self._overrides)

    def doc_boost(self) -> dict[str, float]:
        with self._lock:
            return dict(self._doc_boost)

    # --- query-time boost ------------------------------------------------
    def _refresh_content_index(self) -> None:
        """(Re)build content -> full_doc_id from the text-chunks KV store,
        only when the file has changed since the last read."""
        try:
            mtime = self._chunks_path.stat().st_mtime
        except OSError:
            self._content_to_doc = {}
            self._chunks_mtime = None
            return
        if mtime == self._chunks_mtime and self._content_to_doc:
            return
        try:
            raw = json.loads(self._chunks_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._content_to_doc = {
            rec["content"]: rec["full_doc_id"]
            for rec in raw.values()
            if rec.get("content") and rec.get("full_doc_id")
        }
        self._chunks_mtime = mtime

    async def rerank(self, query, documents, top_n=None):
        """LightRAG rerank hook.

        Matches apply_rerank_if_enabled's contract: given the query and the
        list of retrieved chunk texts, return
            [{"index": i, "relevance_score": s}, ...]
        sorted best-first. The base score preserves retrieval order and is
        multiplied by the source document's boost, so favored documents float
        up. With no committed overrides this returns the input order unchanged.
        """
        n = len(documents)
        if n == 0:
            return []

        with self._lock:
            doc_boost = dict(self._doc_boost)
            if doc_boost:
                self._refresh_content_index()
            content_to_doc = self._content_to_doc if doc_boost else {}

        scored = []
        for i, content in enumerate(documents):
            base = (n - i) / n  # first candidate ~1.0, decreasing with rank
            doc_id = content_to_doc.get(content)
            boost = doc_boost.get(doc_id, 1.0) if doc_id else 1.0
            scored.append({"index": i, "relevance_score": base * boost})

        scored.sort(key=lambda r: r["relevance_score"], reverse=True)
        return scored


_boost: Optional[RelatednessBoost] = None


def get_boost() -> RelatednessBoost:
    """Process-wide singleton, consumed by both the API routes and the
    rerank hook wired into LightRAG at construction time."""
    global _boost
    if _boost is None:
        _boost = RelatednessBoost()
    return _boost


async def rerank(query, documents, top_n=None):
    """Module-level rerank hook handed to LightRAG.

    Must be a plain function, not a bound method: LightRAG's __post_init__ runs
    asdict(self), which deep-copies every config field. Deep-copying a bound
    method would recurse into the RelatednessBoost instance and its
    threading.Lock (unpicklable -> 'cannot pickle _thread.lock'); a function is
    copied atomically instead. The singleton (and its lock) is reached lazily
    here, never through the config.
    """
    return await get_boost().rerank(query=query, documents=documents, top_n=top_n)
