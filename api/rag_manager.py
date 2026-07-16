import asyncio
import json
import logging
import os
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from raganything.callbacks import ProcessingCallback
from raganything.utils import strip_thinking_tags

from .config import settings
from .container_store import get_registry
from .lightrag_scope import apply_scope_patches
from .milvus_health import wait_for_milvus_ready
from .doc_names_store import (
    load_doc_names,
    save_doc_name,
    strip_upload_prefix,
    strip_upload_prefixes,
)
from .models import TaskStatus
from .relatedness_boost import get_boost, rerank
from .task_store import (
    advance_progress,
    complete_task,
    fail_task,
    get_task,
    get_task_id_by_file_path,
    import_existing,
    set_doc_id,
    start_processing,
    update_progress,
)


class ProgressTrackingCallback(ProcessingCallback):
    """Mirrors RAGAnything's processing events into the API's task store.

    Resolves the file path each event carries back to a task_id via task_store's
    path index, then advances that task's stage/progress so GET /documents
    reflects live progress.

    Progress is a *single monotonic overall percentage* across the whole
    pipeline (parse -> text_insert -> multimodal -> complete), not a per-stage
    0-100 that resets. Each stage owns a weighted band of the 0-100 range; events
    set the band boundaries and the (rare) real intra-stage signals interpolate
    within them. The frontend fills the silent gaps between these milestones with
    a time-based creep, so the bar keeps advancing during the long, event-less
    text_insert (LLM extraction) phase. Bands (see the plan):
        parsing      3 -> 20
        text_insert 22 -> 65   (dominant phase; no intra events)
        multimodal  67 -> 97   (has real item-level progress)
        complete           100
    """

    # Stage band boundaries as overall-progress percentages.
    _PARSE_START = 3
    _PARSE_END = 20
    _TEXT_END = 65
    _MM_START = 67
    _MM_END = 97

    def _task_id_for(self, file_path: str) -> Optional[str]:
        return get_task_id_by_file_path(file_path)

    def on_parse_start(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            advance_progress(
                task_id,
                stage="parsing",
                progress=self._PARSE_START,
                message="Parsing started",
            )

    def on_parse_progress(
        self,
        file_path: str,
        message: str = "",
        percent: Optional[float] = None,
        **kwargs,
    ) -> None:
        task_id = self._task_id_for(file_path)
        if not task_id:
            return
        # Map a real parse percentage (only some parsers emit one) into the parse
        # band; when percent is None we still surface the live message but leave
        # the numeric bar to the frontend creep.
        overall = None
        if percent is not None:
            overall = self._PARSE_START + (percent / 100.0) * (
                self._PARSE_END - self._PARSE_START
            )
        advance_progress(
            task_id, stage="parsing", progress=overall, message=message
        )

    def on_parse_complete(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            advance_progress(
                task_id,
                stage="parsing",
                progress=self._PARSE_END,
                message="Parsing complete",
            )

    def on_text_insert_start(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            advance_progress(
                task_id,
                stage="text_insert",
                progress=self._PARSE_END + 2,
                message="Extracting entities & relationships",
            )

    def on_text_insert_complete(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            advance_progress(
                task_id,
                stage="text_insert",
                progress=self._TEXT_END,
                message="Text indexed",
            )

    def on_multimodal_start(
        self, file_path: str, item_count: int = 0, **kwargs
    ) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            advance_progress(
                task_id,
                stage="multimodal",
                progress=self._MM_START,
                message=f"0/{item_count} items",
            )

    def on_multimodal_item_complete(
        self,
        file_path: str,
        item_index: int = 0,
        total_items: int = 0,
        **kwargs,
    ) -> None:
        task_id = self._task_id_for(file_path)
        if not task_id:
            return
        overall = None
        if total_items:
            overall = self._MM_START + (item_index / total_items) * (
                self._MM_END - self._MM_START
            )
        advance_progress(
            task_id,
            stage="multimodal",
            progress=overall,
            message=f"{item_index}/{total_items} items",
        )

    def on_multimodal_complete(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            advance_progress(
                task_id,
                stage="multimodal",
                progress=self._MM_END,
                message="Multimodal processing complete",
            )

    def on_document_complete(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            advance_progress(
                task_id, stage="complete", progress=100, message="Document complete"
            )

    def on_document_error(self, file_path: str, error=None, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            advance_progress(task_id, stage="failed", message=str(error))


_rag: Optional[RAGAnything] = None
_fast_llm_func = None
# Limits how many documents parse at once (settings.max_concurrent_documents,
# default 1). asyncio.Semaphore grants waiters in FIFO order, so queued
# uploads start in upload order regardless of the limit.
_parse_semaphore = asyncio.Semaphore(settings.max_concurrent_documents)

# Per-request switch for the synthesis model's "thinking" (chain-of-thought)
# tokens, read inside the synthesis branch of _build_llm_func. A ContextVar is
# async-task-local, so concurrent queries each keep their own value. Default
# True preserves the model's native behavior for any caller that doesn't set it
# (the query routes always do). The fast/mechanical model is unaffected -- it
# forces thinking off unconditionally (see _build_fast_llm_func).
synthesis_thinking: ContextVar[bool] = ContextVar("synthesis_thinking", default=True)

# Per-request cap on the synthesis model's reasoning tokens (qwen "thinking_budget").
# None = uncapped. Only applied when thinking is on (see the synthesis branch);
# set by the query routes in the same context as synthesis_thinking.
synthesis_thinking_budget: ContextVar[Optional[int]] = ContextVar(
    "synthesis_thinking_budget", default=None
)


def resolve_thinking_budget(value) -> Optional[int]:
    """Normalize a request's thinking_budget to a positive int or None (uncapped).

    Non-numeric / missing / <= 0 -> None; anything larger is clamped to a sane
    ceiling so a stray huge value can't blow up generation.
    """
    try:
        budget = int(value)
    except (TypeError, ValueError):
        return None
    if budget <= 0:
        return None
    return min(budget, 32768)

# Cheap intent cues that a query needs multi-step reasoning (analytical/
# multi-hop) rather than a direct extractive lookup. Used by "auto" mode.
_ANALYTICAL_CUES = re.compile(
    r"\b(why|how|compare|comparison|versus|vs|differ|difference|tradeoff|"
    r"trade-off|explain|analyz|evaluat|summariz|relationship|relate|cause|"
    r"impact|implication|recommend|best|worst|rank|across|both|overall|"
    r"synthesiz)\b",
    re.IGNORECASE,
)


def _auto_thinking(query: str) -> bool:
    """Heuristic router for "auto" mode: enable thinking only when the query
    looks analytical/multi-hop, else keep it off for speed.

    Intent-only signal (question verbs + length). Retrieval-spread/confidence
    signals would be stronger but live deep inside aquery and aren't cheaply
    reachable from the route; escalating on those is a future improvement.
    """
    if _ANALYTICAL_CUES.search(query or ""):
        return True
    return len((query or "").split()) > 25


def resolve_thinking(setting: Optional[str], query: str) -> bool:
    """Map the request's thinking setting ("on"|"off"|"auto") to a bool.

    Unknown/missing values fall back to "off" (the UI default, speed-first).
    Note: "auto" is handled by the escalation flow in routes/query.py; this
    heuristic result (via auto_first_thinking) is only its *first-pass* choice.
    """
    value = (setting or "off").lower()
    if value == "on":
        return True
    if value == "auto":
        return _auto_thinking(query)
    return False


# Public alias so the auto-escalation flow (routes/query.py) can pick the
# first-pass thinking state without reaching into a private name.
auto_first_thinking = _auto_thinking


# Retrieval-budget bumps applied on an auto-mode escalation pass. "Insufficient
# context" is usually a retrieval gap, so widen the candidate pools beyond the
# LightRAG defaults (top_k=40, chunk_top_k=20); max_total_tokens (30000) leaves
# room for the extra chunks.
ESCALATION_PARAMS = {"top_k": 64, "chunk_top_k": 32}

# Control-frame delimiter for out-of-band status notices on the plain-text query
# stream. \x1e (ASCII Record Separator) never appears in LLM/answer text, so the
# client can split it out unambiguously. Frame: SENTINEL + json + SENTINEL.
STATUS_SENTINEL = "\x1e"


def status_frame(message: str) -> str:
    """Encode a one-off status notice for the query stream (see STATUS_SENTINEL)."""
    return (
        STATUS_SENTINEL
        + json.dumps({"type": "status", "message": message})
        + STATUS_SENTINEL
    )


# Signals that a fast (thinking-off) answer failed to ground. LightRAG returns a
# fail_response ending in [no-context] when retrieval is empty, and its
# rag_response prompt tells the model to say it lacks enough information when the
# context is thin. Kept tight to avoid false positives -- a false positive wastes
# a slow escalation pass on an already-good answer.
_INSUFFICIENT_PATTERNS = re.compile(
    r"\[no-context\]"
    r"|not\s+(?:have\s+)?enough\s+(?:information|context)"
    r"|do(?:es)?\s+not\s+have\s+enough"
    r"|do(?:es)?\s+not\s+(?:contain|mention|provide|include)\b"
    r"|cannot\s+be\s+found\s+in\s+the"
    r"|no\s+(?:relevant\s+)?information\s+(?:is\s+)?(?:available|found|provided)"
    r"|unable\s+to\s+(?:answer|provide)"
    r"|insufficient\s+(?:information|context)",
    re.IGNORECASE,
)


def looks_insufficient(answer: str) -> bool:
    """True if a fast-pass answer admits it lacked enough grounded context."""
    return bool(answer) and bool(_INSUFFICIENT_PATTERNS.search(answer))


def _scrub_messages(messages: list) -> list:
    """Strip the upload "<uuid>_" prefix from the text of OpenAI-style messages.

    Handles both plain-string content and multimodal content lists (only the
    {"type": "text", ...} parts -- image parts are left untouched). Returns new
    dicts so the caller's message structures aren't mutated in place.
    """
    scrubbed = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            msg = {**msg, "content": strip_upload_prefixes(content)}
        elif isinstance(content, list):
            msg = {
                **msg,
                "content": [
                    {**p, "text": strip_upload_prefixes(p["text"])}
                    if isinstance(p, dict)
                    and p.get("type") == "text"
                    and isinstance(p.get("text"), str)
                    else p
                    for p in content
                ],
            }
        scrubbed.append(msg)
    return scrubbed


def _complete(model: str, prompt, system_prompt=None, history_messages=None, **kwargs):
    """Call openai_complete_if_cache with this deployment's api_key/base_url baked in.

    Scrubs the upload "<uuid>_" prefix out of everything the model sees (prompt,
    system prompt, and any multimodal `messages`). This is the single chokepoint
    every generation path funnels through -- text (llm_model_func), fast, and
    vision (both the messages and image_data branches) -- so citations come back
    with clean document names no matter which path produced the answer.
    """
    if isinstance(prompt, str):
        prompt = strip_upload_prefixes(prompt)
    if isinstance(system_prompt, str):
        system_prompt = strip_upload_prefixes(system_prompt)
    if isinstance(kwargs.get("messages"), list):
        kwargs["messages"] = _scrub_messages(kwargs["messages"])
    return openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=settings.llm_binding_api_key,
        base_url=settings.llm_binding_host,
        **kwargs,
    )


def _build_fast_llm_func():
    def fast_llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        # Disable qwen's "thinking" reasoning tokens on the fast model. This
        # model only handles mechanical steps -- keyword extraction (routed here
        # from _build_llm_func), modal table/equation/generic captioning, and
        # chat-history condensing -- none of which benefit from chain-of-thought.
        # Left on, qwen3.6-flash emits hundreds of hidden reasoning tokens per
        # call (measured ~750 on a keyword-extraction prompt) purely as latency.
        # The full model's synthesis call is untouched, so answer-quality
        # reasoning is preserved. DashScope reads this from the request body, so
        # it goes via extra_body; merge into any caller-supplied extra_body
        # rather than clobber it.
        extra_body = {**kwargs.pop("extra_body", {}), "enable_thinking": False}
        return _complete(
            settings.fast_llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            extra_body=extra_body,
            **kwargs,
        )

    return fast_llm_model_func


def _build_llm_func(fast_llm_func):
    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        # Note: the upload "<uuid>_" prefix is scrubbed centrally in _complete
        # (covers this path plus the vision/fast paths), so no stripping here.
        # LightRAG marks its keyword-extraction call with keyword_extraction=True
        # (see lightrag/operate.py's extract_keywords_only) -- route that cheap,
        # mechanical task to the fast model and keep the full model for everything
        # else (in particular, the final answer synthesis call).
        if kwargs.pop("keyword_extraction", False):
            return fast_llm_func(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
        # Synthesis (final answer) call. Honor the per-request thinking switch
        # (set by the query routes; defaults True). qwen models read this from
        # the request body via extra_body; merge rather than clobber any
        # caller-supplied extra_body.
        extra_body = {
            **kwargs.pop("extra_body", {}),
            "enable_thinking": synthesis_thinking.get(),
        }
        # Cap reasoning length only when thinking is actually on; meaningless
        # (and rejected by some models) otherwise.
        budget = synthesis_thinking_budget.get()
        if extra_body["enable_thinking"] and budget:
            extra_body["thinking_budget"] = budget
        return _complete(
            settings.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            extra_body=extra_body,
            **kwargs,
        )

    return llm_model_func


def _build_vision_func(llm_model_func):
    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages:
            return _complete(
                settings.vision_model,
                "",
                messages=messages,
                **kwargs,
            )
        if image_data:
            messages = []
            if system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }
            )
            return _complete(
                settings.vision_model,
                "",
                messages=messages,
                **kwargs,
            )
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    return vision_model_func


def _build_embedding_func():
    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=settings.embedding_model,
            api_key=settings.llm_binding_api_key,
            base_url=settings.llm_binding_host,
        ),
    )


def _export_mineru_api_env() -> None:
    """Mirror the MinerU cloud-API settings into os.environ.

    MineruApiParser reads its token / base-url / model-version / TLS flag from the
    process environment (so the same knobs work for the library, CLI, and API).
    pydantic's Settings only reads .env into the settings object, not os.environ,
    so bridge them here. setdefault keeps a real environment variable authoritative
    over the .env-derived value.
    """
    if settings.mineru_api_token:
        os.environ.setdefault("MINERU_API_TOKEN", settings.mineru_api_token)
    os.environ.setdefault("MINERU_API_BASE_URL", settings.mineru_api_base_url)
    os.environ.setdefault("MINERU_API_MODEL_VERSION", settings.mineru_api_model_version)
    if settings.mineru_api_insecure:
        os.environ.setdefault("MINERU_API_INSECURE", "true")


def initialize_rag() -> RAGAnything:
    global _rag, _fast_llm_func
    # Install the container query-scoping wrappers on lightrag.operate before any
    # query runs (idempotent; a no-op unless a query sets the query_scope contextvar).
    apply_scope_patches()
    # Must run before RAGAnything is built so the parser's env-based config resolves.
    if settings.parser in ("mineru-api", "mineru-cloud"):
        _export_mineru_api_env()
    _fast_llm_func = _build_fast_llm_func()
    llm_func = _build_llm_func(_fast_llm_func)
    # LightRAG already runs a rerank step for every graph/vector query when
    # enable_rerank is true (the default), but does nothing without a rerank
    # func. We supply one that reorders retrieved chunks by user-committed
    # document relatedness (see relatedness_boost). min_rerank_score=0.0 keeps
    # LightRAG's score-threshold filter off so our synthetic scores never drop
    # chunks -- with no committed overrides the rerank is a pure no-op.
    #
    # rerank_model_func must be the module-level `rerank` function, NOT a bound
    # method: LightRAG's __post_init__ runs asdict(self), which deep-copies
    # every config field, and a bound method would drag in the boost singleton's
    # threading.Lock (unpicklable). get_boost() here just warms the override
    # store at startup.
    get_boost()
    _rag = RAGAnything(
        config=RAGAnythingConfig(
            working_dir=settings.working_dir,
            parser=settings.parser,
            parse_method=settings.parse_method,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        ),
        llm_model_func=llm_func,
        vision_model_func=_build_vision_func(llm_func),
        embedding_func=_build_embedding_func(),
        lightrag_kwargs={
            "vector_storage": "MilvusVectorDBStorage",
            "rerank_model_func": rerank,
            "min_rerank_score": 0.0,
        },
    )
    _rag.callback_manager.register(ProgressTrackingCallback())
    return _rag


def get_rag() -> RAGAnything:
    if _rag is None:
        raise RuntimeError(
            "RAGAnything not initialized — call initialize_rag() at startup"
        )
    return _rag


async def condense_history(summary: str, turns: list[dict]) -> str:
    """Fold one or more {query, answer} turns into a running chat summary.

    Uses the fast/cheap model (the same one used for keyword extraction) so
    that condensing aging-out turns stays cheap regardless of how long a
    chat conversation gets -- callers only ever pass in the existing
    (already-condensed) summary plus the handful of turns that just aged
    out of the raw history window, never the whole conversation.
    """
    turns_text = "\n\n".join(
        f"User: {turn['query']}\nAssistant: {turn['answer']}" for turn in turns
    )
    prompt = (
        "Update the running summary of this conversation to also cover the "
        "new turns below. Keep the result under 120 words, preserve key "
        "facts/decisions, and drop small talk.\n\n"
        f"Existing summary:\n{summary or '(none yet)'}\n\n"
        f"New turns:\n{turns_text}\n\n"
        "Updated summary:"
    )
    result = await _fast_llm_func(prompt)
    return strip_thinking_tags(result)


# Maps every known LightRAG DocStatus string + legacy RAGAnything values to TaskStatus.
_STATUS_MAP: dict[str, TaskStatus] = {
    "pending": TaskStatus.PROCESSING,
    "processing": TaskStatus.PROCESSING,
    "preprocessed": TaskStatus.PROCESSING,
    "handling": TaskStatus.PROCESSING,  # legacy RAGAnything batch status
    "processed": TaskStatus.COMPLETED,
    "failed": TaskStatus.FAILED,
}


def load_existing_documents() -> None:
    """Read LightRAG's doc_status store and pre-populate the task store."""
    doc_status_path = Path(settings.working_dir) / "kv_store_doc_status.json"
    if not doc_status_path.exists():
        return

    # Defense in depth: writers should be atomic (see lightrag write_json), but a
    # bad/empty/partial read must never 500 the /documents endpoint. Skip this
    # refresh on a transient bad read -- the in-memory task store keeps its
    # existing entries and the next poll (~1s) repopulates.
    try:
        text = doc_status_path.read_text(encoding="utf-8")
        raw = json.loads(text) if text.strip() else {}
    except (OSError, json.JSONDecodeError):
        _log.warning("doc_status unreadable this cycle; skipping refresh")
        return

    doc_names = load_doc_names()
    for doc_id, entry in raw.items():
        raw_status = entry.get("status", "")
        if raw_status not in _STATUS_MAP:
            _log.warning(
                "Unknown doc status %r for %s — defaulting to FAILED",
                raw_status,
                doc_id,
            )
        status = _STATUS_MAP.get(raw_status, TaskStatus.FAILED)
        created_at = (
            datetime.fromisoformat(entry["created_at"]).replace(tzinfo=timezone.utc)
            if entry.get("created_at")
            else datetime.now(timezone.utc)
        )
        completed_at = (
            datetime.fromisoformat(entry["updated_at"]).replace(tzinfo=timezone.utc)
            if status == TaskStatus.COMPLETED and entry.get("updated_at")
            else None
        )
        error = (
            "Processing was interrupted before completion. Re-upload to reprocess."
            if raw_status == "handling"
            else entry.get("error_msg") or None
        )
        file_name = strip_upload_prefix(
            doc_names.get(doc_id) or Path(entry.get("file_path", doc_id)).name
        )
        import_existing(
            task_id=doc_id,
            file_name=file_name,
            status=status,
            created_at=created_at,
            completed_at=completed_at,
            error=error,
        )


async def ensure_rag_ready() -> None:
    """Force LightRAG initialization at startup so queries work without uploading first."""
    # Gate on Milvus being query-ready before storage init touches any
    # collection. The MilvusVectorDBStorage init ends in a synchronous, no-timeout
    # load_collection() (lightrag/kg/milvus_impl.py); if Milvus's QueryNodes
    # aren't ready yet that call blocks the event loop indefinitely. Waiting here
    # turns an unbounded hang into a bounded, loudly-failing wait.
    await wait_for_milvus_ready(
        settings.milvus_uri,
        timeout_s=settings.milvus_ready_timeout_seconds,
    )
    rag = get_rag()
    result = await rag._ensure_lightrag_initialized()
    if isinstance(result, dict) and not result.get("success", True):
        raise RuntimeError(f"LightRAG initialization failed: {result.get('error')}")

    # Modal captioning (table/equation/generic) is a cheap, mechanical task --
    # route it to the fast model. Image captioning stays on the vision model
    # since cheap tiers typically don't support vision input.
    for content_type in ("table", "equation", "generic"):
        if content_type in rag.modal_processors:
            rag.modal_processors[content_type].modal_caption_func = _fast_llm_func


def _compute_timeout_seconds(file_path: str) -> int:
    """Scale the parse timeout with the uploaded file's estimated page count.

    File size / assumed KB-per-page stands in for a real page count (see
    config.py comment for why); the result is multiplied by mineru's
    measured per-page cold-start cost.
    """
    size_kb = Path(file_path).stat().st_size / 1024
    estimated_pages = max(1, size_kb / settings.document_processing_kb_per_page)
    scaled = estimated_pages * settings.document_processing_seconds_per_page
    return min(int(scaled), settings.document_processing_timeout_max_seconds)


def _resolve_doc_id(file_path: str) -> Optional[str]:
    """Find the LightRAG doc_id assigned to a just-processed file.

    process_document_complete doesn't return the doc_id it computed, so the
    only way to recover it afterwards is to match the upload's destination
    filename (unique per upload, since it's prefixed with the task UUID)
    against LightRAG's persistent doc_status store.
    """
    doc_status_path = Path(settings.working_dir) / "kv_store_doc_status.json"
    if not doc_status_path.exists():
        return None
    # Tolerate a transient bad/empty read (see load_existing_documents); a missing
    # doc_id here is already a handled outcome for the caller.
    try:
        text = doc_status_path.read_text(encoding="utf-8")
        raw = json.loads(text) if text.strip() else {}
    except (OSError, json.JSONDecodeError):
        return None
    target_name = Path(file_path).name
    for doc_id, entry in raw.items():
        if Path(entry.get("file_path", "")).name == target_name:
            return doc_id
    return None


def record_task_containers(task, doc_id: str) -> None:
    """Apply a task's pending container choice to the registry once its doc_id exists.

    Upload records the chosen container on the task (DocumentTask.container_ids)
    before the LightRAG doc_id is known; registry membership is keyed by doc_id, so
    it can only be written here, when the doc_id resolves. A no-op when the upload
    chose "All" (empty container_ids). Kept as a small standalone helper so the
    membership wiring is unit-testable without the parse pipeline.
    """
    if task is None:
        return
    registry = get_registry()
    for container_id in task.container_ids:
        registry.add_document(container_id, doc_id)


async def process_document_task(task_id: str, file_path: str) -> None:
    timeout_seconds = _compute_timeout_seconds(file_path)
    update_progress(
        task_id,
        stage="queued",
        message="Waiting for another document to finish processing",
    )
    async with _parse_semaphore:
        start_processing(task_id)
        update_progress(task_id, stage="parsing", progress=0, message="Parsing started")
        try:
            # MINERU_DEVICE_MODE only applies to the local mineru subprocess; the
            # cloud API parser runs inference remotely and doesn't use it.
            parse_kwargs = {}
            if settings.parser == "mineru":
                parse_kwargs["env"] = {"MINERU_DEVICE_MODE": "cuda"}
            await asyncio.wait_for(
                get_rag().process_document_complete(
                    file_path=file_path,
                    output_dir=settings.output_dir,
                    parse_method=settings.parse_method,
                    **parse_kwargs,
                ),
                timeout=timeout_seconds,
            )
            complete_task(task_id)
        except asyncio.TimeoutError:
            # The underlying parser call runs in a worker thread (see
            # RAGAnythingPDFProcessor.parse_document's use of asyncio.to_thread),
            # which can't be force-killed from here — it may keep running in the
            # background until it exits on its own. This at least stops a hung
            # parser from leaving the task (and the UI) stuck on "processing"
            # forever, and keeps the server responsive to other requests.
            fail_task(
                task_id,
                f"Processing timed out after {timeout_seconds}s",
            )
        except Exception as exc:
            fail_task(task_id, str(exc))
        finally:
            # On a clean failure, process_document_complete now rolls back
            # doc_id's storage (chunks/entities/vectors/graph + its own
            # doc_status row) before its exception reaches the `except`
            # above, so _resolve_doc_id correctly finds nothing here and
            # set_doc_id is skipped -- this is expected, not a bug.
            doc_id = _resolve_doc_id(file_path)
            if doc_id is not None:
                set_doc_id(task_id, doc_id)
                task = get_task(task_id)
                if task is not None:
                    save_doc_name(doc_id, task.file_name)
                    record_task_containers(task, doc_id)
