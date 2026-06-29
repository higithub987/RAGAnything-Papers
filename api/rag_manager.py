import asyncio
import json
import logging
import sys
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
from .doc_names_store import load_doc_names, save_doc_name
from .models import TaskStatus
from .task_store import (
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

    Resolves the file path each event carries back to a task_id via
    task_store's path index, then updates that task's stage/progress fields
    so GET /documents/{task_id} reflects live progress instead of only
    flipping between "processing" and "completed"/"failed".
    """

    def _task_id_for(self, file_path: str) -> Optional[str]:
        return get_task_id_by_file_path(file_path)

    def on_parse_start(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            update_progress(
                task_id, stage="parsing", progress=0, message="Parsing started"
            )

    def on_parse_progress(
        self,
        file_path: str,
        message: str = "",
        percent: Optional[float] = None,
        **kwargs,
    ) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            update_progress(task_id, stage="parsing", progress=percent, message=message)

    def on_parse_complete(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            update_progress(
                task_id, stage="parsing", progress=100, message="Parsing complete"
            )

    def on_text_insert_start(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            update_progress(
                task_id, stage="text_insert", progress=0, message="Inserting text"
            )

    def on_text_insert_complete(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            update_progress(
                task_id, stage="text_insert", progress=100, message="Text inserted"
            )

    def on_multimodal_start(
        self, file_path: str, item_count: int = 0, **kwargs
    ) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            update_progress(
                task_id, stage="multimodal", progress=0, message=f"0/{item_count} items"
            )

    def on_multimodal_item_complete(
        self,
        file_path: str,
        item_index: int = 0,
        total_items: int = 0,
        **kwargs,
    ) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            progress = (item_index / total_items * 100) if total_items else None
            update_progress(
                task_id,
                stage="multimodal",
                progress=progress,
                message=f"{item_index}/{total_items} items",
            )

    def on_multimodal_complete(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            update_progress(
                task_id,
                stage="multimodal",
                progress=100,
                message="Multimodal processing complete",
            )

    def on_document_complete(self, file_path: str, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            update_progress(
                task_id, stage="complete", progress=100, message="Document complete"
            )

    def on_document_error(self, file_path: str, error=None, **kwargs) -> None:
        task_id = self._task_id_for(file_path)
        if task_id:
            update_progress(task_id, stage="failed", message=str(error))


_rag: Optional[RAGAnything] = None
_fast_llm_func = None
# Limits how many documents parse at once (settings.max_concurrent_documents,
# default 1). asyncio.Semaphore grants waiters in FIFO order, so queued
# uploads start in upload order regardless of the limit.
_parse_semaphore = asyncio.Semaphore(settings.max_concurrent_documents)


def _complete(model: str, prompt, system_prompt=None, history_messages=None, **kwargs):
    """Call openai_complete_if_cache with this deployment's api_key/base_url baked in."""
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
        return _complete(
            settings.fast_llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    return fast_llm_model_func


def _build_llm_func(fast_llm_func):
    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
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
        return _complete(
            settings.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
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


def initialize_rag() -> RAGAnything:
    global _rag, _fast_llm_func
    _fast_llm_func = _build_fast_llm_func()
    llm_func = _build_llm_func(_fast_llm_func)
    _rag = RAGAnything(
        config=RAGAnythingConfig(
            working_dir=settings.working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        ),
        llm_model_func=llm_func,
        vision_model_func=_build_vision_func(llm_func),
        embedding_func=_build_embedding_func(),
        lightrag_kwargs={"vector_storage": "MilvusVectorDBStorage"},
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

    raw = json.loads(doc_status_path.read_text(encoding="utf-8"))
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
        file_name = doc_names.get(doc_id) or Path(entry.get("file_path", doc_id)).name
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
    """Scale the parse timeout with the uploaded file's size on disk.

    Raw file size is a cheap, dependency-free proxy for parse duration —
    there's no existing page-count/complexity signal anywhere in
    raganything/ or api/, and adding one (e.g. pypdfium2) would mean a new
    core dependency just for this.
    """
    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    scaled = settings.document_processing_timeout_base_seconds + (
        size_mb * settings.document_processing_timeout_per_mb_seconds
    )
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
    raw = json.loads(doc_status_path.read_text(encoding="utf-8"))
    target_name = Path(file_path).name
    for doc_id, entry in raw.items():
        if Path(entry.get("file_path", "")).name == target_name:
            return doc_id
    return None


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
            await asyncio.wait_for(
                get_rag().process_document_complete(
                    file_path=file_path,
                    output_dir=settings.output_dir,
                    parse_method="auto",
                    env={"MINERU_DEVICE_MODE": "cuda"},
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
