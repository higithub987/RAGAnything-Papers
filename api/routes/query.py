import asyncio
import logging
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import session_store
from ..container_scope import resolve_scope
from ..lightrag_scope import query_scope
from ..models import (
    CondenseHistoryRequest,
    CondenseHistoryResponse,
    MultimodalQueryRequest,
    QueryRequest,
    QueryResponse,
)
from ..rag_manager import (
    ESCALATION_PARAMS,
    auto_first_thinking,
    condense_history,
    get_rag,
    looks_insufficient,
    resolve_thinking,
    resolve_thinking_budget,
    status_frame,
    synthesis_thinking,
    synthesis_thinking_budget,
)
from raganything.utils import strip_thinking_tags

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)

# Shown (instead of calling the model) when a query is scoped to container(s) that
# contain no documents -- see rag_manager/container_scope. ContainerScope.matches_nothing.
NO_CONTAINER_MATCH = (
    "There are no documents in the selected container(s) yet, so there's nothing "
    "to answer from. Add documents to this container or switch to All."
)

# Strong refs to in-flight background condense tasks so they aren't GC'd mid-run.
_background_tasks: set = set()


def _schedule_condense(session) -> None:
    """Summarize aged-out turns AFTER a turn is recorded, off the query's
    critical path, so a chat's next answer starts streaming without waiting on
    summarization LLM round-trips. Chat use is sequential per session, so the
    at-most-one background condense finishes before the following query in
    practice; if it hasn't, that query just sends a slightly longer history.
    """

    async def _run():
        try:
            await session_store.condense_aged_out_turns(session)
        except Exception:
            logger.exception("Background history condense failed")

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

# Longest opening/closing tag we filter, minus one: how many trailing
# characters of a not-yet-resolved buffer must be held back in case a tag is
# split across two stream chunks (e.g. "...<thi" + "nking>...").
_MAX_TAG_HOLDBACK = len("</thinking>") - 1


async def _strip_thinking_stream(chunks):
    """Filter <think>/<thinking> blocks out of a live text stream.

    Mirrors strip_thinking_tags(), but a tag can arrive split across two
    chunks, so a plain regex per-chunk won't catch it -- this buffers text
    and only emits/discards once a tag boundary (or enough lookahead to rule
    one out) has actually been seen.
    """
    buffer = ""
    in_think = False

    async for chunk in chunks:
        buffer += chunk

        while True:
            if not in_think:
                open_match = re.search(r"<think(?:ing)?>", buffer, re.IGNORECASE)
                if open_match:
                    yield buffer[: open_match.start()]
                    buffer = buffer[open_match.end() :]
                    in_think = True
                    continue
                if len(buffer) > _MAX_TAG_HOLDBACK:
                    yield buffer[:-_MAX_TAG_HOLDBACK]
                    buffer = buffer[-_MAX_TAG_HOLDBACK:]
                break
            else:
                close_match = re.search(r"</think(?:ing)?>", buffer, re.IGNORECASE)
                if close_match:
                    buffer = buffer[close_match.end() :]
                    in_think = False
                    continue
                # Still inside the thinking block: everything so far is
                # discardable except a tail that might be a split close tag.
                buffer = (
                    buffer[-_MAX_TAG_HOLDBACK:]
                    if len(buffer) > _MAX_TAG_HOLDBACK
                    else buffer
                )
                break

    if not in_think and buffer:
        yield buffer


async def _stream_result(result, retry=None, retries=1):
    """Adapt an aquery()/aquery_with_multimodal() result (str or AsyncIterator[str])
    into a chunked text stream, with reasoning (<think>/<thinking> blocks) filtered
    out. A failure mid-generation can't be turned into an HTTP error status anymore
    (the response has already started), so it's surfaced as a trailing error marker
    instead of left to hang the client.

    If `retry` is given (a zero-arg async callable that redoes the original LLM
    call) and the stream fails before any content was emitted -- e.g. the remote
    endpoint drops the connection mid-chunk -- it's safe to silently redo the
    whole call once, since nothing has reached the client yet. Once any content
    has been emitted, retrying would duplicate/garble what's already been shown,
    so failures from then on always fall back to the trailing error marker.
    """
    attempt = 0
    while True:
        emitted = False
        try:
            if isinstance(result, str):
                yield strip_thinking_tags(result)
                return
            async for chunk in _strip_thinking_stream(result):
                emitted = True
                yield chunk
            return
        except Exception as exc:
            if not emitted and retry is not None and attempt < retries:
                attempt += 1
                logger.warning(
                    "Stream failed before any content was sent, retrying: %s", exc
                )
                try:
                    result = await retry()
                    continue
                except Exception:
                    pass
            yield f"\n[error: {exc}]"
            return


def _multimodal_item_to_dict(item) -> dict:
    d = {"type": item.type}
    if item.type == "table":
        d["table_data"] = item.table_data
        d["table_caption"] = item.table_caption
    elif item.type == "equation":
        d["latex"] = item.latex
        d["equation_caption"] = item.equation_caption
    elif item.type == "image":
        d["image_path"] = item.image_path
        d["image_caption"] = item.image_caption
    return d


def _to_multimodal_content(request: MultimodalQueryRequest) -> list[dict]:
    return [_multimodal_item_to_dict(item) for item in request.multimodal_content]


@router.post("", response_model=QueryResponse)
async def text_query(request: QueryRequest):
    session = None
    try:
        kwargs = {}
        if request.session_id:
            session = session_store.get_session(request.session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Chat session not found")
            kwargs["conversation_history"] = session_store.build_conversation_history(
                session
            )
        elif request.conversation_history:
            kwargs["conversation_history"] = request.conversation_history

        scope = resolve_scope(request.container_ids)
        if scope.matches_nothing:
            if session is not None:
                session_store.add_turn(session, request.query, NO_CONTAINER_MATCH)
                _schedule_condense(session)
            return QueryResponse(
                answer=NO_CONTAINER_MATCH, query=request.query, mode=request.mode
            )
        # text_query runs aquery inline, so one set covers every branch below.
        query_scope.set(scope)

        synthesis_thinking_budget.set(resolve_thinking_budget(request.thinking_budget))
        escalated = False
        if request.thinking == "auto":
            first_thinking = auto_first_thinking(request.query)
            synthesis_thinking.set(first_thinking)
            clean_answer = strip_thinking_tags(
                await get_rag().aquery(request.query, mode=request.mode, **kwargs)
            )
            # Escalate only a thinking-off pass that admitted it lacked context.
            if not first_thinking and looks_insufficient(clean_answer):
                synthesis_thinking.set(True)
                clean_answer = strip_thinking_tags(
                    await get_rag().aquery(
                        request.query,
                        mode=request.mode,
                        **ESCALATION_PARAMS,
                        **kwargs,
                    )
                )
                escalated = True
        else:
            synthesis_thinking.set(resolve_thinking(request.thinking, request.query))
            clean_answer = strip_thinking_tags(
                await get_rag().aquery(request.query, mode=request.mode, **kwargs)
            )

        if session is not None:
            session_store.add_turn(session, request.query, clean_answer)
            _schedule_condense(session)
        return QueryResponse(
            answer=clean_answer,
            query=request.query,
            mode=request.mode,
            escalated=escalated,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/condense", response_model=CondenseHistoryResponse)
async def condense_history_endpoint(request: CondenseHistoryRequest):
    try:
        summary = await condense_history(request.summary, request.turns)
        return CondenseHistoryResponse(summary=summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/multimodal", response_model=QueryResponse)
async def multimodal_query(request: MultimodalQueryRequest):
    try:
        content = _to_multimodal_content(request)
        scope = resolve_scope(request.container_ids)
        if scope.matches_nothing:
            return QueryResponse(
                answer=NO_CONTAINER_MATCH, query=request.query, mode=request.mode
            )
        query_scope.set(scope)
        synthesis_thinking.set(resolve_thinking(request.thinking, request.query))
        synthesis_thinking_budget.set(resolve_thinking_budget(request.thinking_budget))
        answer = await get_rag().aquery_with_multimodal(
            request.query,
            multimodal_content=content,
            mode=request.mode,
        )
        return QueryResponse(
            answer=strip_thinking_tags(answer), query=request.query, mode=request.mode
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def _stream_and_record(chunks, session, query: str):
    """Wrap a chunk stream to record the full answer once it's done.

    The route can't know the final answer text until streaming completes, so
    the turn is appended to the session only after the last chunk is yielded.

    The recording runs in a ``finally`` so that a client disconnect mid-stream
    (e.g. the user navigates to another tab, which reloads the page and aborts
    the request) still persists whatever was generated instead of dropping the
    turn entirely. ``add_turn`` runs exactly once, whether the stream completed
    normally or was cancelled.
    """
    parts = []
    try:
        async for chunk in chunks:
            parts.append(chunk)
            yield chunk
    finally:
        if parts:
            session_store.add_turn(session, query, "".join(parts))
            _schedule_condense(session)


_ESCALATION_NOTICE = (
    "The quick answer didn't find enough context — reasoning more deeply and "
    "searching wider. This will take a little longer…"
)


async def _emit(result, parts):
    """Yield an aquery result (a full string or an async chunk iterator) as
    thinking-stripped text, accumulating it into `parts` for later recording."""
    if isinstance(result, str):
        cleaned = strip_thinking_tags(result)
        parts.append(cleaned)
        yield cleaned
    else:
        async for chunk in _strip_thinking_stream(result):
            parts.append(chunk)
            yield chunk


async def _auto_escalating_stream(run, query, record=None, budget=None):
    """Auto-mode stream with an insufficient-context safety net.

    - Clearly analytical queries stream a thinking-on answer directly.
    - Otherwise a fast thinking-off pass is BUFFERED so it can be inspected; if
      it admits it lacked context (looks_insufficient), a one-off delay notice is
      emitted and an escalated pass (thinking-on + wider retrieval) is streamed.

    ``run(stream: bool, **overrides)`` executes one query pass; this generator
    sets the thinking contextvar before each. ``record(text)`` optionally
    persists the final answer (runs in a finally so a client disconnect mid-
    stream still records what was generated). The status frame is never recorded.
    ``budget`` caps reasoning tokens on any thinking-on pass; set once here since
    this generator runs in the StreamingResponse context after the route returns.
    """
    parts = []
    synthesis_thinking_budget.set(budget)
    try:
        try:
            if auto_first_thinking(query):
                synthesis_thinking.set(True)
                async for chunk in _emit(await run(stream=True), parts):
                    yield chunk
                return
            synthesis_thinking.set(False)
            first = await run(stream=False)
            first_clean = strip_thinking_tags(first if isinstance(first, str) else "")
            if not looks_insufficient(first_clean):
                parts.append(first_clean)
                yield first_clean
                return
            yield status_frame(_ESCALATION_NOTICE)
            synthesis_thinking.set(True)
            escalated = await run(stream=True, **ESCALATION_PARAMS)
            async for chunk in _emit(escalated, parts):
                yield chunk
        except Exception as exc:
            logger.exception("Auto-escalation query failed")
            yield f"\n[error: {exc}]"
    finally:
        if record and parts:
            record("".join(parts))


@router.post("/stream")
async def text_query_stream(request: QueryRequest):
    session = None
    try:
        kwargs = {}
        if request.session_id:
            session = session_store.get_session(request.session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Chat session not found")
            kwargs["conversation_history"] = session_store.build_conversation_history(
                session
            )
        elif request.conversation_history:
            kwargs["conversation_history"] = request.conversation_history

        budget = resolve_thinking_budget(request.thinking_budget)

        scope = resolve_scope(request.container_ids)
        if scope.matches_nothing:
            if session is not None:
                session_store.add_turn(session, request.query, NO_CONTAINER_MATCH)
                _schedule_condense(session)

            async def _no_match():
                yield NO_CONTAINER_MATCH

            return StreamingResponse(_no_match(), media_type="text/plain")

        # Auto mode runs a buffered fast pass and may escalate; its own generator
        # sets the thinking contextvar and records the turn, so it bypasses the
        # single-pass _stream_result/_stream_and_record path below.
        if request.thinking == "auto":

            async def run(stream, **overrides):
                # Set inside run so the scope is active on each (deferred) pass.
                query_scope.set(scope)
                return await get_rag().aquery(
                    request.query,
                    mode=request.mode,
                    stream=stream,
                    **overrides,
                    **kwargs,
                )

            def record(text):
                session_store.add_turn(session, request.query, text)
                _schedule_condense(session)

            return StreamingResponse(
                _auto_escalating_stream(
                    run,
                    request.query,
                    record=record if session is not None else None,
                    budget=budget,
                ),
                media_type="text/plain",
            )

        thinking = resolve_thinking(request.thinking, request.query)

        async def make_result():
            # Set inside make_result so the flags are active on both the initial
            # call and the retry path -- the latter runs from _stream_result in
            # the StreamingResponse context, after this route has returned.
            query_scope.set(scope)
            synthesis_thinking.set(thinking)
            synthesis_thinking_budget.set(budget)
            return await get_rag().aquery(
                request.query, mode=request.mode, stream=True, **kwargs
            )

        result = await make_result()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    stream = _stream_result(result, retry=make_result)
    if session is not None:
        stream = _stream_and_record(stream, session, request.query)
    return StreamingResponse(stream, media_type="text/plain")


@router.post("/multimodal/stream")
async def multimodal_query_stream(request: MultimodalQueryRequest):
    try:
        content = _to_multimodal_content(request)
        budget = resolve_thinking_budget(request.thinking_budget)

        scope = resolve_scope(request.container_ids)
        if scope.matches_nothing:

            async def _no_match():
                yield NO_CONTAINER_MATCH

            return StreamingResponse(_no_match(), media_type="text/plain")

        if request.thinking == "auto":

            async def run(stream, **overrides):
                query_scope.set(scope)
                return await get_rag().aquery_with_multimodal(
                    request.query,
                    multimodal_content=content,
                    mode=request.mode,
                    stream=stream,
                    **overrides,
                )

            return StreamingResponse(
                _auto_escalating_stream(run, request.query, budget=budget),
                media_type="text/plain",
            )

        thinking = resolve_thinking(request.thinking, request.query)

        async def make_result():
            query_scope.set(scope)
            synthesis_thinking.set(thinking)
            synthesis_thinking_budget.set(budget)
            return await get_rag().aquery_with_multimodal(
                request.query,
                multimodal_content=content,
                mode=request.mode,
                stream=True,
            )

        result = await make_result()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(
        _stream_result(result, retry=make_result), media_type="text/plain"
    )
