import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import session_store
from ..models import (
    CondenseHistoryRequest,
    CondenseHistoryResponse,
    MultimodalQueryRequest,
    QueryRequest,
    QueryResponse,
)
from ..rag_manager import condense_history, get_rag
from raganything.utils import strip_thinking_tags

router = APIRouter(prefix="/query", tags=["query"])

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


async def _stream_result(result):
    """Adapt an aquery()/aquery_with_multimodal() result (str or AsyncIterator[str])
    into a chunked text stream, with reasoning (<think>/<thinking> blocks) filtered
    out. A failure mid-generation can't be turned into an HTTP error status anymore
    (the response has already started), so it's surfaced as a trailing error marker
    instead of left to hang the client.
    """
    try:
        if isinstance(result, str):
            yield strip_thinking_tags(result)
        else:
            async for chunk in _strip_thinking_stream(result):
                yield chunk
    except Exception as exc:
        yield f"\n[error: {exc}]"


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
            await session_store.condense_aged_out_turns(session)
            kwargs["conversation_history"] = session_store.build_conversation_history(
                session
            )
        elif request.conversation_history:
            kwargs["conversation_history"] = request.conversation_history
        answer = await get_rag().aquery(request.query, mode=request.mode, **kwargs)
        clean_answer = strip_thinking_tags(answer)
        if session is not None:
            session_store.add_turn(session, request.query, clean_answer)
        return QueryResponse(
            answer=clean_answer, query=request.query, mode=request.mode
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
    """
    parts = []
    async for chunk in chunks:
        parts.append(chunk)
        yield chunk
    session_store.add_turn(session, query, "".join(parts))


@router.post("/stream")
async def text_query_stream(request: QueryRequest):
    session = None
    try:
        kwargs = {}
        if request.session_id:
            session = session_store.get_session(request.session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Chat session not found")
            await session_store.condense_aged_out_turns(session)
            kwargs["conversation_history"] = session_store.build_conversation_history(
                session
            )
        elif request.conversation_history:
            kwargs["conversation_history"] = request.conversation_history
        result = await get_rag().aquery(
            request.query, mode=request.mode, stream=True, **kwargs
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    stream = _stream_result(result)
    if session is not None:
        stream = _stream_and_record(stream, session, request.query)
    return StreamingResponse(stream, media_type="text/plain")


@router.post("/multimodal/stream")
async def multimodal_query_stream(request: MultimodalQueryRequest):
    try:
        content = _to_multimodal_content(request)
        result = await get_rag().aquery_with_multimodal(
            request.query,
            multimodal_content=content,
            mode=request.mode,
            stream=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(_stream_result(result), media_type="text/plain")
