from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from raganything.reconciliation import reconcile_orphaned_documents

from .rag_manager import (
    ensure_rag_ready,
    get_rag,
    initialize_rag,
    load_existing_documents,
)
from .routes.containers import router as containers_router
from .routes.documents import router as documents_router
from .routes.query import router as query_router
from .routes.sessions import router as sessions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rag()
    await ensure_rag_ready()
    # Roll back any doc_ids left stuck in HANDLING/PROCESSING/PENDING by a
    # prior crash before load_existing_documents() imports them as active.
    await reconcile_orphaned_documents(get_rag().lightrag)
    load_existing_documents()
    yield
    await get_rag().finalize_storages()


app = FastAPI(title="RAG-Anything API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def revalidate_html(request, call_next):
    """Force browsers to revalidate the served HTML pages on every load.

    StaticFiles doesn't set Cache-Control, so browsers apply heuristic caching
    and can keep running a stale chat.html (old JS) after the file changes.
    "no-cache" means "always revalidate before use" -- combined with the ETag
    StaticFiles already sends, an unchanged file still returns a cheap 304, but
    an edited one is always picked up on a normal refresh.
    """
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.include_router(documents_router)
app.include_router(query_router)
app.include_router(sessions_router)
app.include_router(containers_router)


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="web", html=True), name="web")
