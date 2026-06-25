from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .rag_manager import ensure_rag_ready, get_rag, initialize_rag, load_existing_documents
from .routes.documents import router as documents_router
from .routes.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rag()
    await ensure_rag_ready()
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

app.include_router(documents_router)
app.include_router(query_router)


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="web", html=True), name="web")