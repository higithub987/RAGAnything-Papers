from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    llm_binding_api_key: str
    llm_binding_host: str
    llm_model: str = "qwen3.7-max-2026-06-08"
    vision_model: str = "qwen-vl-plus"
    fast_llm_model: str = "qwen3.6-flash"

    # Embedding
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_db_name: str = "lightrag"
    # How long to wait at startup for Milvus to become *query-ready* (not just
    # container-up) before failing. Milvus's proxy answers on 19530 before its
    # QueryNodes can serve a load; starting the API in that window makes the
    # blocking load_collection() hang. We gate on this instead (see
    # milvus_health.wait_for_milvus_ready). Generous so a slow cold Milvus start
    # isn't aborted; a truly-down Milvus fails loudly after it elapses.
    milvus_ready_timeout_seconds: int = 120

    # Storage paths (relative to where the server is launched from)
    working_dir: str = "./rag_storage"
    upload_dir: str = "./uploads"
    output_dir: str = "./output"

    # Parser selection.
    #   'mineru-api' — MinerU's hosted cloud API (default). No local GPU/model,
    #                  and boot is instant (the local 'mineru' install-check
    #                  shells out to `mineru --version`, which loads MinerU's
    #                  full ML stack and adds ~70s to every startup).
    #   'mineru'     — local MinerU (needs GPU + models; slow boot).
    parser: str = "mineru-api"
    parse_method: str = "auto"

    # MinerU cloud API (used when parser == 'mineru-api'). These are mirrored into
    # os.environ at startup because MineruApiParser reads them from the environment
    # (see rag_manager.initialize_rag). Get a token from https://mineru.net.
    mineru_api_token: str = ""
    mineru_api_base_url: str = "https://mineru.net"
    mineru_api_model_version: str = "pipeline"
    mineru_api_insecure: bool = False

    # Safety net: a stuck/hung parser (e.g. mineru) must not block uploads
    # forever. Page count isn't known before parsing starts (office/text
    # files are only converted to PDF partway through the pipeline, and
    # reading real PDF page counts would mean adding pypdfium2 as a new core
    # dependency just for this — see raganything/parser.py's OFFICE_FORMATS
    # conversion). File size is instead used as an inexpensive proxy for page
    # count: divide by an assumed average page size, then multiply by
    # mineru's measured cold-start cost of ~7 minutes/page.
    document_processing_kb_per_page: int = 200
    document_processing_seconds_per_page: int = 420  # 7 minutes
    document_processing_timeout_max_seconds: int = 7200

    # How many documents may parse at once. 1 preserves today's serialized
    # behavior (safest for GPU memory contention under mineru's CUDA mode);
    # raise it to let multiple documents parse concurrently.
    max_concurrent_documents: int = 1

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
