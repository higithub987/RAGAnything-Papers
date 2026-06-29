from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    llm_binding_api_key: str
    llm_binding_host: str
    llm_model: str = "qwen3.6-plus"
    vision_model: str = "qwen3.6-plus"
    fast_llm_model: str = "qwen3.6-flash"

    # Embedding
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_db_name: str = "lightrag"

    # Storage paths (relative to where the server is launched from)
    working_dir: str = "./rag_storage"
    upload_dir: str = "./uploads"
    output_dir: str = "./output"

    # Safety net: a stuck/hung parser (e.g. mineru) must not block uploads
    # forever. The timeout scales with the uploaded file's size so large
    # documents get the headroom they need without making small uploads
    # wait as long as a worst-case multi-page PDF would. mineru's cold start
    # alone (fresh subprocess + model load on every upload) measured ~7
    # minutes for a single page in this environment, hence the floor.
    document_processing_timeout_base_seconds: int = 1800
    document_processing_timeout_per_mb_seconds: int = 60
    document_processing_timeout_max_seconds: int = 7200

    # How many documents may parse at once. 1 preserves today's serialized
    # behavior (safest for GPU memory contention under mineru's CUDA mode);
    # raise it to let multiple documents parse concurrently.
    max_concurrent_documents: int = 1

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
