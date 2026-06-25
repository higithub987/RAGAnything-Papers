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
    # forever. After this many seconds, the task is marked failed instead.
    # mineru's cold start alone (fresh subprocess + model load on every
    # upload) measured ~7 minutes for a single page in this environment, so
    # this needs real headroom for larger/multi-page documents.
    document_processing_timeout_seconds: int = 1800

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
