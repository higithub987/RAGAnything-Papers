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
