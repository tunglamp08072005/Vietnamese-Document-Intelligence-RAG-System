from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Vietnamese Document Intelligence"
    database_path: str = "./data/app.db"
    qdrant_path: str = "./data/qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "vietnamese_documents"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str | None = None
    ocr_language: str = "vie+eng"
    ocr_dpi: int = 250
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""
    ollama_timeout_seconds: float = 600.0
    ollama_keep_alive: str = "10m"
    max_upload_mb: int = 25
    chunk_size_chars: int = 2000
    chunk_overlap_chars: int = 300
    embedding_batch_size: int = 8
    index_batch_size: int = 128
    retrieval_candidate_count: int = 40
    retrieval_top_k: int = 8
    rrf_k: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
