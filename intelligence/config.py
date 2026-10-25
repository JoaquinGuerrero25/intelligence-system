"""Configuración central: todas las variables de entorno se leen y validan acá (nada hard-codeado)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

LLMProvider = Literal["mock", "openai", "google"]
EmbeddingProvider = Literal["local", "openai", "google"]
VectorBackend = Literal["memory", "pinecone"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore", case_sensitive=False)

    # --- LLM ---
    llm_provider: LLMProvider = "mock"
    llm_model: str = ""
    llm_requests_per_minute: float | None = None
    openai_api_key: str = ""
    google_api_key: str = ""

    # --- RAG ---
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = ""
    embedding_dimension: int = Field(256, ge=8)
    vector_backend: VectorBackend = "memory"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "intelligence-kb"
    pinecone_namespace: str = "productos-es"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    rag_top_k: int = Field(5, ge=1, le=20)
    rag_bm25_weight: float = Field(0.5, ge=0)
    rag_vector_weight: float = Field(0.5, ge=0)
    knowledge_base_path: Path = ROOT_DIR / "data" / "knowledge_base.json"

    # --- Infraestructura ---
    redis_url: str = "redis://localhost:6379"
    worker_concurrency: int = Field(5, ge=1, le=64)
    job_timeout_s: float = Field(600, gt=0)
    log_level: str = "INFO"

    # --- Observabilidad ---
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "intelligence-system"

    @model_validator(mode="after")
    def _credenciales_coherentes(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai requiere OPENAI_API_KEY.")
        if self.llm_provider == "google" and not self.google_api_key:
            raise ValueError("LLM_PROVIDER=google requiere GOOGLE_API_KEY.")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai requiere OPENAI_API_KEY.")
        if self.embedding_provider == "google" and not self.google_api_key:
            raise ValueError("EMBEDDING_PROVIDER=google requiere GOOGLE_API_KEY.")
        if self.vector_backend == "pinecone" and not self.pinecone_api_key:
            raise ValueError("VECTOR_BACKEND=pinecone requiere PINECONE_API_KEY.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
