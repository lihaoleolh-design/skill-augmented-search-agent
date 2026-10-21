from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ==============================
    # LLM
    # ==============================

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    llm_temperature: float = 0
    llm_max_tokens: int = Field(default=2048, ge=1)
    llm_timeout: int = Field(default=120, ge=1)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_enable_thinking: bool = False

    # ==============================
    # Dataset
    # ==============================

    corpus_path: Path = Path("./data/corpus")

    # ==============================
    # Retrieval
    # ==============================

    bm25_db_path: Path = Path("./indexes/bm25_full.sqlite")

    retrieval_top_k: int = Field(default=8, ge=1, le=100)
    retrieval_candidate_k: int = Field(default=50, ge=1, le=1000)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)
    retrieval_method: str = "bm25"

    dense_model_name: str = "BAAI/bge-large-en-v1.5"
    dense_vector_size: int = Field(default=1024, ge=1)
    dense_api_url: str = "https://api.siliconflow.cn/v1/embeddings"
    dense_api_timeout: int = Field(default=120, ge=1)
    dense_api_max_retries: int = Field(default=3, ge=0, le=10)
    dense_batch_size: int = Field(default=32, ge=1)
    dense_chunk_tokens: int = Field(default=384, ge=32, le=510)
    dense_chunk_overlap: int = Field(default=64, ge=0, le=256)
    qdrant_path: Path = Path("./indexes/qdrant")
    # Use the service from compose.yaml by default. Set this to an empty value
    # to use the embedded, file-backed Qdrant database at qdrant_path instead.
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "browsecomp_bge_large_en_v1_5"

    # ==============================
    # Document Tools
    # ==============================

    max_read_chars: int = Field(default=4000, ge=1)
    max_matches: int = Field(default=10, ge=1)
    max_context_chars: int = Field(default=500, ge=0)

    # ==============================
    # Agent
    # ==============================

    max_agent_steps: int = Field(default=10, ge=1)
    max_tool_calls: int = Field(default=15, ge=1)
    max_stalled_searches: int = Field(default=3, ge=1, le=10)
    max_repeated_candidates: int = Field(default=2, ge=1, le=10)

    # ==============================
    # Skills
    # ==============================

    skills_path: Path = Path("./skills")

    # ==============================
    # Logging
    # ==============================

    log_level: str = "INFO"

    def resolve_path(self, path: Path) -> Path:
        """Resolve relative paths from the project root."""
        if path.is_absolute():
            return path

        return BASE_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
