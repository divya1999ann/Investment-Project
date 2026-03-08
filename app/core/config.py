from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Database (PostgreSQL)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@db:5432/investment_committee"

    # Vector DB (pgvector via same PostgreSQL instance)
    VECTOR_DIMENSIONS: int = 1536  # text-embedding-3-small

    # OpenAI
    OPENAI_API_KEY: str

    # OpenAI (for embeddings)
    OPENAI_EMBEDDINGS_KEY: str = ""

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Analysis
    OPENAI_MODEL: str = "gpt-4o-mini"
    MAX_TOKENS: int = 1000
    CHUNK_SIZE: int = 800       # characters per RAG chunk
    CHUNK_OVERLAP: int = 100
    TOP_K_CHUNKS: int = 5       # chunks retrieved per agent query

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
