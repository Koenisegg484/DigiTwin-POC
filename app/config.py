"""Application configuration.

Loads from environment variables (and a local ``.env`` file via
``python-dotenv``) into a singleton ``settings`` object. Run anything that
needs ``OPENAI_API_KEY`` through ``get_settings()`` so the env is first
loaded from ``.env``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env once at import time so settings see overrides.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONBOARDING_", extra="ignore")

    # ---- OpenAI --------------------------------------------------------
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ---- Chunking -----------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---- Chroma -------------------------------------------------------
    chroma_persist_dir: str = "data/chroma"
    # If true, use an in-memory Chroma client instead (handy in tests).
    chroma_in_memory: bool = False

    # ---- API ----------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
