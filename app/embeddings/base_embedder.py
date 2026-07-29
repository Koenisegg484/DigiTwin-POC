"""Abstract interface for embedding providers.

Every concrete embedder (OpenAI, Ollama, HuggingFace, ...) must subclass
``BaseEmbedder``. The rest of the codebase only ever depends on
``BaseEmbedder`` - never a concrete impl. This is what lets us swap
providers without touching the ingestion service or vector store.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Contract: convert text into dense float vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vector this embedder produces."""
        ...

    @abstractmethod
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed multiple texts (used at index time)."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query (used at retrieval time)."""
        ...

    # Convenience: same embedder for both code paths means query/docs
    # share the same vector space. Subclasses inherit this for free.
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
