"""Abstract vector store contract.

Keeps us honest about the separation: the ingestion service depends on
``BaseVectorStore``, never on ChromaDB. Tomorrow we can add a
``PineconeStore`` or ``QdrantStore`` subclass and nothing else changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass
class SearchHit:
    """One similarity-search result."""

    document: Document
    score: float
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """Contract for a vector store that consumes pre-computed embeddings."""

    @abstractmethod
    def create_collection(self, project_id: str) -> None:
        """Ensure the collection for ``project_id`` exists."""
        ...

    @abstractmethod
    def add_documents(
        self,
        project_id: str,
        documents: list[Document],
        embeddings: list[list[float]],
        ids: list[str],
    ) -> None:
        """Store chunks + their (already computed) embeddings + ids."""
        ...

    @abstractmethod
    def similarity_search(
        self,
        project_id: str,
        query_embedding: list[float],
        k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return the top-k nearest chunks for the given query vector."""
        ...

    @abstractmethod
    def delete_collection(self, project_id: str) -> None:
        """Remove an entire project collection and everything in it."""
        ...

    @abstractmethod
    def count(self, project_id: str) -> int:
        """Number of chunks stored for the project."""
        ...
