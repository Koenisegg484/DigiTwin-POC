"""Retriever - one job: turn a question into the top-k relevant chunks.

Pipeline (per the architecture diagram):

    Question
        |
        v
    Embed question          (BaseEmbedder)
        |
        v
    Search Chroma          (BaseVectorStore)
        |
        v
    Return top-k chunks

Critically it knows NOTHING about an LLM. No OpenAI chat model, no
LangChain chain, no prompt engineering lives here. That separation is
what lets us evaluate retrieval quality *before* wiring generation in,
and what keeps the Retriever reusable in non-LLM contexts (e.g. a
"find similar docs" UI button).

Metadata filters (Step 4 of the spec) are accepted but optional; the
caller can pass e.g. ``filters={"file_type": "markdown"}`` or
``filters={"language": "python"}`` without touching this class again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.embeddings.base_embedder import BaseEmbedder
from app.vectorstores.base_store import BaseVectorStore, SearchHit


@dataclass
class RetrievalResult:
    """A returned chunk + why it was returned.

    Wraps the lower-level ``SearchHit`` from the vector store so the
    retriever stays agnostic to the exact hit type of any future store
    (Pinecone, Qdrant, ...) while giving callers a stable surface.
    """

    page_content: str
    metadata: dict[str, Any]
    score: float

    # Convenience accessor; matches the spec's print loop:
    #     print(result.metadata["source"]); print(result.page_content)
    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "?"))

    @property
    def chunk_id(self) -> str | None:
        return self.metadata.get("chunk_id")


class Retriever:
    """Top-k similarity retriever over a project's indexed documentation.

    Constructor takes the same two abstract interfaces the
    ``IngestionService`` does, so a single embedder + store pair powers
    both indexing and retrieval - the vectors are guaranteed to share
    the same space.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        default_k: int = 4,
    ):
        self._embedder = embedder
        self._store = vector_store
        self._default_k = default_k

    def retrieve(
        self,
        project_id: str,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-k chunks for ``query`` within ``project_id``.

        Args:
            project_id: Which project collection to search.
            query: Natural-language question.
            k: Override the retriever's default k. ``None`` falls back to
                the constructor default (4 per spec).
            filters: Optional Chroma-style ``where`` metadata filter.

        Returns:
            Chunks ranked best-first. Empty list if the project has no
            collection yet or no documents match.
        """
        n = k or self._default_k
        # 1. Embed the question with the same embedder the indexer used.
        query_vec = self._embedder.embed_query(query)
        # 2. Delegate search to the vector store - we never reach into
        #    Chroma directly from here.
        hits: list[SearchHit] = self._store.similarity_search(
            project_id=project_id,
            query_embedding=query_vec,
            k=n,
            where=filters,
        )
        # 3. Wrap hits in a stable retriever-level type so downstream
        #    code (RAG chain, API) doesn't bind to SearchHit.
        return [
            RetrievalResult(
                page_content=h.document.page_content,
                metadata=h.metadata,
                score=h.score,
            )
            for h in hits
        ]

    # Spec-style alias.
    search = retrieve


# --- Filter helpers -----------------------------------------------------
# Small builders so callers don't have to remember Chroma's `$eq`-default
# syntax. Keeps filter construction readable and explicit at call sites.


def filter_file_type(value: str) -> dict[str, Any]:
    """e.g. ``filter_file_type("markdown")`` -> only markdown chunks."""
    return {"file_type": value}


def filter_language(value: str) -> dict[str, Any]:
    """e.g. ``filter_language("python")`` -> only Python code chunks."""
    return {"language": value}
