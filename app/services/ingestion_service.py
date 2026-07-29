"""IngestionService — the orchestrator of the index pipeline.

Pipeline:

    Loaders -> Documents -> Chunker -> Chunks
        -> Embedder       -> Embeddings
        -> VectorStore    -> Chroma collection

This class depends only on:
  * ``BaseEmbedder``        (abstract)
  * ``BaseVectorStore``     (abstract)
  * ``DocumentChunker``     (concrete; pure-Python, no providers to swap)

It never imports OpenAI or ChromaDB. Both are injected from the caller
(e.g. ``main.py``, a CLI, tests). That keeps the orchestration logic
provider-agnostic and trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.embeddings.base_embedder import BaseEmbedder
from app.processors.chunker import DocumentChunker
from app.vectorstores.base_store import BaseVectorStore


@dataclass
class IngestionResult:
    """Summary of a single ingest() run, for logging / API responses."""

    project_id: str
    documents_loaded: int
    chunks_generated: int
    chunks_stored: int
    per_source: dict[str, int]  # source -> chunk count, mirrors the chunker summary


class IngestionService:
    """Wires loaders / chunker / embedder / vector store together.

    Designed for dependency injection - everything it needs arrives as a
    constructor argument. The class itself owns no provider-specific state.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        chunker: DocumentChunker | None = None,
    ):
        self._embedder = embedder
        self._store = vector_store
        self._chunker = chunker or DocumentChunker()

    # ---- public API -----------------------------------------------------
    def ingest(
        self,
        project_id: str,
        documents: list[Document],
        reset: bool = False,
    ) -> IngestionResult:
        """Run the full pipeline for one project.

        Args:
            project_id: Logical project name; becomes the Chroma collection.
            documents: Already-loaded Documents (from any loader).
            reset: If True, drop the existing collection before indexing.
                    Useful for a clean re-index.
        """
        if not documents:
            return IngestionResult(
                project_id=project_id,
                documents_loaded=0,
                chunks_generated=0,
                chunks_stored=0,
                per_source={},
            )

        # 1. Ensure (or reset) the project collection.
        if reset:
            self._store.delete_collection(project_id)
        self._store.create_collection(project_id)

        # 2. Chunk the documents, preserving + enriching metadata.
        chunks = self._chunker.chunk(documents)

        # 3. Stamp project_id onto every chunk's metadata so queries can
        #    be filtered by project later, and citations carry origin info.
        for c in chunks:
            c.metadata["project_id"] = project_id

        # 4. Embed all chunk texts in one batch (embedder does its own batching).
        texts = [c.page_content for c in chunks]
        vectors = self._embedder.embed_documents(texts)

        # 5. Build stable, human-readable ids: <project>_chunk_0001
        #    Stable ids = idempotent ingest (re-running the same source
        #    yields the same ids so we can upsert vs duplicate-noted later).
        ids = [self._make_id(project_id, i) for i in range(len(chunks))]

        # 6. Persist to the vector store.
        self._store.add_documents(
            project_id=project_id,
            documents=chunks,
            embeddings=vectors,
            ids=ids,
        )

        # 7. Per-source breakdown for the summary output (matches the
        #    spec's "README.md -> 7 chunks" format).
        per_source: dict[str, int] = {}
        for c in chunks:
            src = _basename(c.metadata.get("source", "?"))
            per_source[src] = per_source.get(src, 0) + 1

        return IngestionResult(
            project_id=project_id,
            documents_loaded=len(documents),
            chunks_generated=len(chunks),
            chunks_stored=len(chunks),
            per_source=per_source,
        )

    def query(
        self,
        project_id: str,
        question: str,
        k: int = 4,
        where: dict[str, Any] | None = None,
    ):
        """Index-time helper used in tests; production retrieval sits in
        the retrieval service. Kept here so we can *verify* indexing
        before the LLM is wired up.
        """
        query_vec = self._embedder.embed_query(question)
        return self._store.similarity_search(
            project_id=project_id,
            query_embedding=query_vec,
            k=k,
            where=where,
        )

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _make_id(project_id: str, index: int) -> str:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in project_id.lower())
        return f"{safe}_chunk_{index:04d}"


def _basename(path: str) -> str:
    return str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
