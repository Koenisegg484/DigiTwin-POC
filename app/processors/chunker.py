"""DocumentChunker — splits loaded Documents into embedded-able chunks.

Responsibilities:
  * accept ``list[Document]``
  * split with ``RecursiveCharacterTextSplitter``
  * **preserve** every metadata field from the source document
  * **add**  ``chunk_id`` / ``chunk_index`` / ``total_chunks`` per chunk

It does NOT embed, store, or otherwise touch ChromaDB. One job.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.processors.metadata import make_chunk


class DocumentChunker:
    """Wraps ``RecursiveCharacterTextSplitter`` with metadata preservation."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_document(self, document: Document) -> list[Document]:
        """Split a single Document, preserving its metadata on every chunk."""
        texts = self._splitter.split_text(document.page_content)
        total = len(texts)
        if total == 0:
            return []

        # Guard: recursive splitter may return a single chunk identical to the
        # source. In that case we still emit one Document so metadata stays
        # consistent across the index.
        return [
            make_chunk(content=text, metadata=document.metadata,
                       chunk_index=i, total_chunks=total)
            for i, text in enumerate(texts)
        ]

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split many documents into chunked Documents in order."""
        chunks: list[Document] = []
        for doc in documents:
            chunks.extend(self.chunk_document(doc))
        return chunks

    # Backwards-compatible alias matching the spec wording "split".
    split = chunk

    def summarize(self, documents: list[Document], chunks: list[Document]) -> None:
        """Pretty-print the chunking summary requested in the spec.

        Example::

            Loaded documents: 18
            Generated chunks: 146

            README.md -> 7 chunks
            authentication.md -> 5 chunks
            database.pdf -> 19 chunks
        """
        print(f"Loaded documents: {len(documents)}")
        print(f"Generated chunks: {len(chunks)}")
        print()
        for doc in documents:
            src = doc.metadata.get("source", "?")
            name = str(src).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            count = sum(
                1 for c in chunks if c.metadata.get("source") == src
            )
            print(f"{name} -> {count} chunks")
