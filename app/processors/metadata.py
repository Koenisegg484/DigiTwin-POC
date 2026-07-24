"""Helpers for building and merging chunk metadata.

Chunk metadata is everything we add to a `Document`'s metadata *after*
splitting. Keeping these helpers separate keeps the chunker focused on
splitting logic, not metadata plumbing.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any

from langchain_core.documents import Document


def chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Deterministic, collision-resistant id for a chunk.

    Combines source + chunk index + content hash so re-indexing the same
    file yields stable ids (useful for incremental ChromaDB upserts).
    """
    digest = hashlib.sha1(
        f"{source}::{chunk_index}::{content}".encode("utf-8")
    ).hexdigest()
    return f"chunk_{uuid.uuid5(uuid.NAMESPACE_URL, source)}_{chunk_index}_{digest[:8]}"


def build_chunk_metadata(
    original: dict[str, Any], chunk_index: int, total_chunks: int, content: str
) -> dict[str, Any]:
    """Return a NEW metadata dict: original metadata preserved +
    chunk-specific fields appended.

    Never mutates the original metadata dict so callers can reuse it
    across chunks safely.
    """
    new_meta = dict(original)  # preserve everything inherited from the doc
    new_meta["chunk_index"] = chunk_index
    new_meta["total_chunks"] = total_chunks
    new_meta["chunk_id"] = chunk_id(
        str(original.get("source", "unknown")), chunk_index, content
    )
    return new_meta


def make_chunk(
    content: str, metadata: dict[str, Any], chunk_index: int, total_chunks: int
) -> Document:
    """Construct a chunk `Document` carrying preserved + extended metadata."""
    return Document(
        page_content=content,
        metadata=build_chunk_metadata(metadata, chunk_index, total_chunks, content),
    )
