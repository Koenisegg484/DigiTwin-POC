"""OpenAI-backed implementation of ``BaseEmbedder``.

Knows *only* about OpenAI. Has no idea ChromaDB, the ingestion service,
or any vector store exists. That decoupling is the entire point of the
``BaseEmbedder`` interface.
"""
from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from app.embeddings.base_embedder import BaseEmbedder


# OpenAI's small embedding model is cheap, fast and 1536-dim - a good
# default. Override via the constructor when needed.
_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMENSION = 1536
# OpenAI allows up to 2048 inputs per request; we stay safely under.
_BATCH_SIZE = 512


class OpenAIEmbedder(BaseEmbedder):
    """Thin wrapper around the OpenAI ``embeddings`` endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        dimensions: int | None = None,
        client: OpenAI | None = None,
    ):
        # Allow injecting a pre-built client (handy for tests / proxies).
        self._client = client or OpenAI(api_key=api_key)
        self._model = model
        # If the caller doesn't pin a dimension, fall back to the model's
        # native size so self.dimension stays consistent in both paths.
        self._dim = dimensions or _DEFAULT_DIMENSION

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        if not documents:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(documents), _BATCH_SIZE):
            batch = documents[start : start + _BATCH_SIZE]
            resp = self._client.embeddings.create(
                input=batch,
                model=self._model,
                dimensions=self._dim,
            )
            # OpenAI returns embeddings in the same order as the input batch.
            vectors.extend([d.embedding for d in resp.data])
        return vectors

    def embed_query(self, query: str) -> list[float]:
        resp = self._client.embeddings.create(
            input=query,
            model=self._model,
            dimensions=self._dim,
        )
        return resp.data[0].embedding

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(list(texts))
