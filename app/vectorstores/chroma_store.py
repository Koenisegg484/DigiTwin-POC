"""ChromaDB implementation of ``BaseVectorStore``.

Important: this class **never** computes embeddings. The caller supplies
vectors and ids - Chroma is treated as a pure storage + ANN layer.

One Chroma collection per project: ``project_pingpal``, ``project_inv-1``.
This keeps documentation from different projects from polluting each
other's results, and lets us wipe / re-index a single project cleanly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from app.vectorstores.base_store import BaseVectorStore, SearchHit


# Chroma requires metadata values to be primitives (str/int/float/bool).
# Anything else blows up at add() time, so we scrub defensively.
def _scrub(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _scrub_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: _scrub(v) for k, v in meta.items()}


class ChromaStore(BaseVectorStore):
    """Persistent (or in-memory) Chroma client keyed by project id."""

    def __init__(self, persist_directory: str | None = None):
        """Create the store.

        Args:
            persist_directory: Folder for on-disk persistence. ``None`` runs
                entirely in-memory (useful for tests).
        """
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        self._persist_directory = persist_directory
        # In-process cache so create_collection is idempotent & cheap.
        self._collections: dict[str, Any] = {}

    # ---- helpers ---------------------------------------------------------
    def _collection_name(self, project_id: str) -> str:
        # Lowercase + replace anything not [a-z0-9._-] so Chroma accepts it.
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in project_id.lower())
        return f"project_{safe}"

    def _get_collection(self, project_id: str):
        name = self._collection_name(project_id)
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    # ---- BaseVectorStore impl -------------------------------------------
    def create_collection(self, project_id: str) -> None:
        # get_or_create => idempotent.
        self._get_collection(project_id)

    def add_documents(
        self,
        project_id: str,
        documents: list[Document],
        embeddings: list[list[float]],
        ids: list[str],
    ) -> None:
        if not (len(documents) == len(embeddings) == len(ids)):
            raise ValueError(
                "documents / embeddings / ids must have equal length: "
                f"{len(documents)}/{len(embeddings)}/{len(ids)}"
            )
        if not documents:
            return

        collection = self._get_collection(project_id)
        texts = [d.page_content for d in (documents or [])]
        metadatas = [_scrub_meta(d.metadata) for d in (documents or [])]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        project_id: str,
        query_embedding: list[float],
        k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        collection = self._get_collection(project_id)
        # Pass query_embeddings (NOT query_texts): we already embedded.
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        hits: list[SearchHit] = []
        ids_batch = result.get("ids", [[]])
        docs_batch = result.get("documents", [[]])
        metas_batch = result.get("metadatas", [[]])
        dist_batch = result.get("distances", [[]])
        # query() with a single query returns one batch.
        ids_list = ids_batch[0] if ids_batch else []
        docs_list = docs_batch[0] if docs_batch else []
        metas_list = metas_batch[0] if metas_batch else []
        dists_list = dist_batch[0] if dist_batch else []
        for i in range(len(docs_list)):
            doc = Document(
                page_content=docs_list[i],
                metadata=metas_list[i] if i < len(metas_list) else {},
            )
            # Chroma returns a *distance*; cosine space => smaller is better.
            # Convert to a similarity score in [0,1] for readability.
            dist = dists_list[i] if i < len(dists_list) else 1.0
            score = 1.0 - float(dist)
            hits.append(SearchHit(document=doc, score=score, metadata=doc.metadata))
        return hits

    def delete_collection(self, project_id: str) -> None:
        name = self._collection_name(project_id)
        try:
            self._client.delete_collection(name=name)
        except Exception:
            # Tolerant: deleting a missing collection shouldn't crash callers.
            pass
        self._collections.pop(name, None)

    def count(self, project_id: str) -> int:
        collection = self._get_collection(project_id)
        return collection.count()
