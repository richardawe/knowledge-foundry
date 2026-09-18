"""Semantic retrieval: cosine similarity over stored chunk embeddings.

A flat scan, deliberately. At MVP corpus sizes (10^4-10^5 chunks) it costs
milliseconds and has no index to corrupt, no rebuild step and no extra service.
Swapping in an ANN index later changes this file and nothing else.
"""

from __future__ import annotations

from typing import Sequence

from ..embeddings import cosine, unpack
from ..storage import Store
from .base import Candidate


class SemanticRetriever:
    name = "semantic"

    def __init__(self, store: Store, embedder) -> None:
        self.store = store
        self.embedder = embedder
        self._cache: list[tuple[str, list[float]]] | None = None

    def _vectors(self) -> list[tuple[str, list[float]]]:
        if self._cache is None:
            self._cache = [
                (row["chunk_id"], unpack(row["vector"])) for row in self.store.embeddings()
            ]
        return self._cache

    def search(self, query: str, limit: int = 40) -> Sequence[Candidate]:
        vectors = self._vectors()
        if not vectors:
            return []
        try:
            query_vector = self.embedder.encode([query])[0]
        except Exception:
            # A missing or unreachable embedding model must degrade retrieval,
            # never take down the answer path: the other retrievers still work.
            return []
        if not any(query_vector):
            return []

        scored = [
            (chunk_id, cosine(query_vector, vector))
            for chunk_id, vector in vectors
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        top = [(cid, score) for cid, score in scored[:limit] if score > 0.0]
        if not top:
            return []

        rows = self.store.chunks_with_provenance([cid for cid, _ in top])
        out: list[Candidate] = []
        for chunk_id, score in top:
            row = rows.get(chunk_id)
            if row is None:
                continue
            out.append(
                Candidate(
                    chunk_id=chunk_id,
                    score=float(score),
                    retriever=self.name,
                    text=row["text"],
                    section=row["section"],
                    source_id=row["source_id"],
                    explain=f"cos={score:.3f}",
                )
            )
        return out
