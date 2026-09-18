"""Index building: fit the embedder on the corpus and encode every chunk.

Separated from ingestion because they have different costs and different
triggers. Ingestion is per-source and incremental; the semantic index is
corpus-global -- a corpus-trained embedder must be refitted when the corpus
changes materially, and re-encoding is cheap relative to re-fetching.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..embeddings import embedder_for, model_path, pack
from ..manifest import Manifest
from ..storage import Store


@dataclass
class IndexReport:
    knowledge_area: str
    model: str
    dim: int
    chunks_encoded: int
    skipped: str | None = None

    def as_dict(self) -> dict:
        return {
            "knowledge_area": self.knowledge_area,
            "model": self.model,
            "dim": self.dim,
            "chunks_encoded": self.chunks_encoded,
            "skipped": self.skipped,
        }


def build_semantic_index(manifest: Manifest, store: Store, batch_size: int = 256) -> IndexReport:
    """Fit (if the provider learns from the corpus) and encode all chunks."""
    if not manifest.retrieval.semantic_enabled:
        return IndexReport(manifest.id, "disabled", 0, 0, skipped="semantic retrieval disabled")

    chunks = store.chunks()
    if not chunks:
        return IndexReport(manifest.id, "none", 0, 0, skipped="no chunks to index")

    embedder = embedder_for(manifest)
    texts = [
        # The section heading is prepended because it carries context the chunk
        # body often assumes -- the same reason it is a separate FTS column.
        f"{row['section']}\n{row['text']}" if row["section"] else row["text"]
        for row in chunks
    ]
    embedder.fit(texts)
    embedder.save(model_path(manifest))

    store.clear_embeddings()
    encoded = 0
    with store.transaction():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = embedder.encode(batch)
            for row, vector in zip(chunks[start : start + batch_size], vectors):
                store.set_embedding(row["id"], embedder.name, len(vector), pack(vector))
                encoded += 1

    store.set_meta("embedding_model", embedder.name)
    store.set_meta("embedding_dim", str(getattr(embedder, "dim", 0)))
    return IndexReport(manifest.id, embedder.name, int(getattr(embedder, "dim", 0)), encoded)


def load_embedder(manifest: Manifest):
    """Load the persisted embedder for query encoding, or None if absent."""
    if not manifest.retrieval.semantic_enabled:
        return None
    path = model_path(manifest)
    if not path.exists():
        return None
    embedder = embedder_for(manifest)
    try:
        embedder.load(path)
    except Exception:
        return None
    return embedder
