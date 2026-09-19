"""The corpus, packaged so a browser can answer from it (§16).

Without a model, answering is arithmetic over stored text: score passages
against the question, run the gates, select and cite sentences. None of that
needs a server, a key or a network round trip -- it needs the corpus, and the
corpus is small. So it ships with the page and the answering happens where the
reader is.

**What travels and what does not.** Keyword retrieval travels, because BM25
needs only term statistics that are computed from the passages themselves. The
semantic retriever does not: its projection matrix is the vocabulary by the
embedding dimension, tens of megabytes, which is not a page. The structured and
graph retrievers are left out with it, to keep one honest story about what the
browser runs rather than a partial blend.

That sounds like a compromise and measurement says it is not. Scored across the
whole evaluation suite, keyword retrieval alone beats the full hybrid on this
corpus -- 79 of 103 against 77. The browser build is not a lesser copy of the
system; on the numbers it is the better half of it travelling light.

**Provenance travels too.** A passage without its source, licence and retrieval
date is not evidence, and an answer assembled from such passages could not be
checked -- which would make the page a chatbot with extra steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .manifest import Manifest
from .storage import Store

# The retrievers a browser build runs. Named here rather than inferred so the
# page can state it, and so a reader comparing a local answer with a recorded
# one can see why they might differ.
BROWSER_RETRIEVERS = ("keyword",)


@dataclass
class IndexReport:
    knowledge_area: str
    chunks: int
    sources: int
    bytes: int

    def as_dict(self) -> dict:
        return {
            "knowledge_area": self.knowledge_area,
            "chunks": self.chunks,
            "sources": self.sources,
            "bytes": self.bytes,
        }


def build_browser_index(manifest: Manifest, store: Store) -> dict:
    """Everything the in-page engine needs, and nothing it does not."""
    sources = {}
    for row in store.sources():
        sources[row["id"]] = {
            "id": row["id"],
            "title": row["title"],
            "publisher": row["publisher"],
            "uri": row["uri"],
            "source_type": row["source_type"],
            "authority": row["authority"],
            "published_at": row["published_at"] or "",
            "retrieved_at": (row["retrieved_at"] or "")[:10],
            "licence": row["licence"] or "",
            # The licence guarantee is enforced from the register, so the
            # register's own flag has to travel. Without it the page would
            # happily quote round a source the pipeline refuses to.
            "mirrored": bool(row["mirrored"]),
        }

    chunks = [
        {
            "id": row["id"],
            "source_id": row["source_id"],
            "section": row["section"] or "",
            "text": row["text"],
        }
        for row in store.conn.execute(
            "SELECT id, source_id, section, text FROM chunks ORDER BY source_id, ordinal"
        )
    ]

    spec = manifest.specialist
    return {
        "knowledge_area": manifest.id,
        "name": manifest.name,
        "version": store.get_meta("current_version", "unversioned"),
        "evaluation_status": store.get_meta("evaluation_status", "unknown"),
        "retrievers": list(BROWSER_RETRIEVERS),
        "scope": {
            "in_scope": list(manifest.scope.in_scope),
            "out_of_scope": list(manifest.scope.out_of_scope),
        },
        "fusion_k": manifest.retrieval.fusion_k,
        "fusion_weights": {"keyword": manifest.retrieval.fusion_weights.get("keyword", 1.0)},
        "rerank": {
            "authority_weight": manifest.retrieval.authority_weight,
            "recency_half_life_days": manifest.retrieval.recency_half_life_days,
        },
        "thresholds": {
            "top_k": manifest.retrieval.top_k,
            "candidate_k": manifest.retrieval.candidate_k,
            "min_evidence_score": spec.min_evidence_score,
            "min_chunks": spec.min_chunks,
            "max_missing_subject_weight": spec.max_missing_subject_weight,
            "min_question_coverage": spec.min_question_coverage,
            "min_sentence_support": spec.min_sentence_support,
            "max_unsupported_ratio": spec.max_unsupported_ratio,
        },
        "governance": {
            "disclaimer": manifest.governance.disclaimer,
            "known_limitations": list(manifest.governance.known_limitations),
        },
        "sources": sources,
        "chunks": chunks,
    }


def render_browser_index(manifest: Manifest, store: Store) -> tuple[bytes, IndexReport]:
    payload = build_browser_index(manifest, store)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return encoded, IndexReport(
        knowledge_area=manifest.id,
        chunks=len(payload["chunks"]),
        sources=len(payload["sources"]),
        bytes=len(encoded),
    )
