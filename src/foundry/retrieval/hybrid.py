"""Hybrid retrieval: reciprocal rank fusion plus an authority/recency rerank.

**Why RRF and not score normalisation.** BM25 scores and cosine similarities are
not comparable, and every scheme for making them comparable (min-max, z-score,
softmax) is a tuning liability that quietly changes behaviour whenever the
corpus changes. RRF only reads *ranks*, so it is scale-free and stable:

    score(d) = Σ_i  w_i / (k + rank_i(d))

**Why rerank afterwards and only once.** A regulator's text should outrank a
vendor blog on a tie, and a 2025 revision should outrank a 2015 one. Both are
properties of the source, not of the match, so they are applied once to the
fused ranking rather than inside each retriever -- otherwise a chunk found by
three retrievers gets its authority counted three times.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Sequence

from ..manifest import Manifest
from ..storage import Store
from .base import Candidate, Evidence
from .graph import GraphRetriever
from .bm25 import Bm25KeywordRetriever
from .keyword import KeywordRetriever
from .semantic import SemanticRetriever
from .structured import Filters, StructuredRetriever, infer_filters, merge_cues


@dataclass
class RetrievalResult:
    """Fused evidence plus enough detail to explain why it was chosen."""

    query: str
    evidence: list[Evidence] = field(default_factory=list)
    per_retriever: dict[str, list[str]] = field(default_factory=dict)
    filters: dict = field(default_factory=dict)

    @property
    def top_score(self) -> float:
        return self.evidence[0].score if self.evidence else 0.0

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "evidence": [e.as_dict() for e in self.evidence],
            "per_retriever": self.per_retriever,
            "filters": self.filters,
        }


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Sequence[Candidate]],
    weights: dict[str, float],
    k: int = 60,
) -> dict[str, float]:
    """Fuse ranked lists into one score per chunk id."""
    fused: dict[str, float] = {}
    for retriever, candidates in ranked_lists.items():
        weight = float(weights.get(retriever, 1.0))
        if weight <= 0:
            continue
        for rank, candidate in enumerate(candidates, start=1):
            fused[candidate.chunk_id] = fused.get(candidate.chunk_id, 0.0) + weight / (k + rank)
    return fused


def _recency_factor(published_at: str | None, half_life_days: int) -> float:
    """1.0 for undated sources; decays smoothly with age. Never zero."""
    if not published_at or half_life_days <= 0:
        return 1.0
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            when = _dt.datetime.strptime(str(published_at), fmt)
            break
        except ValueError:
            continue
    else:
        return 1.0
    age_days = max(0.0, (_dt.datetime.now() - when).days)
    return 0.5 + 0.5 * math.exp(-math.log(2) * age_days / half_life_days)


class HybridRetriever:
    """The knowledge area's retrieval front door."""

    def __init__(self, store: Store, manifest: Manifest, embedder=None) -> None:
        self.store = store
        self.manifest = manifest
        cfg = manifest.retrieval
        self.cfg = cfg
        self.retrievers: dict[str, object] = {}
        if cfg.keyword_enabled:
            # BM25 over foundry's own tokenizer, not SQLite's. text.py opens by
            # saying tokenisation choices leak into retrieval, grounding and
            # grading alike and must not drift; keyword retrieval was the one
            # component that drifted, and nothing noticed until a second
            # implementation had to agree with it. Set
            # ``retrieval.keyword.engine: fts5`` to get the old one back.
            self.retrievers["keyword"] = (
                KeywordRetriever(store) if cfg.keyword_engine == "fts5"
                else Bm25KeywordRetriever(store)
            )
        if cfg.semantic_enabled and embedder is not None:
            self.retrievers["semantic"] = SemanticRetriever(store, embedder)
        self.cues = merge_cues(cfg.structured_cues)
        if cfg.structured_enabled:
            self.retrievers["structured"] = StructuredRetriever(store, cues=self.cues)
        if cfg.graph_enabled:
            self.retrievers["graph"] = GraphRetriever(store, max_hops=cfg.graph_max_hops)

    def retrieve(
        self, query: str, top_k: int | None = None, filters: Filters | None = None
    ) -> RetrievalResult:
        cfg = self.cfg
        top_k = top_k or cfg.top_k
        inferred = filters or infer_filters(query, self.cues)

        ranked: dict[str, Sequence[Candidate]] = {}
        seen_text: dict[str, Candidate] = {}
        for name, retriever in self.retrievers.items():
            if name == "structured":
                candidates = retriever.search(query, limit=cfg.candidate_k, filters=inferred)
            else:
                candidates = retriever.search(query, limit=cfg.candidate_k)
            ranked[name] = candidates
            for candidate in candidates:
                seen_text.setdefault(candidate.chunk_id, candidate)

        fused = reciprocal_rank_fusion(ranked, cfg.fusion_weights, k=cfg.fusion_k)
        if not fused:
            return RetrievalResult(query=query, per_retriever={}, filters=inferred.as_dict())

        provenance = self.store.chunks_with_provenance(fused.keys())

        # Authority and recency rerank, applied once to the fused ranking.
        adjusted: list[tuple[str, float]] = []
        for chunk_id, score in fused.items():
            row = provenance.get(chunk_id)
            if row is None:
                continue
            authority = int(row["authority"])
            authority_factor = 1.0 + cfg.authority_weight * ((authority - 3) / 2.0)
            recency = _recency_factor(row["published_at"], cfg.recency_half_life_days)
            recency_factor = 1.0 + cfg.authority_weight * (recency - 1.0)
            adjusted.append((chunk_id, score * authority_factor * recency_factor))
        adjusted.sort(key=lambda kv: kv[1], reverse=True)

        contributors: dict[str, list[str]] = {}
        for name, candidates in ranked.items():
            for candidate in candidates:
                contributors.setdefault(candidate.chunk_id, []).append(name)

        evidence: list[Evidence] = []
        for rank, (chunk_id, score) in enumerate(adjusted[:top_k], start=1):
            row = provenance[chunk_id]
            evidence.append(
                Evidence(
                    rank=rank,
                    chunk_id=chunk_id,
                    text=row["text"],
                    section=row["section"] or "",
                    score=score,
                    retrievers=sorted(set(contributors.get(chunk_id, []))),
                    source_id=row["source_id"],
                    source_title=row["source_title"],
                    publisher=row["publisher"],
                    uri=row["uri"],
                    source_type=row["source_type"],
                    authority=int(row["authority"]),
                    published_at=row["published_at"],
                    retrieved_at=row["retrieved_at"],
                    licence=row["licence"],
                    document_title=row["document_title"],
                )
            )

        return RetrievalResult(
            query=query,
            evidence=evidence,
            per_retriever={
                name: [c.chunk_id for c in candidates[:top_k]]
                for name, candidates in ranked.items()
            },
            filters=inferred.as_dict(),
        )
