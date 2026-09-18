"""Structured retrieval: metadata, classification and time (§8).

Answers the questions the other two retrievers cannot express at all: "what do
*regulators* say", "what changed since 2023", "what does the accident record
show". It reads intent from the query with a small, auditable set of cues and
then filters in SQL, ranking what survives by term overlap and source authority.

The cues are heuristics and are meant to be. A wrong guess costs one retriever's
contribution to a fused ranking, never the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..storage import Store
from ..text import STOPWORDS, coverage, content_terms, tokenize
from .base import Candidate

# Query cue -> source_type. Ordered scanning, first match wins per type.
_TYPE_CUES: dict[str, tuple[str, ...]] = {
    "regulatory": ("regulation", "regulatory", "legally", "law", "statutory", "directive", "regulator", "mandated", "compliance"),
    "standard": ("standard", "standards", "clause", "iec", "nfpa", "iso", "astm", "certification", "test method"),
    "investigation": ("incident", "accident", "investigation", "fire at", "failure report", "post-incident", "root cause", "case study"),
    "academic": ("study", "studies", "research", "paper", "literature", "experiment", "measured", "journal"),
    "industry": ("industry", "vendor", "manufacturer", "guidance", "best practice"),
    "dataset": ("dataset", "data set", "measurements", "test data"),
}
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SINCE_RE = re.compile(r"\b(since|after|from)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_BEFORE_RE = re.compile(r"\b(before|prior to|up to)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_LATEST_RE = re.compile(r"\b(latest|current|most recent|newest|up to date|today)\b", re.IGNORECASE)


@dataclass
class Filters:
    """Structured constraints read off a query, or supplied explicitly."""

    source_types: list[str] = field(default_factory=list)
    published_after: str | None = None
    published_before: str | None = None
    prefer_recent: bool = False
    min_authority: int = 1

    def is_empty(self) -> bool:
        return not (
            self.source_types
            or self.published_after
            or self.published_before
            or self.prefer_recent
            or self.min_authority > 1
        )

    def as_dict(self) -> dict:
        return {
            "source_types": self.source_types,
            "published_after": self.published_after,
            "published_before": self.published_before,
            "prefer_recent": self.prefer_recent,
            "min_authority": self.min_authority,
        }


def infer_filters(query: str) -> Filters:
    """Read structured intent from a natural-language query."""
    lowered = query.lower()
    filters = Filters()

    for source_type, cues in _TYPE_CUES.items():
        if any(cue in lowered for cue in cues):
            filters.source_types.append(source_type)

    since = _SINCE_RE.search(query)
    if since:
        filters.published_after = f"{since.group(2)}-01-01"
    before = _BEFORE_RE.search(query)
    if before:
        filters.published_before = f"{before.group(2)}-01-01"
    if _LATEST_RE.search(query):
        filters.prefer_recent = True
    return filters


class StructuredRetriever:
    name = "structured"

    def __init__(self, store: Store) -> None:
        self.store = store

    def search(self, query: str, limit: int = 40, filters: Filters | None = None) -> Sequence[Candidate]:
        filters = filters or infer_filters(query)
        if filters.is_empty():
            return []

        clauses = ["s.status = 'indexed'"]
        params: list = []
        if filters.source_types:
            placeholders = ", ".join("?" * len(filters.source_types))
            clauses.append(f"s.source_type IN ({placeholders})")
            params.extend(filters.source_types)
        if filters.published_after:
            clauses.append("s.published_at >= ?")
            params.append(filters.published_after)
        if filters.published_before:
            clauses.append("s.published_at < ?")
            params.append(filters.published_before)
        if filters.min_authority > 1:
            clauses.append("s.authority >= ?")
            params.append(filters.min_authority)

        order = "s.published_at DESC" if filters.prefer_recent else "s.authority DESC"
        rows = self.store.conn.execute(
            f"""
            SELECT c.id AS chunk_id, c.text, c.section, c.source_id,
                   s.authority, s.published_at
            FROM chunks c
            JOIN sources s ON s.id = c.source_id
            WHERE {' AND '.join(clauses)}
            ORDER BY {order}
            LIMIT ?
            """,
            (*params, limit * 20),
        ).fetchall()
        if not rows:
            return []

        terms = content_terms(query)
        scored: list[Candidate] = []
        for row in rows:
            chunk_terms = {t for t in tokenize(row["text"]) if t not in STOPWORDS}
            overlap = coverage(terms, chunk_terms)
            if overlap <= 0.0:
                continue
            # Authority is a mild tie-breaker here; the real authority rerank
            # happens once, after fusion, so it is not applied twice.
            score = overlap * (1.0 + 0.05 * (int(row["authority"]) - 3))
            scored.append(
                Candidate(
                    chunk_id=row["chunk_id"],
                    score=score,
                    retriever=self.name,
                    text=row["text"],
                    section=row["section"],
                    source_id=row["source_id"],
                    explain=f"overlap={overlap:.2f} auth={row['authority']}",
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:limit]
