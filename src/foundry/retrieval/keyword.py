"""Keyword retrieval over SQLite FTS5 with BM25.

This retriever earns its place on exactly the queries a vector index handles
worst: ``UL 9540A``, ``UN 38.3``, ``LiPF6``, ``IEC 62619``, clause numbers and
part codes. In a technical domain those are most of the high-value queries.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..storage import Store
from ..text import STOPWORDS, tokenize
from .base import Candidate

# Quoted phrases and bare alphanumerics with separators (e.g. "9540a", "62619")
_PHRASE_RE = re.compile(r'"([^"]+)"')


def build_match_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Every token is quoted, which neutralises FTS5 operator syntax in user input
    -- a query containing ``AND`` or ``*`` must not be able to change the search
    semantics, let alone error out.
    """
    phrases = [p.strip() for p in _PHRASE_RE.findall(query) if p.strip()]
    remainder = _PHRASE_RE.sub(" ", query)
    tokens = [t for t in tokenize(remainder) if t not in STOPWORDS and len(t) > 1]

    parts: list[str] = []
    for phrase in phrases:
        escaped = phrase.replace('"', '""')
        parts.append(f'"{escaped}"')
    for token in tokens:
        escaped = token.replace('"', '""')
        parts.append(f'"{escaped}"')

    if not parts:
        return ""
    return " OR ".join(parts)


class KeywordRetriever:
    name = "keyword"

    def __init__(self, store: Store) -> None:
        self.store = store

    def search(self, query: str, limit: int = 40) -> Sequence[Candidate]:
        match = build_match_query(query)
        if not match:
            return []
        rows = self.store.conn.execute(
            """
            SELECT c.id AS chunk_id, c.text, c.section, c.source_id,
                   bm25(chunks_fts, 1.0, 0.6) AS rank_score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank_score
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
        # FTS5 bm25() is negative and lower-is-better; flip it so that across the
        # whole system "higher score is better" holds without exception.
        return [
            Candidate(
                chunk_id=row["chunk_id"],
                score=-float(row["rank_score"]),
                retriever=self.name,
                text=row["text"],
                section=row["section"],
                source_id=row["source_id"],
                explain=f"bm25={row['rank_score']:.3f}",
            )
            for row in rows
        ]
