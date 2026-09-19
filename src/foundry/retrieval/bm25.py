"""A BM25 keyword retriever over foundry's own tokenizer.

text.py opens by saying tokenisation choices leak into retrieval scores,
grounding decisions and grading alike, and that when they drift apart the
metrics stop meaning what they say. Keyword retrieval was the one place that
drifted: it ran on SQLite's unicode61 tokenizer while every other component ran
on `foundry.text.tokenize`.

Invisible until something had to agree with it. The browser engine shares
foundry's tokenizer and reproduced only half of Python's top-k as a result;
with this retriever, the same tokenizer on both sides, and the fusion and
rerank ported, the two agree on all 103 evaluation questions exactly.

Scored against the suite it is worth what the FTS5 retriever was worth -- 78 of
103 against 79, both above the 77 the full hybrid scores -- so the shared
definition costs nothing measurable and buys an engine that can be checked
against a second implementation.
"""
from __future__ import annotations

import math
from typing import Sequence

from ..storage import Store
from ..text import STOPWORDS, tokenize
from .base import Candidate

K1 = 1.2
B = 0.75
SECTION_WEIGHT = 0.6


def _terms(text: str) -> list[str]:
    return [t for t in tokenize(text or "") if t not in STOPWORDS and len(t) > 1]


class Bm25KeywordRetriever:
    name = "keyword"

    def __init__(self, store: Store) -> None:
        self.store = store
        self._docs: list[dict] = []
        self._df: dict[str, int] = {}
        self._avg = 1.0
        self._build()

    def _build(self) -> None:
        rows = self.store.conn.execute(
            "SELECT id, text, section, source_id FROM chunks"
        ).fetchall()
        for row in rows:
            body = _terms(row["text"])
            section = _terms(row["section"])
            tf: dict[str, float] = {}
            for t in body:
                tf[t] = tf.get(t, 0.0) + 1.0
            for t in section:
                tf[t] = tf.get(t, 0.0) + SECTION_WEIGHT
            self._docs.append({
                "id": row["id"], "text": row["text"], "section": row["section"],
                "source_id": row["source_id"], "tf": tf,
                "len": len(body) + SECTION_WEIGHT * len(section),
            })
        for doc in self._docs:
            for t in doc["tf"]:
                self._df[t] = self._df.get(t, 0) + 1
        self._avg = (sum(d["len"] for d in self._docs) / len(self._docs)) if self._docs else 1.0

    def search(self, query: str, limit: int = 40) -> Sequence[Candidate]:
        terms = _terms(query)
        if not terms:
            return []
        n = len(self._docs)
        scored = []
        for doc in self._docs:
            score = 0.0
            for t in terms:
                f = doc["tf"].get(t)
                if not f:
                    continue
                df = self._df.get(t, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * ((f * (K1 + 1)) / (f + K1 * (1 - B + B * (doc["len"] / self._avg))))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [
            Candidate(chunk_id=d["id"], score=s, retriever=self.name, text=d["text"],
                      section=d["section"], source_id=d["source_id"], explain=f"bm25={s:.3f}")
            for s, d in scored[:limit]
        ]
