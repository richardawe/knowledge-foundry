"""Retriever contract and the candidate record that flows through fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass
class Candidate:
    """One retrieved chunk, with the evidence trail needed to cite it."""

    chunk_id: str
    score: float
    retriever: str
    text: str = ""
    section: str = ""
    source_id: str = ""
    explain: str = ""

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "retriever": self.retriever,
            "section": self.section,
            "source_id": self.source_id,
            "explain": self.explain,
        }


@dataclass
class Evidence:
    """A fused, provenance-resolved passage as presented to the model and the user."""

    rank: int
    chunk_id: str
    text: str
    section: str
    score: float
    retrievers: list[str] = field(default_factory=list)
    source_id: str = ""
    source_title: str = ""
    publisher: str = ""
    uri: str = ""
    source_type: str = ""
    authority: int = 3
    published_at: str | None = None
    retrieved_at: str | None = None
    licence: str = ""
    document_title: str = ""

    def citation(self) -> str:
        """A human-readable citation line for the SOURCES block (§10)."""
        bits = [self.source_title or self.document_title, self.publisher]
        if self.published_at:
            bits.append(str(self.published_at))
        if self.section:
            bits.append(f"§ {self.section}")
        if self.uri and not self.uri.startswith("inline:"):
            bits.append(self.uri)
        return " — ".join(b for b in bits if b)

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "score": round(self.score, 6),
            "retrievers": self.retrievers,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "publisher": self.publisher,
            "uri": self.uri,
            "source_type": self.source_type,
            "authority": self.authority,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "licence": self.licence,
            "citation": self.citation(),
            "text": self.text,
        }


class Retriever(Protocol):
    name: str

    def search(self, query: str, limit: int) -> Sequence[Candidate]: ...
