"""The answer record: evidence-first, inspectable, machine-checkable (§10)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Status values. "answered" is the only one that asserts anything about the
# world; every other value is the system declining, and declining is a success
# state, not a failure (§9).
ANSWERED = "answered"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
OUT_OF_SCOPE = "out_of_scope"
UNSUPPORTED = "unsupported"          # generated, but failed grounding validation
IRRELEVANT = "irrelevant"            # generated and grounded, but not about what was asked
AMBIGUOUS = "ambiguous"              # under-specified; answering would be guessing
ERROR = "error"

ABSTENTION_STATUSES = frozenset(
    {INSUFFICIENT_EVIDENCE, OUT_OF_SCOPE, UNSUPPORTED, AMBIGUOUS, IRRELEVANT}
)

CONFIDENCE_NONE = "none"
CONFIDENCE_LOW = "low"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_HIGH = "high"


@dataclass
class Answer:
    """What the specialist returns, and what the API and UI render (§10, §19)."""

    question: str
    knowledge_area: str
    status: str
    answer: str
    reasoning: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence: str = CONFIDENCE_NONE
    confidence_score: float = 0.0
    limitations: list[str] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    unsupported_claims: list[dict] = field(default_factory=list)
    citation_validity: float = 1.0
    grounded_ratio: float = 1.0
    knowledge_version: str = "unversioned"
    evaluation_status: str = "unknown"
    retrieval: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    elapsed_ms: int = 0

    @property
    def abstained(self) -> bool:
        return self.status in ABSTENTION_STATUSES

    @property
    def cited_source_ids(self) -> set[str]:
        return {s["source_id"] for s in self.sources if s.get("cited")}

    def as_dict(self) -> dict:
        """The API response shape from §19, plus the evidence surface from §10."""
        return {
            "question": self.question,
            "knowledge_area": self.knowledge_area,
            "status": self.status,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "sources": self.sources,
            "confidence": self.confidence,
            "confidence_score": round(self.confidence_score, 3),
            "limitations": self.limitations,
            "contradictions": self.contradictions,
            "unsupported_claims": self.unsupported_claims,
            "citation_validity": round(self.citation_validity, 3),
            "grounded_ratio": round(self.grounded_ratio, 3),
            "knowledge_version": self.knowledge_version,
            "evaluation_status": self.evaluation_status,
            "retrieval": self.retrieval,
            "llm": self.llm,
            "elapsed_ms": self.elapsed_ms,
        }

    def render(self) -> str:
        """The §10 text layout, for the CLI and for anyone reading a transcript."""
        lines = [f"ANSWER\n{self.answer}"]
        if self.reasoning:
            lines.append(f"\nREASONING\n{self.reasoning}")
        if self.sources:
            lines.append("\nSOURCES")
            for source in self.sources:
                marker = "*" if source.get("cited") else " "
                lines.append(f" [{source['rank']}]{marker} {source['citation']}")
        lines.append(
            f"\nEVIDENCE CONFIDENCE\n {self.confidence} "
            f"(score {self.confidence_score:.2f}, "
            f"citations valid {self.citation_validity:.0%}, "
            f"claims grounded {self.grounded_ratio:.0%})"
        )
        if self.contradictions:
            lines.append("\nCONTRADICTIONS IN EVIDENCE")
            for item in self.contradictions:
                lines.append(f" - {item['description']}")
        if self.unsupported_claims:
            lines.append("\nUNSUPPORTED CLAIMS DETECTED")
            for claim in self.unsupported_claims:
                lines.append(f" - {claim['sentence']} ({claim['reason']})")
        if self.limitations:
            lines.append("\nLIMITATIONS")
            for limitation in self.limitations:
                lines.append(f" - {limitation}")
        return "\n".join(lines)
