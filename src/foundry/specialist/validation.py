"""Post-generation validation: the part that does not trust the model.

Every answer is checked *after* generation, independently of whatever produced
it, against the evidence that was actually retrieved:

* **Citation resolution** -- does every ``[n]`` marker point at a passage that
  was really in the context?
* **Sentence grounding** -- does each claim-bearing sentence's content actually
  appear in the passages it cites?
* **Numeric fidelity** -- if the answer states ``80 °C``, does ``80 °C`` appear
  in the cited evidence? In a domain of thresholds and temperatures this is the
  cheapest effective hallucination trap there is.
* **Quotation fidelity** -- a double-quoted span claims to be verbatim, so it
  must appear verbatim.

This is why the system's usefulness does not depend on the model being good. A
weak model that abstains honestly is a better product than a strong model that
is confidently wrong (§25). It is also why an unsupported answer can be
*downgraded* to an abstention rather than shipped.

The checker is lexical, not semantic. It therefore misses paraphrase that
changes meaning while preserving vocabulary, and it can flag a correct
paraphrase that shares few words. Both error directions are measured by the
evaluation suite rather than assumed away, and the thresholds live in the
manifest so a knowledge area can tune them against its own metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..retrieval.base import Evidence
from ..text import content_terms, coverage, numbers_with_units, quoted_spans, sentences

_CITATION_RE = re.compile(r"\[(\d+)\]")
# Sentences that talk about the answer rather than making a claim about the
# world. They carry no factual burden, so grounding them is meaningless.
_META_RE = re.compile(
    r"\b(evidence (is|was) (not )?(sufficient|insufficient)|"
    r"this answer is|the sources? (do not|does not|available)|"
    r"i (cannot|can't|could not)|no (relevant )?evidence|"
    r"not enough (evidence|information)|out of scope|"
    r"assembled from the cited|without inference beyond|"
    r"based on the cited (passages|evidence|sources))\b",
    re.IGNORECASE,
)
_SECTION_HEADING_RE = re.compile(r"^(ANSWER|REASONING|SOURCES|LIMITATIONS|CONFIDENCE)\b", re.I)

# Vocabulary about the citation apparatus rather than about the world.
_APPARATUS_TERMS = frozenset({
    "cite", "cited", "citation", "citations", "passage", "passages", "source",
    "sources", "evidence", "stated", "state", "states", "according", "above",
    "below", "section", "quoted", "quote", "reference", "referenced", "answer",
    "question", "excerpt", "extract", "document", "text", "directly", "taken",
    "drawn", "provided", "given", "shown",
})
# Enough apparatus vocabulary, and no figure or quotation of its own, and the
# sentence is commentary on the answer rather than a claim in it.
_APPARATUS_SHARE = 0.5


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[int]
    supported: bool
    support: float
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "citations": self.citations,
            "support": round(self.support, 3),
            "reason": self.reason,
        }


@dataclass
class ValidationResult:
    sentences: list[SentenceCheck] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    cited_ranks: set[int] = field(default_factory=set)
    total_citations: int = 0

    @property
    def claim_sentences(self) -> list[SentenceCheck]:
        return self.sentences

    @property
    def unsupported(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if not s.supported]

    @property
    def grounded_ratio(self) -> float:
        if not self.sentences:
            return 1.0
        return sum(1 for s in self.sentences if s.supported) / len(self.sentences)

    @property
    def unsupported_ratio(self) -> float:
        return 1.0 - self.grounded_ratio

    @property
    def citation_validity(self) -> float:
        """Fraction of citation markers that resolve to real evidence."""
        if self.total_citations == 0:
            return 1.0
        return 1.0 - (len(self.invalid_citations) / self.total_citations)


def extract_citations(text: str) -> list[int]:
    return [int(m.group(1)) for m in _CITATION_RE.finditer(text)]


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub(" ", text)


def is_meta(sentence: str) -> bool:
    """True for sentences about the answer rather than about the world.

    Two ways a sentence qualifies. The first is a fixed list of phrasings that
    decline or describe the response. The second is statistical, and exists
    because models narrate their own citations in endlessly varied wording:
    "Stated directly in the cited FAA passage", "Taken from the evidence
    above", "This is drawn from source [2]". Those assert nothing about the
    world, but graded as claims they fail grounding and -- on a short answer,
    where one sentence is a large share of the total -- drag the whole answer
    over the unsupported threshold and get a correct answer withheld.

    A sentence is apparatus if most of its content words are *about* citing and
    it states no figure or quotation of its own. That last condition is what
    keeps it safe: "The source states the limit is 100 Wh" still carries a
    number, so it is still checked.
    """
    sentence = sentence.strip()
    if _META_RE.search(sentence) or _SECTION_HEADING_RE.match(sentence):
        return True
    # Citation markers are numerals; counting "[1]" as a stated figure would
    # make every cited sentence look like it carries one.
    bare = strip_citations(sentence)
    if numbers_with_units(bare) or quoted_spans(bare):
        return False
    terms = content_terms(bare)
    if len(terms) < 2:
        return False
    apparatus = sum(1 for t in terms if t in _APPARATUS_TERMS or f"{t}e" in _APPARATUS_TERMS)
    return apparatus / len(terms) >= _APPARATUS_SHARE


def validate_answer(
    answer_text: str,
    evidence: Sequence[Evidence],
    min_sentence_support: float = 0.34,
) -> ValidationResult:
    """Check an answer against the evidence it was given."""
    by_rank = {item.rank: item for item in evidence}
    result = ValidationResult()

    # Terms available across *all* evidence: used to decide whether an
    # uncited sentence is a claim at all, so that connective prose is not
    # reported as a hallucination.
    all_terms: set[str] = set()
    for item in evidence:
        all_terms |= content_terms(item.text)

    carried: list[int] = []  # citations carry to following uncited sentences
    for sentence in sentences(answer_text):
        stripped = sentence.strip()
        if not stripped or is_meta(stripped):
            continue

        citations = extract_citations(stripped)
        result.total_citations += len(citations)
        for citation in citations:
            if citation not in by_rank:
                result.invalid_citations.append(citation)
            else:
                result.cited_ranks.add(citation)

        bare = strip_citations(stripped).strip()
        terms = content_terms(bare)
        if len(terms) < 3:
            continue  # too short to carry a checkable claim

        effective = [c for c in citations if c in by_rank] or [c for c in carried if c in by_rank]
        if citations:
            carried = citations

        if not effective:
            # No citation, here or inherited. If it says nothing the evidence
            # says, it is an unsupported claim; if it is connective prose built
            # from evidence vocabulary, let it pass -- the cited sentences
            # around it carry the factual weight.
            overlap = coverage(terms, all_terms)
            supported = overlap >= min_sentence_support
            result.sentences.append(
                SentenceCheck(
                    sentence=bare,
                    citations=[],
                    supported=supported,
                    support=overlap,
                    reason="" if supported else "no citation and not present in any evidence",
                )
            )
            continue

        cited_terms: set[str] = set()
        cited_text = ""
        for rank in effective:
            item = by_rank[rank]
            cited_terms |= content_terms(item.text)
            cited_text += " " + item.text.lower()

        support = coverage(terms, cited_terms)
        reason = ""
        supported = support >= min_sentence_support
        if not supported:
            reason = f"only {support:.0%} of the sentence's content appears in the cited evidence"

        # Numeric fidelity: a stated number must appear in the cited evidence.
        stated_numbers = numbers_with_units(bare)
        if stated_numbers:
            evidence_numbers = numbers_with_units(cited_text)
            # Compare bare magnitudes too: evidence may carry the unit in an
            # adjacent token the regex attaches differently.
            bare_evidence = {n.rstrip("abcdefghijklmnopqrstuvwxyz°%") for n in evidence_numbers}
            missing = {
                n for n in stated_numbers
                if n not in evidence_numbers
                and n.rstrip("abcdefghijklmnopqrstuvwxyz°%") not in bare_evidence
            }
            if missing:
                supported = False
                reason = f"states {', '.join(sorted(missing))}, absent from the cited evidence"

        # Quotation fidelity: a quoted span claims to be verbatim.
        for span in quoted_spans(bare):
            if span not in cited_text:
                supported = False
                reason = f'quotes "{span}", which does not appear in the cited evidence'
                break

        result.sentences.append(
            SentenceCheck(
                sentence=bare,
                citations=effective,
                supported=supported,
                support=support,
                reason=reason,
            )
        )

    return result


def detect_contradictions(evidence: Sequence[Evidence], min_overlap: float = 0.5) -> list[dict]:
    """Flag evidence passages that discuss the same thing but disagree numerically.

    Feeds the ``contradiction_rate`` metric (§12) and the CONTRADICTIONS block
    in the response. Surfacing disagreement between sources is a feature: a
    knowledge system that silently picks one number is hiding the thing an
    expert most wants to see.
    """
    out: list[dict] = []
    items = list(evidence)
    for i, first in enumerate(items):
        first_terms = content_terms(first.text)
        first_numbers = numbers_with_units(first.text)
        if not first_numbers:
            continue
        for second in items[i + 1 :]:
            second_numbers = numbers_with_units(second.text)
            if not second_numbers or first.source_id == second.source_id:
                continue
            overlap = coverage(first_terms, content_terms(second.text))
            if overlap < min_overlap:
                continue
            # Same unit, different magnitude.
            def by_unit(values: set[str]) -> dict[str, set[str]]:
                grouped: dict[str, set[str]] = {}
                for value in values:
                    digits = value.rstrip("abcdefghijklmnopqrstuvwxyz°%")
                    unit = value[len(digits):]
                    if unit:
                        grouped.setdefault(unit, set()).add(digits)
                return grouped

            first_by_unit, second_by_unit = by_unit(first_numbers), by_unit(second_numbers)
            for unit in set(first_by_unit) & set(second_by_unit):
                if first_by_unit[unit] != second_by_unit[unit] and not (
                    first_by_unit[unit] & second_by_unit[unit]
                ):
                    out.append({
                        "description": (
                            f"{first.publisher or first.source_id} gives "
                            f"{'/'.join(sorted(first_by_unit[unit]))}{unit} while "
                            f"{second.publisher or second.source_id} gives "
                            f"{'/'.join(sorted(second_by_unit[unit]))}{unit}"
                        ),
                        "source_a": first.source_id,
                        "source_b": second.source_id,
                        "unit": unit,
                        "ranks": [first.rank, second.rank],
                    })
                    break
    return out
