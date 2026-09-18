"""The specialist agent (§9).

    question -> scope check -> retrieve -> sufficiency gate -> generate
             -> validate -> respond

Three of those five steps can end the turn without an answer, and that is the
design. The brief's critical rule is that a system which cannot establish
sufficient evidence must say so, and the only way to make that reliable is to
give it several independent chances to notice.

The gates, in order of cost:

* **Scope** is free and runs before retrieval. A question about tax law never
  touches the index.
* **Sufficiency** runs after retrieval and before generation, so a question the
  corpus cannot support never reaches a model that might confabulate one.
* **Grounding** runs after generation and can *downgrade* a fluent answer to an
  abstention. This is the only gate that catches a model inventing detail that
  the evidence does not contain.
"""

from __future__ import annotations

import re
import time
from typing import Sequence

from ..llm import GenerationContext, provider_for
from ..manifest import Manifest
from ..retrieval import HybridRetriever, load_embedder
from ..retrieval.base import Evidence
from ..storage import Store
from ..text import content_terms, coverage
from . import prompts
from .answer import (
    ANSWERED,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_NONE,
    ERROR,
    INSUFFICIENT_EVIDENCE,
    OUT_OF_SCOPE,
    UNSUPPORTED,
    Answer,
)
from .validation import detect_contradictions, validate_answer

INSUFFICIENT_TEXT = (
    "The evidence in this knowledge area does not establish an answer to this "
    "question. Rather than guess, the system is declining to answer."
)
OUT_OF_SCOPE_TEMPLATE = (
    "This question is outside the declared scope of {name}. "
    "This knowledge area covers: {in_scope}."
)
UNSUPPORTED_TEXT = (
    "A draft answer was generated but failed automated grounding validation: too "
    "much of it could not be traced to the retrieved evidence. It has been "
    "withheld rather than published."
)


class Specialist:
    """One knowledge area's question-answering agent."""

    def __init__(
        self,
        manifest: Manifest,
        store: Store,
        provider=None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.manifest = manifest
        self.store = store
        self.provider = provider or provider_for(manifest)
        self.retriever = retriever or HybridRetriever(store, manifest, load_embedder(manifest))
        self._system_prompt = prompts.build_system_prompt(manifest.system_prompt())

    # -- gate 1: scope ---------------------------------------------------

    def scope_check(self, question: str) -> tuple[bool, float, float]:
        """Compare the question against the knowledge area's declared boundaries.

        Deliberately conservative: it refuses only when the question clearly
        matches a declared out-of-scope topic *and* matches it better than
        anything in scope. A borderline question is allowed through, where the
        sufficiency gate will handle it on evidence rather than on keywords.
        """
        scope = self.manifest.scope
        if not scope.out_of_scope:
            return True, 0.0, 0.0

        terms = content_terms(question)
        if not terms:
            return True, 0.0, 0.0

        def best(topics: Sequence[str]) -> float:
            return max((coverage(content_terms(t), terms) for t in topics), default=0.0)

        out_score = best(scope.out_of_scope)
        in_score = best(scope.in_scope) if scope.in_scope else 0.0
        in_scope = not (out_score >= 0.5 and out_score > in_score)
        return in_scope, in_score, out_score

    # -- gate 2: sufficiency ---------------------------------------------

    def sufficient(self, evidence: Sequence[Evidence], top_score: float) -> bool:
        cfg = self.manifest.specialist
        return len(evidence) >= cfg.min_chunks and top_score >= cfg.min_evidence_score

    # -- confidence ------------------------------------------------------

    def _confidence(
        self, evidence: Sequence[Evidence], validation, contradictions: list[dict]
    ) -> tuple[str, float]:
        """A composite score, reported with its components rather than alone.

        §12 warns against a single meaningless number. This one is only ever
        shown next to the citation-validity and grounding figures it is built
        from, so a reader can see why it says what it says.
        """
        if not evidence:
            return CONFIDENCE_NONE, 0.0

        cited = [e for e in evidence if e.rank in validation.cited_ranks] or list(evidence)
        authority = sum(e.authority for e in cited) / len(cited) / 5.0
        corroboration = min(1.0, len({e.source_id for e in cited}) / 2.0)
        agreement = sum(len(e.retrievers) for e in cited) / (len(cited) * 4.0)

        score = (
            0.35 * validation.grounded_ratio
            + 0.25 * validation.citation_validity
            + 0.20 * authority
            + 0.10 * corroboration
            + 0.10 * min(1.0, agreement * 2)
        )
        if contradictions:
            score *= 0.75  # disagreement between sources is a reason to hedge

        if score >= 0.75:
            return CONFIDENCE_HIGH, score
        if score >= 0.55:
            return CONFIDENCE_MEDIUM, score
        return CONFIDENCE_LOW, score

    # -- limitations -----------------------------------------------------

    def _limitations(self, evidence: Sequence[Evidence], validation, status: str) -> list[str]:
        """Standing limitations plus whatever is true of *this* answer."""
        limitations = list(self.manifest.governance.known_limitations)
        if self.manifest.governance.disclaimer:
            limitations.insert(0, self.manifest.governance.disclaimer)

        if status == ANSWERED:
            sources = {e.source_id for e in evidence if e.rank in validation.cited_ranks}
            if len(sources) == 1:
                limitations.append(
                    "This answer rests on a single source; it has not been corroborated."
                )
            if validation.unsupported:
                limitations.append(
                    f"{len(validation.unsupported)} sentence(s) could not be traced to the "
                    "cited evidence and are listed under unsupported claims."
                )
            undated = [e for e in evidence if e.rank in validation.cited_ranks and not e.published_at]
            if undated:
                limitations.append(
                    "Some cited sources carry no publication date, so their currency is unknown."
                )
        return limitations

    def _sources_block(self, evidence: Sequence[Evidence], cited_ranks: set[int]) -> list[dict]:
        out = []
        for item in evidence:
            record = item.as_dict()
            record["cited"] = item.rank in cited_ranks
            out.append(record)
        return out

    # -- main entry point ------------------------------------------------

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        started = time.monotonic()
        version = self.store.get_meta("current_version", "unversioned")
        evaluation_status = self.store.get_meta("evaluation_status", "unknown")

        def finish(answer: Answer) -> Answer:
            answer.elapsed_ms = int((time.monotonic() - started) * 1000)
            answer.knowledge_version = version
            answer.evaluation_status = evaluation_status
            return answer

        question = (question or "").strip()
        if not question:
            return finish(
                Answer(
                    question=question,
                    knowledge_area=self.manifest.id,
                    status=ERROR,
                    answer="No question was provided.",
                )
            )

        # Gate 1 -- scope.
        in_scope, in_score, out_score = self.scope_check(question)
        if not in_scope:
            return finish(
                Answer(
                    question=question,
                    knowledge_area=self.manifest.id,
                    status=OUT_OF_SCOPE,
                    answer=OUT_OF_SCOPE_TEMPLATE.format(
                        name=self.manifest.name,
                        in_scope="; ".join(self.manifest.scope.in_scope) or "see the manifest",
                    ),
                    confidence=CONFIDENCE_NONE,
                    limitations=self._limitations([], _EmptyValidation(), OUT_OF_SCOPE),
                    retrieval={"scope_in": round(in_score, 3), "scope_out": round(out_score, 3)},
                )
            )

        retrieval = self.retriever.retrieve(question, top_k=top_k)
        evidence = retrieval.evidence

        # Gate 2 -- sufficiency.
        if not self.sufficient(evidence, retrieval.top_score):
            return finish(
                Answer(
                    question=question,
                    knowledge_area=self.manifest.id,
                    status=INSUFFICIENT_EVIDENCE,
                    answer=INSUFFICIENT_TEXT,
                    confidence=CONFIDENCE_NONE,
                    sources=self._sources_block(evidence, set()),
                    limitations=self._limitations(evidence, _EmptyValidation(), INSUFFICIENT_EVIDENCE),
                    retrieval={
                        "top_score": round(retrieval.top_score, 4),
                        "evidence_count": len(evidence),
                        "threshold": self.manifest.specialist.min_evidence_score,
                        "per_retriever": retrieval.per_retriever,
                    },
                )
            )

        system = self._system_prompt
        user = prompts.build_user_prompt(question, evidence)
        response = self.provider.complete(
            system,
            user,
            context=GenerationContext(
                question=question, evidence=evidence, knowledge_area=self.manifest.id
            ),
            temperature=self.manifest.specialist.temperature,
            max_tokens=self.manifest.specialist.max_tokens,
        )
        if not response.ok:
            return finish(
                Answer(
                    question=question,
                    knowledge_area=self.manifest.id,
                    status=ERROR,
                    answer=f"The language model could not be reached: {response.error}",
                    sources=self._sources_block(evidence, set()),
                    llm=response.as_dict(),
                )
            )

        answer_text, reasoning = _split_sections(response.text)

        # Gate 3 -- grounding, which can withhold a fluent answer.
        validation = validate_answer(
            answer_text + ("\n" + reasoning if reasoning else ""),
            evidence,
            min_sentence_support=self.manifest.specialist.min_sentence_support,
        )
        contradictions = detect_contradictions(evidence)

        model_abstained = _looks_like_abstention(answer_text)
        too_unsupported = (
            validation.unsupported_ratio > self.manifest.specialist.max_unsupported_ratio
        )

        if model_abstained:
            status = INSUFFICIENT_EVIDENCE
            body = answer_text
        elif too_unsupported:
            status = UNSUPPORTED
            body = UNSUPPORTED_TEXT
        else:
            status = ANSWERED
            body = answer_text

        confidence, score = self._confidence(evidence, validation, contradictions)
        if status != ANSWERED:
            confidence, score = CONFIDENCE_NONE, 0.0

        return finish(
            Answer(
                question=question,
                knowledge_area=self.manifest.id,
                status=status,
                answer=body,
                reasoning=reasoning if status == ANSWERED else "",
                sources=self._sources_block(evidence, validation.cited_ranks),
                confidence=confidence,
                confidence_score=score,
                limitations=self._limitations(evidence, validation, status),
                contradictions=contradictions,
                unsupported_claims=[s.as_dict() for s in validation.unsupported],
                citation_validity=validation.citation_validity,
                grounded_ratio=validation.grounded_ratio,
                retrieval={
                    "top_score": round(retrieval.top_score, 4),
                    "evidence_count": len(evidence),
                    "per_retriever": retrieval.per_retriever,
                    "filters": retrieval.filters,
                },
                llm=response.as_dict(),
            )
        )


class _EmptyValidation:
    """Stand-in so the limitation and source helpers work on the gate paths."""

    cited_ranks: set[int] = set()
    unsupported: list = []
    grounded_ratio = 1.0
    citation_validity = 1.0
    unsupported_ratio = 0.0


_ABSTENTION_MARKERS = (
    "not sufficient to answer",
    "does not establish",
    "insufficient evidence",
    "no evidence",
    "cannot be answered",
    "evidence does not",
    "not enough evidence",
    "unable to answer",
)


def _looks_like_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


# Headings must be line-anchored. Matching the bare word would truncate any
# answer containing "answer" -- including, fatally, the abstention text
# "the evidence is not sufficient to answer this question".
_ANSWER_HEADING_RE = re.compile(r"^[ \t]*ANSWER[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE)
_REASONING_HEADING_RE = re.compile(
    r"^[ \t]*(REASONING|EXPLANATION)[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE
)


def _split_sections(text: str) -> tuple[str, str]:
    """Split a model response into its ANSWER and REASONING parts.

    Models do not reliably follow a template, so an unlabelled response is
    treated as all answer and no reasoning rather than being rejected.
    """
    body = text.strip()
    if not body:
        return "", ""

    heading = _ANSWER_HEADING_RE.search(body)
    if heading:
        body = body[heading.end() :].lstrip("\n")

    reasoning_heading = _REASONING_HEADING_RE.search(body)
    if reasoning_heading:
        return (
            body[: reasoning_heading.start()].strip(),
            body[reasoning_heading.end() :].strip(),
        )
    return body.strip(), ""
