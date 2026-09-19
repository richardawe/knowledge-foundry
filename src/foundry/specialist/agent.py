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

import math
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..llm import GenerationContext, LLMResponse, provider_for
from ..manifest import Manifest
from ..retrieval import HybridRetriever, load_embedder
from ..retrieval.base import Evidence
from ..storage import Store
from ..text import content_terms, coverage
from . import prompts
from .answer import (
    AMBIGUOUS,
    ANSWERED,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_NONE,
    ERROR,
    INSUFFICIENT_EVIDENCE,
    IRRELEVANT,
    OUT_OF_SCOPE,
    UNSUPPORTED,
    Answer,
)
from .validation import detect_contradictions, validate_answer

@dataclass
class Prepared:
    """Everything the model needs, and everything needed to judge its reply.

    Serialisable on purpose: the prompt can be handed to a model somewhere
    else entirely, and the evidence travels with it so the reply is validated
    against exactly the passages it was shown.
    """

    question: str
    knowledge_area: str
    answer: Answer | None = None            # set when a gate ended the turn
    system: str = ""
    user: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 1200

    @property
    def needs_model(self) -> bool:
        return self.answer is None


INSUFFICIENT_TEXT = (
    "The evidence in this knowledge area does not establish an answer to this "
    "question. Rather than guess, the system is declining to answer."
)
OUT_OF_SCOPE_TEMPLATE = (
    "This question is outside the declared scope of {name}. "
    "This knowledge area covers: {in_scope}."
)
IRRELEVANT_TEMPLATE = (
    "A draft answer was generated, correctly cited and fully traceable to the "
    "evidence -- and it did not answer the question. It never engaged with: "
    "{terms}. An answer about the right topic but the wrong subject is harder "
    "to catch than a wrong one and no more useful, so it has been withheld."
)
NO_SUBJECT_TEMPLATE = (
    "This knowledge area has nothing on: {terms}. Passages were retrieved that "
    "share the question's general vocabulary, but none of them mentions what "
    "was actually asked about, so any answer built from them would be about "
    "something else. Rather than return that, the system is declining and "
    "naming the gap."
)
UNSUPPORTED_TEXT = (
    "A draft answer was generated but failed automated grounding validation: too "
    "much of it could not be traced to the retrieved evidence. It has been "
    "withheld rather than published."
)
AMBIGUOUS_TEMPLATE = (
    "This question is under-specified, so any single figure would be misleading: "
    "{reason}. In this domain the answer depends on cell chemistry, cell format, "
    "state of charge, age and the test conditions. Please specify what you mean, "
    "and the answer can be given against evidence rather than guessed."
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

    # -- gate 3: specificity ---------------------------------------------

    def specific_enough(self, question: str, evidence: Sequence[Evidence]) -> tuple[bool, str]:
        """Is the question pinned down enough that a single answer is honest?

        "Is this battery safe?" and "What is the limit?" have no answer, only a
        clarifying question. Answering them with the first plausible passage is
        a quiet way of being wrong, and §13 names deliberately ambiguous
        questions as a category the system must handle rather than absorb.

        The test is deliberately narrow: does the question name at least three
        content terms? Fewer than that and it cannot have specified a
        chemistry, a format or a condition.

        An earlier version also flagged questions whose retrieved evidence was
        topically scattered. That signal was measuring retrieval diversity, not
        question ambiguity, and it refused "what is the main function of a
        battery separator?" -- a perfectly specific question whose subject
        appears in many contexts. It was removed rather than tuned, because a
        gate that refuses good questions costs more than one that misses bad
        ones.

        So this is honestly incomplete. It catches "what is the limit?" and
        misses "how long does a battery fire burn?", which is equally
        under-specified but lexically richer. The evaluation suite keeps score
        of what it misses instead of pretending otherwise.
        """
        terms = content_terms(question)
        if len(terms) < 3:
            return False, f"it names only {len(terms)} specific term(s)"
        return True, ""

    # -- gate 4: subject coverage ----------------------------------------

    def covers_the_subject(
        self, question: str, evidence: Sequence[Evidence]
    ) -> tuple[bool, list[str], float]:
        """Does the retrieved evidence mention what the question is *about*?

        Every other gate asks whether there is enough evidence, or whether what
        was said is supported by it. None of them asks the question a reader
        actually cares about: is this an answer to what I asked?

        The failure this exists for, from a real submission: asked for a
        specific vehicle's pack capacity, the system returned regulatory
        definitions of "rated capacity" and "endurance in cycles" -- correctly
        cited, fully grounded, scored high confidence, and about nothing the
        person had asked for. The corpus had never heard of the vehicle. It had
        heard plenty about capacity, and that was enough to clear every bar.

        The test: weight each question term by how rare it is in the retrieved
        passages, the same weighting the extractive provider already uses to
        pick sentences. A term absent from every passage carries the most
        weight, because it is the part of the question the corpus cannot speak
        to. When the absent terms carry most of the question's weight, the
        subject itself is missing and no honest answer exists -- whatever the
        generic vocabulary around it scores.

        Returning the missing terms rather than a bare verdict is the point.
        "I have nothing on: tesla" tells the reader why, tells them how to
        rephrase if it was wording, and names exactly what the corpus would
        need to acquire if it was not.
        """
        terms = content_terms(question)
        if not terms or not evidence:
            return True, [], 0.0

        passages = [content_terms(item.text) for item in evidence]
        total = max(1, len(passages))
        weights = {
            term: math.log(1.0 + total / (1.0 + sum(1 for p in passages if term in p)))
            for term in terms
        }
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return True, [], 0.0

        missing = sorted(t for t in terms if not any(t in p for p in passages))
        share = sum(weights[t] for t in missing) / total_weight
        return share < self.manifest.specialist.max_missing_subject_weight, missing, share

    # -- gate 5: responsiveness ------------------------------------------

    def answers_the_question(
        self, question: str, answer_text: str, evidence: Sequence[Evidence]
    ) -> tuple[bool, list[str], float]:
        """Does the answer address what was asked, or merely the topic?

        Gate 4 asks whether the corpus holds the subject. This asks the harder
        and more common question: it held it, and the answer talked about
        something else anyway.

        The case this exists for is issue #1. Asked for a named vehicle's pack
        capacity, the system returned regulatory definitions of "rated capacity"
        and "endurance in cycles". The corpus was not the problem -- it holds
        thirteen passages naming that vehicle. Retrieval was not the problem
        either. The answer simply never mentioned the vehicle, and every
        existing check was satisfied: the sentences were in the cited passages,
        the citations resolved, and confidence came out high.

        So the measure is the answer, not the evidence: weight each question
        term by how rare it is across the retrieved passages, and ask how much
        of that weight the answer actually engages with. A term the corpus uses
        everywhere ("battery") carries almost none; the specific thing being
        asked about carries most. An answer built from generic vocabulary
        scores near zero however fluent and however well cited it is.
        """
        terms = content_terms(question)
        answer_terms = content_terms(answer_text)
        if not terms or not answer_terms:
            return True, [], 0.0

        passages = [content_terms(item.text) for item in evidence] or [answer_terms]
        total = max(1, len(passages))
        weights = {
            term: math.log(1.0 + total / (1.0 + sum(1 for p in passages if term in p)))
            for term in terms
        }
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return True, [], 0.0

        unaddressed = sorted(t for t in terms if t not in answer_terms)
        addressed = sum(weights[t] for t in terms if t in answer_terms) / total_weight
        threshold = self.manifest.specialist.min_question_coverage
        return addressed >= threshold, unaddressed, addressed

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

    # -- the two halves of answering -------------------------------------
    #
    # Everything except the model call is deterministic Python over the
    # knowledge area: retrieval, the gates, prompt assembly, then validation,
    # grounding and scoring. Only one step in the middle needs a model.
    #
    # `prepare` runs the first half and either finishes the turn on a gate or
    # hands back the prompt. `judge` runs the second half over whatever the
    # model returned. `ask` is the two joined together, which is what a live
    # server does.
    #
    # Splitting them is what lets the deterministic half run where the corpus
    # is and the model call run where the model is, without either pretending
    # to be the other.

    def _finish(self, answer: Answer, started: float | None = None) -> Answer:
        if started is not None:
            answer.elapsed_ms = int((time.monotonic() - started) * 1000)
        answer.knowledge_version = self.store.get_meta("current_version", "unversioned")
        answer.evaluation_status = self.store.get_meta("evaluation_status", "unknown")
        return answer

    def prepare(self, question: str, top_k: int | None = None) -> "Prepared":
        """Retrieve, run the gates, and build the prompt. No model is called."""
        question = (question or "").strip()
        area = self.manifest.id

        if not question:
            return Prepared(question=question, knowledge_area=area, answer=self._finish(
                Answer(question=question, knowledge_area=area, status=ERROR,
                       answer="No question was provided.")))

        # Gate 1 -- scope.
        in_scope, in_score, out_score = self.scope_check(question)
        if not in_scope:
            return Prepared(question=question, knowledge_area=area, answer=self._finish(
                Answer(
                    question=question, knowledge_area=area, status=OUT_OF_SCOPE,
                    answer=OUT_OF_SCOPE_TEMPLATE.format(
                        name=self.manifest.name,
                        in_scope="; ".join(self.manifest.scope.in_scope) or "see the manifest",
                    ),
                    confidence=CONFIDENCE_NONE,
                    limitations=self._limitations([], _EmptyValidation(), OUT_OF_SCOPE),
                    retrieval={"scope_in": round(in_score, 3), "scope_out": round(out_score, 3)},
                )))

        retrieval = self.retriever.retrieve(question, top_k=top_k)
        evidence = retrieval.evidence

        # Gate 2 -- specificity. Before sufficiency, so an under-specified
        # question is named as such rather than reported as a gap in the
        # corpus, which would send someone looking for sources that would not
        # have helped.
        specific, reason = self.specific_enough(question, evidence)
        if not specific:
            return Prepared(question=question, knowledge_area=area, evidence=evidence,
                            answer=self._finish(Answer(
                                question=question, knowledge_area=area, status=AMBIGUOUS,
                                answer=AMBIGUOUS_TEMPLATE.format(reason=reason),
                                confidence=CONFIDENCE_NONE,
                                sources=self._sources_block(evidence, set()),
                                limitations=self._limitations(evidence, _EmptyValidation(), AMBIGUOUS),
                                retrieval={
                                    "top_score": round(retrieval.top_score, 4),
                                    "evidence_count": len(evidence),
                                    "ambiguity_reason": reason,
                                },
                            )))

        # Gate 3 -- sufficiency.
        if not self.sufficient(evidence, retrieval.top_score):
            return Prepared(question=question, knowledge_area=area, evidence=evidence,
                            answer=self._finish(Answer(
                                question=question, knowledge_area=area,
                                status=INSUFFICIENT_EVIDENCE, answer=INSUFFICIENT_TEXT,
                                confidence=CONFIDENCE_NONE,
                                sources=self._sources_block(evidence, set()),
                                limitations=self._limitations(
                                    evidence, _EmptyValidation(), INSUFFICIENT_EVIDENCE),
                                retrieval={
                                    "top_score": round(retrieval.top_score, 4),
                                    "evidence_count": len(evidence),
                                    "threshold": self.manifest.specialist.min_evidence_score,
                                    "per_retriever": retrieval.per_retriever,
                                },
                            )))

        # Gate 4 -- subject coverage. Last before generation, because it is the
        # only one that can tell "the corpus cannot answer this" apart from
        # "the corpus has plenty to say nearby".
        covered, missing, missing_share = self.covers_the_subject(question, evidence)
        if not covered:
            return Prepared(question=question, knowledge_area=area, evidence=evidence,
                            answer=self._finish(Answer(
                                question=question, knowledge_area=area,
                                status=INSUFFICIENT_EVIDENCE,
                                answer=NO_SUBJECT_TEMPLATE.format(
                                    terms=", ".join(missing) or "the subject of the question"),
                                confidence=CONFIDENCE_NONE,
                                sources=self._sources_block(evidence, set()),
                                limitations=self._limitations(
                                    evidence, _EmptyValidation(), INSUFFICIENT_EVIDENCE),
                                retrieval={
                                    "top_score": round(retrieval.top_score, 4),
                                    "evidence_count": len(evidence),
                                    "missing_terms": missing,
                                    "missing_weight": round(missing_share, 3),
                                    "threshold": (
                                        self.manifest.specialist.max_missing_subject_weight),
                                },
                            )))

        return Prepared(
            question=question,
            knowledge_area=area,
            system=self._system_prompt,
            user=prompts.build_user_prompt(question, evidence),
            evidence=list(evidence),
            retrieval={
                "top_score": round(retrieval.top_score, 4),
                "evidence_count": len(evidence),
                "per_retriever": retrieval.per_retriever,
                "filters": retrieval.filters,
            },
            temperature=self.manifest.specialist.temperature,
            max_tokens=self.manifest.specialist.max_tokens,
        )

    def judge(self, prepared: "Prepared", response: LLMResponse,
              started: float | None = None) -> Answer:
        """Validate and score a model's reply against the evidence it was given.

        This is the half that does not trust the model, and it runs wherever
        the corpus is -- never alongside the model, so a broken or hostile
        generation step cannot also mark its own work.
        """
        area = prepared.knowledge_area
        evidence = prepared.evidence

        if not response.ok:
            return self._finish(Answer(
                question=prepared.question, knowledge_area=area, status=ERROR,
                answer=f"The language model could not be reached: {response.error}",
                sources=self._sources_block(evidence, set()),
                llm=response.as_dict(),
            ), started)

        answer_text, reasoning = _split_sections(response.text)

        # Gate 6 -- grounding, which can withhold a fluent answer.
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

        # Gate 5 -- responsiveness. Ordered after grounding because an
        # unsupported answer is the worse fault of the two and should be named
        # as such; an irrelevant one that is otherwise well cited is what this
        # catches, and nothing else does.
        responsive, unaddressed, addressed = self.answers_the_question(
            prepared.question, answer_text, evidence
        )

        if model_abstained:
            status, body = INSUFFICIENT_EVIDENCE, answer_text
        elif too_unsupported:
            status, body = UNSUPPORTED, UNSUPPORTED_TEXT
        elif not responsive:
            status, body = IRRELEVANT, IRRELEVANT_TEMPLATE.format(
                terms=", ".join(unaddressed) or "what was asked about"
            )
        else:
            status, body = ANSWERED, answer_text

        confidence, score = self._confidence(evidence, validation, contradictions)
        if status != ANSWERED:
            confidence, score = CONFIDENCE_NONE, 0.0

        return self._finish(Answer(
            question=prepared.question,
            knowledge_area=area,
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
            retrieval=prepared.retrieval,
            llm=response.as_dict(),
        ), started)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """Prepare, call the model, judge. What a live server does."""
        started = time.monotonic()
        prepared = self.prepare(question, top_k=top_k)
        if prepared.answer is not None:
            prepared.answer.elapsed_ms = int((time.monotonic() - started) * 1000)
            return prepared.answer

        response = self.provider.complete(
            prepared.system,
            prepared.user,
            context=GenerationContext(
                question=prepared.question,
                evidence=prepared.evidence,
                knowledge_area=prepared.knowledge_area,
            ),
            temperature=prepared.temperature,
            max_tokens=prepared.max_tokens,
        )
        return self.judge(prepared, response, started=started)


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
