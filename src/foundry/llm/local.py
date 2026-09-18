"""Keyless local providers.

``LocalExtractiveProvider`` is the default, and it is not a stub. It answers by
*selecting* sentences from the retrieved evidence and attaching their citations,
so it is incapable of stating anything the corpus does not contain. It abstains
when nothing scores well enough.

That has three consequences worth stating plainly:

1. **CI works with no API key and no cost.** The regression gate is only useful
   if it always runs; a gate that needs a paid API is a gate that gets disabled.
2. **It is a meaningful floor, not a placeholder.** It is a real extractive QA
   baseline. If a hosted model cannot beat it on the evaluation suite, that is
   information about the hosted model.
3. **It cannot hallucinate**, so any non-zero hallucination rate in a run using
   it indicates a bug in the grounding checker rather than a bad model -- which
   makes it a useful control for the evaluation harness itself.

Its weakness is real and expected: it does not synthesise across passages, and
its prose is stitched rather than written. That is precisely what a hosted model
is for, and what the evaluation suite measures.
"""

from __future__ import annotations

import math

from ..text import content_terms, numbers_with_units, sentences
from .base import GenerationContext, LLMResponse

ABSTENTION = (
    "The available evidence is not sufficient to answer this question."
)

# Question words that signal what kind of sentence is likely to be the answer.
_NUMERIC_CUES = ("what temperature", "how hot", "how many", "how much", "how long",
                 "what pressure", "at what", "how often", "what is the limit", "threshold")
_REQUIREMENT_CUES = ("must", "shall", "required", "permitted", "allowed", "mandatory",
                     "obligation", "compliant", "legal")


class LocalExtractiveProvider:
    """Compose an answer by selecting and citing evidence sentences."""

    name = "local_extractive"

    # 0.30 means a sentence must cover roughly a third of the question's
    # weighted content before it is eligible. Lower, and a single common
    # domain term ("battery", "widget") is enough to answer anything.
    def __init__(self, model: str = "extractive-v1", max_sentences: int = 5,
                 min_score: float = 0.30) -> None:
        self.model = model
        self.max_sentences = max_sentences
        self.min_score = min_score

    @staticmethod
    def _term_weights(question_terms: set[str], passages: list[set[str]]) -> dict[str, float]:
        """Weight question terms by how rare they are *within the evidence*.

        Without this, one ubiquitous term is enough to clear the threshold: in a
        battery knowledge base every passage contains "battery", so the question
        "what is the resale value of a battery in Lagos" would match anything.
        Terms that appear in most retrieved passages carry almost no signal;
        terms absent from all of them carry the most, and score zero, which is
        exactly the behaviour that produces an honest abstention.
        """
        total = max(1, len(passages))
        weights = {}
        for term in question_terms:
            df = sum(1 for passage in passages if term in passage)
            weights[term] = math.log(1.0 + total / (1.0 + df))
        return weights

    def _score_sentence(
        self,
        sentence: str,
        question_terms: set[str],
        question: str,
        weights: dict[str, float],
    ) -> float:
        sentence_terms = content_terms(sentence)
        if not sentence_terms:
            return 0.0
        matched = question_terms & sentence_terms
        if not matched:
            return 0.0
        total_weight = sum(weights.get(t, 1.0) for t in question_terms)
        if total_weight <= 0:
            return 0.0
        score = sum(weights.get(t, 1.0) for t in matched) / total_weight
        # Softly length-normalised, so a long passage cannot win purely by
        # containing more words.
        score *= 1.0 / (1.0 + 0.003 * max(0, len(sentence) - 160))

        lowered = question.lower()
        if any(cue in lowered for cue in _NUMERIC_CUES) and numbers_with_units(sentence):
            score *= 1.6
        if any(cue in lowered for cue in _REQUIREMENT_CUES) and any(
            word in sentence.lower() for word in (" shall ", " must ", "required", "prohibited")
        ):
            score *= 1.4
        return score

    def complete(
        self,
        system: str,
        user: str,
        *,
        context: GenerationContext | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        if context is None or not context.evidence:
            return LLMResponse(text=ABSTENTION, provider=self.name, model=self.model)

        question_terms = content_terms(context.question)
        passages = [content_terms(item.text) for item in context.evidence]
        weights = self._term_weights(question_terms, passages)

        scored: list[tuple[float, int, int, str]] = []
        for item in context.evidence:
            for position, sentence in enumerate(sentences(item.text)):
                sentence = sentence.strip()
                if len(sentence) < 25:
                    continue
                score = self._score_sentence(sentence, question_terms, context.question, weights)
                if score >= self.min_score:
                    scored.append((score, item.rank, position, sentence))

        if not scored:
            return LLMResponse(text=ABSTENTION, provider=self.name, model=self.model)

        scored.sort(key=lambda t: (-t[0], t[1], t[2]))

        # Cap per-source contributions so one long passage cannot monopolise the
        # answer; a multi-document question should read from several sources.
        selected: list[tuple[int, int, str]] = []
        per_rank: dict[int, int] = {}
        seen: set[str] = set()
        for _score, rank, position, sentence in scored:
            key = sentence.lower()
            if key in seen:
                continue
            if per_rank.get(rank, 0) >= 2 and len(selected) >= 2:
                continue
            seen.add(key)
            per_rank[rank] = per_rank.get(rank, 0) + 1
            selected.append((rank, position, sentence))
            if len(selected) >= self.max_sentences:
                break

        # Present in evidence order: the highest-ranked evidence reads first.
        selected.sort(key=lambda t: (t[0], t[1]))
        body = " ".join(f"{sentence} [{rank}]" for rank, _position, sentence in selected)

        cited = sorted({rank for rank, _, _ in selected})
        reasoning = (
            "This answer is assembled from the cited passages "
            f"({', '.join(f'[{r}]' for r in cited)}) without inference beyond them."
        )
        text = f"ANSWER\n{body}\n\nREASONING\n{reasoning}"
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=len(user) // 4,
            completion_tokens=len(text) // 4,
            stop_reason="end_turn",
        )


class ScriptedProvider:
    """Returns pre-set text. For testing the validator, not for production.

    The grounding checker has to be tested against answers a real model might
    produce -- fabricated numbers, citations to passages that say something
    else. This provider makes those cases reproducible.
    """

    name = "scripted"

    def __init__(self, responses: list[str] | str | None = None, model: str = "scripted") -> None:
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses or [])
        self.model = model
        self.calls: list[tuple[str, str]] = []
        self._index = 0

    def complete(self, system: str, user: str, *, context=None, temperature: float = 0.0,
                 max_tokens: int = 1200) -> LLMResponse:
        self.calls.append((system, user))
        if not self.responses:
            return LLMResponse(text=ABSTENTION, provider=self.name, model=self.model)
        text = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return LLMResponse(text=text, provider=self.name, model=self.model)
