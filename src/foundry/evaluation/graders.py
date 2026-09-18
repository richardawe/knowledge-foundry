"""Graders: how a question is judged (§11).

Deterministic first, and deliberately so. The full regression verdict can be
reached with **no model API at all**, because a CI gate that needs a paid API
is a CI gate that eventually gets disabled -- and an evaluation suite that only
runs when someone remembers to pay for it is not a product feature.

``judge`` exists for open-ended answers where no string check is honest, and it
is opt-in. Its results are reported separately so that nobody mistakes a
model's opinion for a measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..specialist.answer import Answer
from ..text import content_terms, numbers_with_units, stem


@dataclass
class GradeResult:
    kind: str
    passed: bool
    score: float
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "passed": self.passed,
            "score": round(self.score, 3),
            "detail": self.detail,
        }


GraderFn = Callable[[dict, Answer], GradeResult]
_REGISTRY: dict[str, GraderFn] = {}


def grader(name: str):
    def register(fn: GraderFn) -> GraderFn:
        _REGISTRY[name] = fn
        return fn

    return register


def _haystack(answer: Answer) -> str:
    return f"{answer.answer}\n{answer.reasoning}".lower()


# -- content ------------------------------------------------------------


@grader("must_include")
def must_include(spec: dict, answer: Answer) -> GradeResult:
    """At least one of ``any_of`` (or all of ``all_of``) must appear."""
    haystack = _haystack(answer)
    any_of = [str(v).lower() for v in spec.get("any_of", [])]
    all_of = [str(v).lower() for v in spec.get("all_of", [])]

    hits_any = [v for v in any_of if v in haystack]
    misses_all = [v for v in all_of if v not in haystack]
    passed = (not any_of or bool(hits_any)) and not misses_all
    detail = ""
    if any_of and not hits_any:
        detail = f"none of {any_of} present"
    if misses_all:
        detail = (detail + "; " if detail else "") + f"missing {misses_all}"
    return GradeResult("must_include", passed, 1.0 if passed else 0.0, detail)


@grader("must_not_include")
def must_not_include(spec: dict, answer: Answer) -> GradeResult:
    haystack = _haystack(answer)
    banned = [str(v).lower() for v in spec.get("none_of", [])]
    hits = [v for v in banned if v in haystack]
    return GradeResult(
        "must_not_include", not hits, 0.0 if hits else 1.0,
        f"contains {hits}" if hits else "",
    )


@grader("numeric_within")
def numeric_within(spec: dict, answer: Answer) -> GradeResult:
    """A stated figure must be near the expected value.

    Exact string matching is wrong for this domain: 80 °C and 80.0 °C are the
    same claim, and a source may report a range where another reports a point.
    """
    expected = float(spec["value"])
    tolerance = float(spec.get("tolerance", 0.0))
    unit = str(spec.get("unit", "")).lower()

    found = []
    for token in numbers_with_units(f"{answer.answer} {answer.reasoning}"):
        digits = token.rstrip("abcdefghijklmnopqrstuvwxyz°%")
        token_unit = token[len(digits):]
        if unit and token_unit and token_unit != unit:
            continue
        try:
            found.append(float(digits))
        except ValueError:
            continue
    ok = [v for v in found if abs(v - expected) <= tolerance]
    return GradeResult(
        "numeric_within", bool(ok), 1.0 if ok else 0.0,
        "" if ok else f"expected {expected}±{tolerance}{unit}, found {sorted(set(found))[:8]}",
    )


@grader("must_match")
def must_match(spec: dict, answer: Answer) -> GradeResult:
    pattern = re.compile(spec["pattern"], re.IGNORECASE | re.DOTALL)
    hit = bool(pattern.search(f"{answer.answer}\n{answer.reasoning}"))
    return GradeResult("must_match", hit, 1.0 if hit else 0.0,
                       "" if hit else f"no match for {spec['pattern']!r}")


# -- citations ----------------------------------------------------------


@grader("must_cite_source")
def must_cite_source(spec: dict, answer: Answer) -> GradeResult:
    """The answer must cite at least one of the named sources."""
    wanted = set(spec.get("source_ids", []))
    cited = answer.cited_source_ids
    hits = wanted & cited
    return GradeResult(
        "must_cite_source", bool(hits), 1.0 if hits else 0.0,
        "" if hits else f"cited {sorted(cited) or 'nothing'}, expected one of {sorted(wanted)}",
    )


@grader("must_cite_type")
def must_cite_type(spec: dict, answer: Answer) -> GradeResult:
    """Tests that the answer leans on the right *kind* of source.

    A regulatory question answered entirely from an encyclopaedia is wrong even
    when the words are right.
    """
    wanted = set(spec.get("source_types", []))
    cited_types = {s["source_type"] for s in answer.sources if s.get("cited")}
    hits = wanted & cited_types
    return GradeResult(
        "must_cite_type", bool(hits), 1.0 if hits else 0.0,
        "" if hits else f"cited types {sorted(cited_types) or 'none'}, expected {sorted(wanted)}",
    )


@grader("min_authority")
def min_authority(spec: dict, answer: Answer) -> GradeResult:
    floor = int(spec.get("value", 4))
    cited = [s["authority"] for s in answer.sources if s.get("cited")]
    best = max(cited) if cited else 0
    return GradeResult(
        "min_authority", best >= floor, 1.0 if best >= floor else 0.0,
        "" if best >= floor else f"best cited authority {best}, need {floor}",
    )


@grader("citations_resolve")
def citations_resolve(spec: dict, answer: Answer) -> GradeResult:
    floor = float(spec.get("value", 1.0))
    ok = answer.citation_validity >= floor
    return GradeResult(
        "citations_resolve", ok, answer.citation_validity,
        "" if ok else f"citation validity {answer.citation_validity:.2f} < {floor}",
    )


# -- behaviour ----------------------------------------------------------


@grader("must_abstain")
def must_abstain(spec: dict, answer: Answer) -> GradeResult:
    """Declining is the correct answer here. This is a pass condition, not a fail."""
    return GradeResult(
        "must_abstain", answer.abstained, 1.0 if answer.abstained else 0.0,
        "" if answer.abstained else f"answered with status={answer.status}",
    )


@grader("must_answer")
def must_answer(spec: dict, answer: Answer) -> GradeResult:
    ok = not answer.abstained
    return GradeResult("must_answer", ok, 1.0 if ok else 0.0,
                       "" if ok else f"abstained: {answer.status}")


@grader("must_flag_uncertainty")
def must_flag_uncertainty(spec: dict, answer: Answer) -> GradeResult:
    """For ambiguous questions: qualifying is success, answering flatly is not."""
    markers = spec.get("markers") or [
        "depends", "varies", "ambiguous", "unclear", "not established", "range",
        "conditions", "insufficient", "which", "could mean", "chemistry",
        "clarify", "specify",
    ]
    haystack = _haystack(answer)
    hit = answer.abstained or any(m in haystack for m in markers)
    return GradeResult("must_flag_uncertainty", hit, 1.0 if hit else 0.0,
                       "" if hit else "answered without qualification")


@grader("max_unsupported")
def max_unsupported(spec: dict, answer: Answer) -> GradeResult:
    ceiling = float(spec.get("value", 0.0))
    ratio = 1.0 - answer.grounded_ratio
    ok = ratio <= ceiling
    return GradeResult("max_unsupported", ok, 1.0 - ratio,
                       "" if ok else f"unsupported ratio {ratio:.2f} > {ceiling}")


@grader("must_flag_contradiction")
def must_flag_contradiction(spec: dict, answer: Answer) -> GradeResult:
    hit = bool(answer.contradictions) or "disagree" in _haystack(answer)
    return GradeResult("must_flag_contradiction", hit, 1.0 if hit else 0.0,
                       "" if hit else "no contradiction surfaced")


# -- retrieval ----------------------------------------------------------


@grader("retrieval_hit")
def retrieval_hit(spec: dict, answer: Answer) -> GradeResult:
    """Did the right source reach the evidence set at all?

    Separating this from answer correctness is what makes a failure
    diagnosable: a retrieval miss and a generation miss need different fixes.
    """
    wanted = set(spec.get("source_ids", []))
    retrieved = {s["source_id"] for s in answer.sources}
    hits = wanted & retrieved
    return GradeResult(
        "retrieval_hit", bool(hits), len(hits) / max(1, len(wanted)),
        "" if hits else f"retrieved {sorted(retrieved)}, expected one of {sorted(wanted)}",
    )


@grader("semantic_overlap")
def semantic_overlap(spec: dict, answer: Answer) -> GradeResult:
    """A soft check: does the answer share enough vocabulary with a reference?

    Used where an exact string is too brittle and a model judge is overkill.
    """
    reference = str(spec.get("reference", ""))
    floor = float(spec.get("value", 0.3))
    reference_terms = {stem(t) for t in content_terms(reference)}
    answer_terms = {stem(t) for t in content_terms(f"{answer.answer} {answer.reasoning}")}
    if not reference_terms:
        return GradeResult("semantic_overlap", True, 1.0, "empty reference")
    overlap = len(reference_terms & answer_terms) / len(reference_terms)
    return GradeResult("semantic_overlap", overlap >= floor, overlap,
                       "" if overlap >= floor else f"overlap {overlap:.2f} < {floor}")


# -- optional model judge -----------------------------------------------


@grader("judge")
def judge(spec: dict, answer: Answer) -> GradeResult:
    """LLM-as-judge for open-ended answers. Opt-in, and reported separately.

    Skipped (and scored as a pass) when no judge provider is configured, so a
    suite containing judge graders still produces a deterministic verdict in CI
    rather than failing for lack of an API key.
    """
    provider = spec.get("_provider")
    if provider is None:
        return GradeResult("judge", True, 1.0, "skipped: no judge provider configured")

    rubric = spec.get("rubric", "Is the answer correct and supported by its citations?")
    prompt = (
        "You are grading a specialist knowledge system. Reply with exactly one "
        "word, PASS or FAIL, then a short reason.\n\n"
        f"RUBRIC\n{rubric}\n\nQUESTION\n{answer.question}\n\n"
        f"ANSWER\n{answer.answer}\n\nREASONING\n{answer.reasoning}\n"
    )
    response = provider.complete("You are a strict, evidence-focused grader.", prompt)
    if not response.ok:
        return GradeResult("judge", True, 1.0, f"skipped: judge unavailable ({response.error})")
    verdict = response.text.strip().upper().startswith("PASS")
    return GradeResult("judge", verdict, 1.0 if verdict else 0.0, response.text.strip()[:200])


# -- dispatch -----------------------------------------------------------


def grade(spec: dict, answer: Answer) -> GradeResult:
    kind = spec.get("kind")
    fn = _REGISTRY.get(kind)
    if fn is None:
        raise ValueError(f"unknown grader {kind!r}; known: {sorted(_REGISTRY)}")
    return fn(spec, answer)


def available_graders() -> list[str]:
    return sorted(_REGISTRY)
