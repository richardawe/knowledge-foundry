"""The specialist agent and its validator (§9, §10).

The validator tests matter most. They are written against answers a real model
might plausibly produce -- fabricated numbers, citations to passages that say
something else, fluent prose with no evidence behind it -- because that is the
failure mode the whole project exists to catch.
"""

from __future__ import annotations

import pytest

from foundry.llm import ScriptedProvider
from foundry.retrieval.base import Evidence
from foundry.specialist import (
    ANSWERED,
    INSUFFICIENT_EVIDENCE,
    OUT_OF_SCOPE,
    UNSUPPORTED,
    Specialist,
    detect_contradictions,
    validate_answer,
)
from foundry.specialist.validation import extract_citations, is_meta


def _evidence(**overrides) -> Evidence:
    base = dict(
        rank=1,
        chunk_id="c1",
        text="Widget overheating begins when the core temperature exceeds 80 °C.",
        section="1",
        score=1.0,
        source_id="src-a",
        source_title="Handbook",
        publisher="Widget Standards Body",
        uri="https://example.test/a",
        source_type="standard",
        authority=5,
        published_at="2024-06-01",
        licence="test",
    )
    base.update(overrides)
    return Evidence(**base)


# -- validation ---------------------------------------------------------


def test_grounded_cited_sentence_passes():
    result = validate_answer(
        "Widget overheating begins when the core temperature exceeds 80 °C. [1]",
        [_evidence()],
    )
    assert result.unsupported == []
    assert result.grounded_ratio == 1.0
    assert result.citation_validity == 1.0


def test_fabricated_number_is_caught_even_when_the_sentence_reads_correctly():
    """The classic failure mode: right shape, wrong number, real citation."""
    result = validate_answer(
        "Widget overheating begins when the core temperature exceeds 65 °C. [1]",
        [_evidence()],
    )
    assert len(result.unsupported) == 1
    assert "65" in result.unsupported[0].reason


def test_citation_to_a_passage_that_does_not_exist_is_invalid():
    result = validate_answer("Widgets vent at 12 bar. [7]", [_evidence()])
    assert result.invalid_citations == [7]
    assert result.citation_validity == 0.0


def test_citation_validity_is_a_fraction_not_a_boolean():
    result = validate_answer(
        "Overheating begins above 80 °C. [1] Venting follows at higher pressure. [9]",
        [_evidence()],
    )
    assert result.total_citations == 2
    assert result.citation_validity == 0.5


def test_uncited_claim_not_present_in_any_evidence_is_unsupported():
    result = validate_answer(
        "Widget overheating begins above 80 °C. [1] "
        "Lithium batteries in Norway are subject to a separate maritime tariff regime.",
        [_evidence()],
    )
    unsupported = [s for s in result.unsupported]
    assert len(unsupported) == 1
    assert "maritime" in unsupported[0].sentence


def test_fabricated_quotation_is_caught():
    result = validate_answer(
        'The standard states "widgets shall never be installed indoors". [1]',
        [_evidence()],
    )
    assert result.unsupported
    assert "quotes" in result.unsupported[0].reason


def test_citation_carries_to_an_immediately_following_uncited_sentence():
    """Models routinely cite once per paragraph; that must not read as fabrication."""
    result = validate_answer(
        "Widget overheating begins when the core temperature exceeds 80 °C. [1] "
        "The core temperature is therefore the quantity that matters for overheating.",
        [_evidence()],
    )
    assert result.grounded_ratio == 1.0


def test_meta_sentences_are_not_graded_as_claims():
    assert is_meta("The available evidence is not sufficient to answer this question.")
    assert is_meta("This answer is assembled from the cited passages.")
    assert not is_meta("Widgets vent at 12 bar.")

    result = validate_answer(
        "The available evidence is not sufficient to answer this question.", [_evidence()]
    )
    assert result.sentences == []
    assert result.grounded_ratio == 1.0


def test_citations_are_extracted_in_order():
    assert extract_citations("a [2] b [1] c [2]") == [2, 1, 2]


def test_no_evidence_means_every_claim_is_unsupported():
    result = validate_answer("Widgets vent at 12 bar and release carbon monoxide.", [])
    assert result.unsupported


# -- contradictions -----------------------------------------------------


def test_disagreeing_sources_are_surfaced_not_silently_reconciled():
    first = _evidence(rank=1, source_id="src-a", publisher="Body A")
    second = _evidence(
        rank=2,
        chunk_id="c2",
        source_id="src-b",
        publisher="Body B",
        text="Widget overheating begins when the core temperature exceeds 95 °C.",
    )
    contradictions = detect_contradictions([first, second])

    assert contradictions
    assert "80" in contradictions[0]["description"]
    assert "95" in contradictions[0]["description"]


def test_agreeing_sources_are_not_flagged():
    first = _evidence(rank=1, source_id="src-a")
    second = _evidence(rank=2, chunk_id="c2", source_id="src-b", publisher="Body B")
    assert detect_contradictions([first, second]) == []


def test_unrelated_passages_with_different_numbers_are_not_contradictions():
    first = _evidence(text="Widget overheating begins above 80 °C.")
    second = _evidence(
        rank=2, chunk_id="c2", source_id="src-b",
        text="Inspection records shall be retained for 5 years by the operator.",
    )
    assert detect_contradictions([first, second]) == []


# -- the agent ----------------------------------------------------------


def test_answers_an_in_scope_question_with_resolvable_citations(built):
    manifest, store = built
    answer = Specialist(manifest, store).ask("At what core temperature does overheating begin?")

    assert answer.status == ANSWERED
    assert "80" in answer.answer
    assert answer.citation_validity == 1.0
    assert answer.cited_source_ids
    for source in answer.sources:
        assert source["uri"] and source["publisher"] and source["citation"]


def test_out_of_scope_question_is_refused_before_retrieval(built):
    """The cheap first line: a question matching a declared out-of-scope topic."""
    manifest, store = built
    answer = Specialist(manifest, store).ask("Can you give me a medical diagnosis?")

    assert answer.status == OUT_OF_SCOPE
    assert answer.sources == []
    assert "outside the declared scope" in answer.answer


def test_scope_gate_is_lexical_so_paraphrases_fall_through_to_sufficiency(built):
    """An honest test of a known limitation.

    The scope gate only matches the vocabulary a knowledge area declares, so
    "how do I treat a bacterial infection" does not look like "medical
    diagnosis" to it. The system must still refuse -- via the evidence gate
    rather than the keyword gate. Refusing for the right reason matters less
    than refusing.
    """
    manifest, store = built
    specialist = Specialist(manifest, store)

    in_scope, _in_score, _out_score = specialist.scope_check(
        "How should I treat a bacterial infection?"
    )
    assert in_scope  # the keyword gate does not catch it

    answer = specialist.ask("How should I treat a bacterial infection?")
    assert answer.abstained  # but the system still declines


def test_scope_gate_does_not_refuse_borderline_in_scope_questions(built):
    manifest, store = built
    specialist = Specialist(manifest, store)
    for question in (
        "what pressure causes venting",
        "how often must widgets be inspected",
        "what happened at Northgate Depot",
    ):
        in_scope, _in_score, _out_score = specialist.scope_check(question)
        assert in_scope, question


def test_unanswerable_question_produces_an_abstention_not_a_guess(built):
    manifest, store = built
    answer = Specialist(manifest, store).ask(
        "What will a widget cost in Lagos in 2031 after the tariff changes?"
    )

    assert answer.abstained
    assert answer.confidence == "none"


def test_a_hallucinating_model_is_caught_and_its_answer_withheld(built):
    """The gate that matters: fluent, fully-cited, entirely invented."""
    manifest, store = built
    liar = ScriptedProvider(
        "ANSWER\nWidget overheating begins at 210 °C and is governed by the Widget "
        "Directive 1994/12/EC. [1] Operators in Chile must file form WX-9 within "
        "48 hours of any venting event. [1]"
    )
    answer = Specialist(manifest, store, provider=liar).ask(
        "At what core temperature does overheating begin?"
    )

    assert answer.status == UNSUPPORTED
    assert "210" not in answer.answer
    assert answer.unsupported_claims
    assert answer.confidence == "none"


def test_a_partially_supported_answer_is_published_with_its_flaws_listed(built):
    """Below the downgrade threshold, publish -- but surface what failed."""
    manifest, store = built
    manifest.specialist.max_unsupported_ratio = 0.6
    provider = ScriptedProvider(
        "ANSWER\nWidget overheating begins when the core temperature exceeds 80 °C. [1] "
        "A separate Chilean filing requirement applies to every venting event worldwide."
    )
    answer = Specialist(manifest, store, provider=provider).ask(
        "At what core temperature does overheating begin?"
    )

    assert answer.status == ANSWERED
    assert answer.unsupported_claims
    assert any("could not be traced" in limitation for limitation in answer.limitations)


def test_model_abstention_is_respected_rather_than_overridden(built):
    manifest, store = built
    provider = ScriptedProvider("ANSWER\nThe evidence does not establish an answer to this.")
    answer = Specialist(manifest, store, provider=provider).ask("what pressure causes venting")

    assert answer.status == INSUFFICIENT_EVIDENCE


def test_llm_failure_is_reported_not_disguised_as_an_answer(built):
    manifest, store = built

    class Broken:
        name, model = "broken", "broken"

        def complete(self, system, user, **kwargs):
            from foundry.llm import LLMResponse

            return LLMResponse(text="", provider="broken", error="connection refused")

    answer = Specialist(manifest, store, provider=Broken()).ask("what pressure causes venting")
    assert answer.status == "error"
    assert "connection refused" in answer.answer


def test_empty_question_is_rejected(built):
    manifest, store = built
    assert Specialist(manifest, store).ask("   ").status == "error"


def test_answer_always_carries_the_governance_disclaimer(built):
    """A safety-critical domain must never ship an answer without its caveat."""
    manifest, store = built
    specialist = Specialist(manifest, store)
    for question in (
        "At what core temperature does overheating begin?",
        "How should I treat a bacterial infection?",
        "What will a widget cost in Lagos in 2031?",
    ):
        answer = specialist.ask(question)
        assert manifest.governance.disclaimer in answer.limitations, question


def test_single_source_answers_declare_that_they_are_uncorroborated(built):
    manifest, store = built
    answer = Specialist(manifest, store).ask("At what core temperature does overheating begin?")

    if len(answer.cited_source_ids) == 1:
        assert any("single source" in limitation for limitation in answer.limitations)


def test_rendered_answer_exposes_the_evidence_surface(built):
    manifest, store = built
    rendered = Specialist(manifest, store).ask(
        "At what core temperature does overheating begin?"
    ).render()

    for section in ("ANSWER", "SOURCES", "EVIDENCE CONFIDENCE", "LIMITATIONS"):
        assert section in rendered


def test_api_shape_matches_the_documented_contract(built):
    manifest, store = built
    payload = Specialist(manifest, store).ask("what pressure causes venting").as_dict()

    for key in ("answer", "sources", "confidence", "knowledge_version", "evaluation_status"):
        assert key in payload


def test_sufficiency_gate_blocks_generation_when_evidence_is_weak(built):
    """The model must never be asked a question the corpus cannot support."""
    manifest, store = built
    manifest.specialist.min_evidence_score = 99.0
    provider = ScriptedProvider("ANSWER\nSomething confident and wrong. [1]")

    answer = Specialist(manifest, store, provider=provider).ask("what pressure causes venting")

    assert answer.status == INSUFFICIENT_EVIDENCE
    assert provider.calls == []  # the model was never called


# -- citation commentary is not a claim ---------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "Stated directly in the cited FAA passage.",
        "Taken from the evidence above.",
        "This is drawn from the cited sources.",
        "The answer is assembled from passages [1] and [2].",
    ],
)
def test_citation_commentary_is_not_graded_as_a_claim(sentence):
    """Models narrate their own citations in endlessly varied wording.

    Graded as claims, those sentences fail grounding -- and on a short answer,
    where one sentence is a large share of the total, that drags the whole
    answer past the unsupported threshold and withholds a correct answer.
    """
    from foundry.specialist.validation import is_meta

    assert is_meta(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Widget overheating begins above 80 °C. [1]",
        "The source states the limit is 100 Wh. [2]",
        "Spare batteries must be carried in the cabin. [1]",
        "Thermal runaway propagates to adjacent cells. [3]",
    ],
)
def test_a_sentence_that_asserts_something_is_still_a_claim(sentence):
    """The exemption must not become a way to smuggle assertions past the check."""
    from foundry.specialist.validation import is_meta

    assert not is_meta(sentence)


def test_a_short_answer_with_a_reasoning_line_is_not_withheld(built):
    """Regression: a two-sentence answer failed at 50% unsupported and was withheld."""
    manifest, store = built
    provider = ScriptedProvider(
        "ANSWER\nWidget overheating begins when the core temperature exceeds 80 °C. [1]\n\n"
        "REASONING\nStated directly in the cited passage."
    )
    answer = Specialist(manifest, store, provider=provider).ask(
        "At what core temperature does overheating begin?"
    )

    assert answer.status == ANSWERED
    assert answer.grounded_ratio == 1.0
