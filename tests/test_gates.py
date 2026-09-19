"""The two gates that ask whether an answer is *about* the question.

Every other check asks whether there is enough evidence, or whether what was
said is supported by it. Both can be satisfied by an answer that is about
something else entirely -- which is what a real submission got back.
"""

from __future__ import annotations

import pytest

from foundry.specialist import Specialist
from foundry.specialist.answer import IRRELEVANT
from foundry.text import content_terms


# -- the tokenisation bug both gates were measured through ---------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What's the battery capacity?", {"battery", "capacity"}),
        ("the battery's vent path", {"battery", "vent", "path"}),
        ("batteries' casings", {"battery", "casing"}),
    ],
)
def test_a_possessive_does_not_become_a_content_term(text, expected):
    """"what's" used to stem to "what'" -- a term in no document ever written.

    Harmless looking, and not harmless: any measure that weights question terms
    by rarity gives an unmatchable term the highest weight available, so one
    apostrophe could dominate what a question was judged to be about.
    """
    assert content_terms(text) == expected


def test_no_content_term_is_left_holding_an_apostrophe():
    for term in content_terms("What's the operator's duty? The batteries' state."):
        assert not term.endswith("'")


# -- gate 4: the corpus does not hold the subject ------------------------


def test_a_question_about_nothing_in_the_corpus_is_refused_by_name(built):
    manifest, store = built
    specialist = Specialist(manifest, store)

    answer = specialist.ask("What is the resale value of a widget in Lagos in 2031?")

    assert answer.abstained
    # The refusal names the gap, which is the whole point: it tells the reader
    # how to rephrase, and tells the knowledge area what it would need.
    assert any(term in answer.answer for term in ("lago", "resal", "valu"))


def test_a_question_the_corpus_covers_is_not_refused(built):
    manifest, store = built
    specialist = Specialist(manifest, store)

    ok, missing, share = specialist.covers_the_subject(
        "At what core temperature does widget overheating begin?",
        specialist.retriever.retrieve("At what core temperature does widget overheating begin?").evidence,
    )
    assert ok
    assert share < manifest.specialist.max_missing_subject_weight


def test_the_subject_gate_can_be_stood_down_by_the_manifest(built):
    manifest, store = built
    manifest.specialist.max_missing_subject_weight = 1.0
    specialist = Specialist(manifest, store)

    question = "What is the resale value of a widget in Lagos in 2031?"
    ok, _missing, _share = specialist.covers_the_subject(
        question, specialist.retriever.retrieve(question).evidence
    )
    assert ok


# -- gate 5: the answer is not about the question ------------------------


def test_an_answer_that_ignores_the_question_is_withheld(built):
    """Grounded, correctly cited, and about the wrong thing."""
    from foundry.llm.local import ScriptedProvider

    manifest, store = built
    # Grounding is ordered ahead of this gate and would claim the answer first,
    # so it is stood down to leave exactly one gate under test -- the same way
    # the red-team tests isolate the judge.
    manifest.specialist.max_unsupported_ratio = 1.0
    evasive = ScriptedProvider(
        "ANSWER\nWidgets must be inspected every 180 days under the Widget "
        "Safety Regulation 2024. [1] Inspection records shall be retained for "
        "5 years. [1]"
    )
    answer = Specialist(manifest, store, provider=evasive).ask(
        "At what internal pressure does a widget vent?"
    )
    assert answer.status == IRRELEVANT
    assert answer.abstained
    assert answer.confidence == "none"


def test_a_responsive_answer_survives_the_gate(built):
    from foundry.llm.local import ScriptedProvider

    manifest, store = built
    good = ScriptedProvider(
        "ANSWER\nA widget vents when internal pressure exceeds 12 bar. [1] "
        "Vented gas contains carbon monoxide and widget vapour. [1]"
    )
    answer = Specialist(manifest, store, provider=good).ask(
        "What does a widget vent and at what pressure?"
    )
    assert answer.status != IRRELEVANT


def test_the_responsiveness_gate_can_be_stood_down_by_the_manifest(built):
    from foundry.llm.local import ScriptedProvider

    manifest, store = built
    manifest.specialist.min_question_coverage = 0.0
    evasive = ScriptedProvider(
        "ANSWER\nWidgets must be inspected every 180 days. [1]"
    )
    answer = Specialist(manifest, store, provider=evasive).ask(
        "At what internal pressure does a widget vent?"
    )
    assert answer.status != IRRELEVANT


# -- the manifest bug found while measuring ------------------------------


@pytest.mark.parametrize("ka_id", ["kb-001-battery-failure", "kb-002-industrial-fire-explosion"])
def test_a_shipped_manifest_means_what_it_says(ka_id, monkeypatch):
    """Adding the new keys orphaned two old ones under the wrong parent.

    Nothing complained. `sufficiency` simply fell back to its defaults, an
    eight-times stricter evidence bar, and the suite dropped from 76 passes to
    49 -- visible only because it happened to be measured. A manifest whose
    keys land under the wrong parent is indistinguishable from one that does
    not set them.
    """
    import yaml

    from foundry import paths
    from foundry.manifest import Manifest

    monkeypatch.delenv("FOUNDRY_ROOT", raising=False)
    raw = yaml.safe_load(
        (paths.knowledge_areas_dir() / ka_id / "manifest.yaml").read_text(encoding="utf-8")
    )
    declared = raw["specialist"]["sufficiency"]
    loaded = Manifest.load(ka_id).specialist

    assert loaded.min_evidence_score == declared["min_evidence_score"]
    assert loaded.min_chunks == declared["min_chunks"]
    assert loaded.max_missing_subject_weight == declared["max_missing_subject_weight"]
    assert (
        loaded.min_question_coverage
        == raw["specialist"]["responsiveness"]["min_question_coverage"]
    )


# -- the licence guarantee, enforced rather than hoped for ---------------


def test_a_request_to_quote_an_unstored_source_is_refused_by_the_register(built):
    """The register says the text is not held. That must be the reason given.

    Five such questions ship across the two real knowledge areas and four of
    them abstained already -- for unrelated reasons, because no sentence
    happened to clear the selection threshold. The fifth answered from a
    secondary source that merely mentioned the standard, grounded and correctly
    cited, reading as though it had complied. A licence guarantee resting on
    retrieval luck is not a guarantee.
    """
    manifest, store = built
    specialist = Specialist(manifest, store)

    answer = specialist.ask("Quote the exact text of Widget Standard 9000 word for word.")

    assert answer.abstained
    assert "does not store" in answer.answer
    assert "Widget Standard 9000" in answer.answer
    # Decided from the register, so it names no evidence at all.
    assert answer.retrieval.get("reproduction_refused")
    assert answer.sources == []


def test_the_refusal_says_what_can_be_asked_instead(built):
    manifest, store = built
    answer = Specialist(manifest, store).ask("Reproduce Widget Standard 9000 in full.")
    assert "describe it can be cited" in answer.answer


@pytest.mark.parametrize("question", [
    # Names the source but asks what it requires, which is answerable.
    "What does Widget Standard 9000 require for inspection?",
    # Asks for a verbatim quote of something the corpus actually holds.
    "Quote the exact text of the Widget Overheating Handbook on venting.",
    # No designation at all.
    "At what core temperature does widget overheating begin?",
])
def test_the_gate_does_not_refuse_what_it_may_answer(built, question):
    """Over-refusing is the more damaging failure of the two.

    Nothing in the shipped suites asks to quote a source whose text *is* held,
    so this half of the gate had no coverage at all until it was written.
    """
    manifest, store = built
    answer = Specialist(manifest, store).ask(question)
    assert not answer.retrieval.get("reproduction_refused"), answer.answer[:120]


def test_the_gate_runs_before_retrieval(built):
    """What may be reproduced is a property of the register.

    Asking the evidence instead only ever finds whatever happens to mention the
    thing -- which is precisely how the failing answer was produced.
    """
    manifest, store = built
    specialist = Specialist(manifest, store)

    class Exploding:
        def retrieve(self, *a, **k):
            raise AssertionError("retrieval ran before the licence was checked")

    specialist.retriever = Exploding()
    answer = specialist.ask("Reproduce Widget Standard 9000 word for word.")
    assert answer.abstained
