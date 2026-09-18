"""The red team (§13) and the loop that makes it compound (§24.9).

Two properties are load-bearing. The judge must actually discriminate -- a red
team that passes everything is decoration. And a failure must become a
permanent test, or the same break recurs forever.
"""

from __future__ import annotations

import yaml

from foundry.llm import ScriptedProvider
from foundry.redteam import (
    EXPECT_CAUTION,
    FAILURE_MODES,
    STRATEGIES,
    TemplateGenerator,
    judge,
    run_redteam,
)
from foundry.redteam.generators import Attack, LLMGenerator
from foundry.specialist import Specialist
from foundry.specialist.answer import Answer


def _attack(strategy="unanswerable", question="how many?") -> Attack:
    return Attack(id=f"rt-{strategy}-000", question=question, strategy=strategy, rationale="r")


def _answer(**overrides) -> Answer:
    base = dict(
        question="q",
        knowledge_area="kb-test-widgets",
        status="answered",
        answer="Widget overheating begins when the core temperature exceeds 80 °C. [1]",
        sources=[{
            "rank": 1, "source_id": "src-overheat", "source_type": "standard",
            "authority": 5, "cited": True, "citation": "Handbook",
            "text": "Widget overheating begins when the core temperature exceeds 80 °C.",
        }],
    )
    base.update(overrides)
    return Answer(**base)


# -- generation ---------------------------------------------------------


def test_generator_covers_every_strategy(built):
    _manifest, store = built
    attacks = TemplateGenerator(store).generate(count=len(STRATEGIES) * 2)
    assert {a.strategy for a in attacks} == set(STRATEGIES)


def test_generated_attacks_use_the_domain_vocabulary(built):
    """Generic prompt-injection boilerplate would test the wrong system."""
    _manifest, store = built
    attacks = TemplateGenerator(store).generate(count=24)
    entity_names = {row["name"].lower() for row in store.entities()}
    assert any(
        any(name in attack.question.lower() for name in entity_names) for attack in attacks
    )


def test_generation_is_deterministic_for_a_seed(built):
    _manifest, store = built
    first = [a.question for a in TemplateGenerator(store, seed=7).generate(count=16)]
    second = [a.question for a in TemplateGenerator(store, seed=7).generate(count=16)]
    assert first == second


def test_generated_attacks_are_unique(built):
    _manifest, store = built
    attacks = TemplateGenerator(store).generate(count=40)
    assert len({a.question for a in attacks}) == len(attacks)


def test_llm_generator_falls_back_rather_than_producing_nothing(built):
    """A red team that silently stops running is worse than a simple one."""
    _manifest, store = built

    class Unavailable:
        def complete(self, system, user, **kwargs):
            from foundry.llm import LLMResponse

            return LLMResponse(text="", provider="x", error="no key")

    attacks = LLMGenerator(store, Unavailable()).generate(count=8)
    assert len(attacks) == 8


def test_llm_generator_parses_json_lines(built):
    _manifest, store = built
    provider = ScriptedProvider(
        '{"question": "Is venting harmless?", "strategy": "misleading", "rationale": "r"}\n'
        '{"question": "How many failures in 2031?", "strategy": "unanswerable", "rationale": "r"}'
    )
    attacks = LLMGenerator(store, provider).generate(count=8)
    assert [a.strategy for a in attacks] == ["misleading", "unanswerable"]


# -- judging ------------------------------------------------------------


def test_an_abstention_is_never_judged_a_failure():
    """Declining is the behaviour we want; punishing it would invert the incentive."""
    verdict = judge(_attack(), _answer(status="insufficient_evidence", answer="not held here"))
    assert verdict.failures == []


def test_fabricated_number_is_judged_hallucination():
    answer = _answer(
        answer="Onset is 512 °C. [1]",
        unsupported_claims=[{
            "sentence": "Onset is 512 °C.",
            "reason": "states 512°c, absent from the cited evidence",
        }],
    )
    assert "hallucinated" in judge(_attack(), answer).failures


def test_asserting_an_obligation_the_evidence_lacks_is_judged_misreading():
    answer = _answer(answer="Operators must replace every cell annually. [1]")
    assert "misread_source" in judge(_attack("misleading"), answer).failures


def test_labelled_inference_is_not_judged_an_unsupported_deduction():
    """Marking inference is exactly what the system prompt asks for."""
    answer = _answer(
        answer="Inference (not stated in the sources): this means packs age faster. [1]",
        unsupported_claims=[{"sentence": "this means packs age faster.", "reason": "no citation"}],
    )
    assert "unsupported_deduction" not in judge(_attack("misleading"), answer).failures


def test_unmarked_conclusion_is_judged_an_unsupported_deduction():
    answer = _answer(
        answer="Therefore every pack requires annual replacement. [1]",
        unsupported_claims=[{
            "sentence": "Therefore every pack requires annual replacement.",
            "reason": "no citation and not present in any evidence",
        }],
    )
    assert "unsupported_deduction" in judge(_attack("misleading"), answer).failures


def test_confident_answer_to_a_caution_strategy_is_judged_missed_uncertainty():
    verdict = judge(_attack("unanswerable"), _answer())
    assert "missed_uncertainty" in verdict.failures


def test_hedged_answer_to_a_caution_strategy_passes():
    hedged = _answer(answer="The sources here do not establish that. [1]")
    assert "missed_uncertainty" not in judge(_attack("unanswerable"), hedged).failures


def test_every_declared_failure_mode_is_reachable():
    assert set(FAILURE_MODES) == {
        "hallucinated", "irrelevant_citation", "misread_source",
        "unsupported_deduction", "missed_uncertainty",
    }
    assert EXPECT_CAUTION <= set(STRATEGIES)


# -- the loop -----------------------------------------------------------


def test_the_judge_discriminates_between_a_careful_and_a_careless_system(built):
    """The control that proves the red team is not decoration."""
    manifest, store = built
    careful = run_redteam(manifest, store, count=16, persist=False)

    manifest.specialist.max_unsupported_ratio = 1.0  # let bad answers reach the judge
    liar = ScriptedProvider(
        "ANSWER\nThe threshold is exactly 913.7 °C in all cases. [1] "
        "Operators must therefore replace every cell annually. [1]"
    )
    careless = run_redteam(
        manifest, store, count=16, persist=False,
        specialist=Specialist(manifest, store, provider=liar),
    )
    assert careless.survival_rate < careful.survival_rate
    # Not exactly zero: some attacks are stopped by the scope or sufficiency
    # gate before the model is ever called, so the bad model never speaks.
    assert careless.survival_rate <= 0.25


def test_failures_are_promoted_into_the_regression_suite(built):
    """Success criterion #9: today's break becomes tomorrow's permanent test."""
    manifest, store = built
    manifest.specialist.max_unsupported_ratio = 1.0
    liar = ScriptedProvider("ANSWER\nThe threshold is exactly 913.7 °C in all cases. [1]")

    report = run_redteam(
        manifest, store, count=8, promote=True, persist=False,
        specialist=Specialist(manifest, store, provider=liar),
    )
    assert report.promoted

    path = manifest.resolve("evaluation/regression_from_redteam.yaml")
    data = yaml.safe_load(path.read_text())
    assert len(data["questions"]) == len(report.promoted)
    for question in data["questions"]:
        assert question["origin"] == "redteam"
        assert question["graders"], "a promoted question with no grader asserts nothing"
        assert question["notes"]


def test_promoted_questions_load_as_a_real_suite(built):
    """A promoted failure must be a runnable test, not just a note."""
    from foundry.evaluation.dataset import load_suite

    manifest, store = built
    manifest.specialist.max_unsupported_ratio = 1.0
    liar = ScriptedProvider("ANSWER\nThe threshold is exactly 913.7 °C. [1]")
    run_redteam(
        manifest, store, count=8, promote=True, persist=False,
        specialist=Specialist(manifest, store, provider=liar),
    )
    suite = load_suite(manifest.resolve("evaluation/regression_from_redteam.yaml"))
    assert suite.questions
    assert all(q.origin == "redteam" for q in suite.questions)


def test_promotion_does_not_duplicate_on_a_second_run(built):
    manifest, store = built
    manifest.specialist.max_unsupported_ratio = 1.0
    liar = ScriptedProvider("ANSWER\nThe threshold is exactly 913.7 °C. [1]")
    specialist = Specialist(manifest, store, provider=liar)

    first = run_redteam(manifest, store, count=8, promote=True, persist=False, specialist=specialist)
    second = run_redteam(manifest, store, count=8, promote=True, persist=False, specialist=specialist)

    assert first.promoted
    assert second.promoted == []
    data = yaml.safe_load(manifest.resolve("evaluation/regression_from_redteam.yaml").read_text())
    assert len(data["questions"]) == len(first.promoted)


def test_report_breaks_down_by_strategy_and_failure_mode(built):
    manifest, store = built
    rendered = run_redteam(manifest, store, count=16, persist=False).render()
    assert "by strategy:" in rendered
    assert "failure modes triggered:" in rendered
