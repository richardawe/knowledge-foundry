"""The evaluation framework (§11, §12).

The tests that matter here are the ones that stop the suite lying: a question
with no graders must not silently pass, a metric must move in the right
direction, and the gate must actually block a bad build.
"""

from __future__ import annotations

import pytest
import yaml

from foundry.evaluation import (
    METRICS,
    Question,
    available_graders,
    compute_metrics,
    grade,
    is_regression,
    load_suite,
    run_evaluation,
)
from foundry.evaluation.dataset import parse_suite
from foundry.evaluation.runner import QuestionResult, _hallucinated
from foundry.llm import ScriptedProvider
from foundry.manifest import ManifestError
from foundry.specialist import Specialist
from foundry.specialist.answer import Answer


def _answer(**overrides) -> Answer:
    base = dict(
        question="q",
        knowledge_area="kb-test-widgets",
        status="answered",
        answer="Widget overheating begins when the core temperature exceeds 80 °C. [1]",
        sources=[
            {
                "rank": 1, "source_id": "src-overheat", "source_type": "standard",
                "authority": 5, "cited": True, "citation": "Handbook",
            }
        ],
        citation_validity=1.0,
        grounded_ratio=1.0,
    )
    base.update(overrides)
    return Answer(**base)


# -- dataset ------------------------------------------------------------


def test_question_without_graders_is_rejected():
    """A question that asserts nothing would inflate every pass rate silently."""
    with pytest.raises(ManifestError, match="no graders"):
        parse_suite({"suite": "s", "questions": [{"id": "q1", "question": "why?"}]})


def test_unknown_question_type_is_rejected():
    with pytest.raises(ManifestError, match="unknown type"):
        parse_suite({
            "suite": "s",
            "questions": [{"id": "q1", "question": "why?", "type": "vibes",
                           "graders": [{"kind": "must_answer"}]}],
        })


def test_duplicate_question_ids_are_rejected():
    spec = {"id": "q1", "question": "why?", "graders": [{"kind": "must_answer"}]}
    with pytest.raises(ManifestError, match="duplicate"):
        parse_suite({"suite": "s", "questions": [spec, dict(spec)]})


@pytest.mark.parametrize("qtype", ["unanswerable", "out_of_scope"])
def test_abstain_types_expect_abstention(qtype):
    question = Question(id="q", question="?", type=qtype, graders=[{"kind": "must_abstain"}])
    assert question.should_abstain


def test_missing_suite_file_is_skipped_not_fatal(manifest):
    """regression_from_redteam.yaml does not exist until the red team finds something."""
    from foundry.evaluation.dataset import load_suites

    manifest.evaluation.suites = ["evaluation/core.yaml", "evaluation/does-not-exist.yaml"]
    suites = load_suites(manifest)
    assert [s.name for s in suites] == ["core"]


# -- graders ------------------------------------------------------------


def test_every_registered_grader_is_documented_by_name():
    assert "must_abstain" in available_graders()
    assert "numeric_within" in available_graders()


def test_unknown_grader_is_a_loud_error():
    with pytest.raises(ValueError, match="unknown grader"):
        grade({"kind": "vibes_check"}, _answer())


def test_must_include_any_of():
    assert grade({"kind": "must_include", "any_of": ["80"]}, _answer()).passed
    assert not grade({"kind": "must_include", "any_of": ["95"]}, _answer()).passed


def test_must_not_include_catches_banned_text():
    result = grade({"kind": "must_not_include", "none_of": ["80 °c"]}, _answer())
    assert not result.passed


def test_numeric_within_tolerates_formatting_not_wrong_values():
    """80 and 80.0 are the same claim; 95 is a different one."""
    assert grade({"kind": "numeric_within", "value": 80, "tolerance": 0, "unit": "°c"}, _answer()).passed
    assert not grade({"kind": "numeric_within", "value": 95, "tolerance": 5, "unit": "°c"}, _answer()).passed


def test_must_cite_source_checks_citation_not_mention():
    """Naming a source in prose is not citing it."""
    uncited = _answer(sources=[{**_answer().sources[0], "cited": False}])
    assert grade({"kind": "must_cite_source", "source_ids": ["src-overheat"]}, _answer()).passed
    assert not grade({"kind": "must_cite_source", "source_ids": ["src-overheat"]}, uncited).passed


def test_must_cite_type_catches_right_words_wrong_source():
    """A regulatory question answered from an encyclopaedia is wrong."""
    tertiary = _answer(sources=[{**_answer().sources[0], "source_type": "reference", "authority": 2}])
    assert not grade({"kind": "must_cite_type", "source_types": ["regulatory"]}, tertiary).passed


def test_must_abstain_treats_declining_as_success():
    declined = _answer(status="insufficient_evidence")
    assert grade({"kind": "must_abstain"}, declined).passed
    assert not grade({"kind": "must_abstain"}, _answer()).passed


def test_must_flag_uncertainty_accepts_either_abstention_or_qualification():
    assert grade({"kind": "must_flag_uncertainty"}, _answer(status="ambiguous")).passed
    qualified = _answer(answer="The onset temperature depends on the cell chemistry. [1]")
    assert grade({"kind": "must_flag_uncertainty"}, qualified).passed
    assert not grade({"kind": "must_flag_uncertainty"}, _answer()).passed


def test_retrieval_hit_is_scored_separately_from_the_answer():
    """So a retrieval miss and a generation miss are distinguishable."""
    result = grade({"kind": "retrieval_hit", "source_ids": ["src-overheat"]}, _answer())
    assert result.passed


def test_judge_grader_is_skipped_without_a_provider():
    """CI must reach a verdict with no API key."""
    result = grade({"kind": "judge", "rubric": "is it right?"}, _answer())
    assert result.passed
    assert "skipped" in result.detail


def test_judge_grader_uses_the_provider_when_one_is_given():
    provider = ScriptedProvider("FAIL - the answer omits the qualifying conditions")
    result = grade({"kind": "judge", "_provider": provider, "rubric": "r"}, _answer())
    assert not result.passed


# -- metrics ------------------------------------------------------------


def test_every_metric_declares_a_direction():
    for name, spec in METRICS.items():
        assert spec.direction in ("higher", "lower"), name


def test_regression_direction_is_respected():
    assert is_regression("answer_correctness", current=0.70, baseline=0.80, tolerance=0.02)
    assert not is_regression("answer_correctness", current=0.85, baseline=0.80, tolerance=0.02)
    # Lower is better, so a rise is the regression.
    assert is_regression("hallucination_rate", current=0.10, baseline=0.02, tolerance=0.02)
    assert not is_regression("hallucination_rate", current=0.01, baseline=0.02, tolerance=0.02)


def test_hallucination_is_narrower_than_unsupported():
    """Connective prose is untidy; a fabricated number is the failure that matters."""
    prose = _answer(unsupported_claims=[
        {"sentence": "This is relevant context about widgets.", "reason": "no citation and not present in any evidence"}
    ])
    fabricated = _answer(unsupported_claims=[
        {"sentence": "Overheating begins at 210 °C.", "reason": "states 210°c, absent from the cited evidence"}
    ])
    assert not _hallucinated(prose)
    assert _hallucinated(fabricated)


def test_abstention_metric_ignores_questions_with_no_single_right_answer(built):
    """Declining a false premise is correct, so adversarial questions are excluded."""
    _manifest, store = built
    adversarial = Question(
        id="a1", question="?", type="adversarial",
        graders=[{"kind": "must_not_include", "none_of": ["nonsense"]}],
    )
    factual = Question(id="f1", question="?", type="factual", graders=[{"kind": "must_answer"}])

    results = [
        QuestionResult(question=adversarial, answer=_answer(status="insufficient_evidence"), grades=[]),
        QuestionResult(question=factual, answer=_answer(), grades=[]),
    ]
    metrics = compute_metrics(results, store)
    # The adversarial abstention is excluded, so the factual answer scores 1.0.
    assert metrics["abstention_correctness"] == 1.0


def test_an_explicit_decision_grader_puts_an_adversarial_question_back_in_scope(built):
    """The exclusion is about absent expectations, not about the type name."""
    _manifest, store = built
    adversarial = Question(
        id="a1", question="?", type="adversarial", graders=[{"kind": "must_abstain"}]
    )
    results = [QuestionResult(question=adversarial, answer=_answer(), grades=[])]

    # It should have abstained and did not, so the metric registers the miss.
    assert compute_metrics(results, store)["abstention_correctness"] == 0.0


def test_regression_rate_needs_a_baseline(built):
    _manifest, store = built
    question = Question(id="q1", question="?", type="factual", graders=[{"kind": "must_answer"}])
    results = [QuestionResult(question=question, answer=_answer(), grades=[])]

    assert "regression_rate" not in compute_metrics(results, store)
    baseline = {"results": [{"id": "q1", "passed": True}]}
    assert compute_metrics(results, store, baseline=baseline)["regression_rate"] == 0.0


# -- runner -------------------------------------------------------------


def test_run_records_answers_not_just_scores(built):
    """A failing question with no record of what was said cannot be investigated."""
    manifest, store = built
    report = run_evaluation(manifest, store, persist=False)

    assert report.total == 3
    for result in report.results:
        assert result.as_dict()["answer"]
        assert "grades" in result.as_dict()


def test_threshold_failure_fails_the_build(built):
    manifest, store = built
    manifest.evaluation.thresholds = {"citation_validity": 1.01}  # unreachable
    report = run_evaluation(manifest, store, persist=False)

    assert not report.passed
    assert any("citation_validity" in f for f in report.threshold_failures)


def test_lower_is_better_thresholds_are_compared_the_right_way(built):
    manifest, store = built
    manifest.evaluation.thresholds = {"unsupported_claim_rate": 0.5}
    report = run_evaluation(manifest, store, persist=False)
    # A rate below the ceiling must not be reported as a failure.
    assert not any("unsupported_claim_rate" in f for f in report.threshold_failures)


def test_results_persist_for_later_investigation(built):
    manifest, store = built
    report = run_evaluation(manifest, store, persist=True)

    rows = store.conn.execute(
        "SELECT COUNT(*) AS n FROM eval_results WHERE run_id = ?", (report.run_id,)
    ).fetchone()["n"]
    assert rows == report.total
    assert store.get_meta("evaluation_status") in ("passed", "failed")


def test_report_renders_per_type_breakdown(built):
    manifest, store = built
    rendered = run_evaluation(manifest, store, persist=False).render()

    assert "by type:" in rendered
    assert "VERDICT" in rendered


def test_a_hallucinating_model_is_caught_by_the_suite(built):
    """End to end: a bad model must fail the gate, not slip through it."""
    manifest, store = built
    liar = ScriptedProvider(
        "ANSWER\nWidget overheating begins at 512 °C under the Widget Directive 1994/12/EC. [1]"
    )
    report = run_evaluation(
        manifest, store, specialist=Specialist(manifest, store, provider=liar), persist=False
    )
    assert report.metrics["answer_correctness"] < 1.0
