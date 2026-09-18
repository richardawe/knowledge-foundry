"""Versioning, the publish gate and rollback (§12).

A version must identify a set of inputs, and the gate must actually stop a bad
build from shipping. Both are tested here against real state transitions rather
than mocks, because the interesting bugs live in the transitions.
"""

from __future__ import annotations

import json

import pytest

from foundry.evaluation import run_evaluation, save_baseline
from foundry.versioning import (
    check_regressions,
    cut_version,
    list_versions,
    publish_version,
    published_version,
    rollback_version,
    version_history,
)


def _passing_report(version: str, **overrides) -> dict:
    report = {
        "run_id": "run_x",
        "version": version,
        "passed": True,
        "threshold_failures": [],
        "metrics": {
            "answer_correctness": 0.80,
            "citation_validity": 1.0,
            "retrieval_recall": 0.90,
            "hallucination_rate": 0.0,
        },
        "results": [
            {"id": "q1", "passed": True},
            {"id": "q2", "passed": True},
        ],
        "total": 2,
        "passed_count": 2,
    }
    report.update(overrides)
    return report


# -- version identity ---------------------------------------------------


def test_identical_inputs_yield_the_identical_version(built):
    """Otherwise 'what changed between versions' stops being answerable."""
    manifest, store = built
    first = cut_version(manifest, store, notes="one")
    second = cut_version(manifest, store, notes="two")

    assert second.version == first.version
    assert second.reused
    assert len(list_versions(store)) == 1


def test_reusing_a_version_repoints_the_serving_pointer(built):
    """A rebuild that restores an earlier corpus must serve under that version."""
    manifest, store = built
    first = cut_version(manifest, store)
    manifest.retrieval.top_k = 6
    second = cut_version(manifest, store)
    assert second.version != first.version

    manifest.retrieval.top_k = 4  # back to the original configuration
    third = cut_version(manifest, store)
    assert third.version == first.version
    assert store.get_meta("current_version") == first.version


def test_configuration_change_bumps_the_minor_number(built):
    """A config change can alter every answer, so it is not a patch."""
    manifest, store = built
    first = cut_version(manifest, store)
    manifest.specialist.min_sentence_support = 0.5
    second = cut_version(manifest, store)

    assert first.version == "v0.1.0"
    assert second.version == "v0.2.0"


def test_corpus_change_bumps_the_patch_number(built):
    """New knowledge, same behaviour."""
    manifest, store = built
    first = cut_version(manifest, store)
    store.conn.execute("DELETE FROM documents WHERE source_id = 'src-incident'")
    store.conn.commit()
    second = cut_version(manifest, store)

    assert first.version == "v0.1.0"
    assert second.version == "v0.1.1"


# -- the publish gate ---------------------------------------------------


def test_publishing_without_an_evaluation_is_refused(built):
    manifest, store = built
    version = cut_version(manifest, store)
    result = publish_version(manifest, store, report=None)

    assert not result.published
    assert any("no evaluation report" in r for r in result.reasons)
    assert published_version(store) is None


def test_a_report_from_another_build_cannot_authorise_a_release(built):
    """A stale pass must not ship a new build."""
    manifest, store = built
    version = cut_version(manifest, store)
    stale = _passing_report("v0.0.9")

    result = publish_version(manifest, store, report=stale)
    assert not result.published
    assert any("not " + version.version in r for r in result.reasons)


def test_failing_thresholds_block_publication(built):
    manifest, store = built
    version = cut_version(manifest, store)
    failing = _passing_report(
        version.version, passed=False,
        threshold_failures=["citation_validity 0.400 < 0.95"],
    )
    result = publish_version(manifest, store, report=failing)

    assert not result.published
    assert any("citation_validity" in r for r in result.reasons)


def test_a_clean_build_publishes_and_becomes_the_baseline(built):
    manifest, store = built
    version = cut_version(manifest, store)
    report = _passing_report(version.version)

    result = publish_version(manifest, store, report=report)

    assert result.published
    assert published_version(store) == version.version
    assert store.get_meta("published_version") == version.version
    history = version_history(manifest)
    assert history and history[0]["version"] == version.version


def test_a_metric_regression_holds_the_build(built):
    """Passing your own thresholds is not enough if you got worse."""
    manifest, store = built
    first = cut_version(manifest, store)
    publish_version(manifest, store, report=_passing_report(first.version))

    manifest.specialist.min_sentence_support = 0.5
    second = cut_version(manifest, store)
    worse = _passing_report(second.version)
    worse["metrics"]["answer_correctness"] = 0.60

    result = publish_version(manifest, store, report=worse)

    assert not result.published
    assert any("answer_correctness" in r for r in result.regressions)
    assert published_version(store) == first.version  # the old build keeps serving


def test_newly_failing_questions_are_named_not_just_counted(built):
    """'Investigate' must be a concrete list."""
    manifest, store = built
    first = cut_version(manifest, store)
    publish_version(manifest, store, report=_passing_report(first.version))

    manifest.specialist.min_sentence_support = 0.5
    second = cut_version(manifest, store)
    regressed = _passing_report(second.version)
    regressed["results"] = [{"id": "q1", "passed": False}, {"id": "q2", "passed": True}]

    result = publish_version(manifest, store, report=regressed)
    assert result.newly_failing == ["q1"]


def test_lower_is_better_metrics_regress_upward(built):
    manifest, store = built
    baseline = _passing_report("v0.1.0")
    report = _passing_report("v0.1.1")
    report["metrics"]["hallucination_rate"] = 0.20

    regressions, _ = check_regressions(manifest, report, baseline)
    assert any("hallucination_rate rose" in r for r in regressions)


def test_a_first_build_has_nothing_to_regress_against(built):
    manifest, store = built
    regressions, failing = check_regressions(manifest, _passing_report("v0.1.0"), None)
    assert regressions == [] and failing == []


def test_force_publishes_but_records_that_it_overrode_the_gate(built):
    """An override must leave a trail, or the gate means nothing."""
    manifest, store = built
    version = cut_version(manifest, store)
    failing = _passing_report(version.version, passed=False, threshold_failures=["bad"])

    result = publish_version(manifest, store, report=failing, force=True)

    assert result.published
    assert any("--force" in r for r in result.reasons)
    assert store.get_meta("evaluation_status") == "forced"


# -- rollback -----------------------------------------------------------


def test_rollback_repoints_at_the_previous_published_version(built):
    manifest, store = built
    first = cut_version(manifest, store)
    publish_version(manifest, store, report=_passing_report(first.version))

    manifest.specialist.min_sentence_support = 0.5
    second = cut_version(manifest, store)
    publish_version(manifest, store, report=_passing_report(second.version))
    assert published_version(store) == second.version

    message = rollback_version(store)
    assert first.version in message
    assert store.get_meta("published_version") == first.version


def test_rollback_with_nothing_to_roll_back_to_says_so(built):
    manifest, store = built
    version = cut_version(manifest, store)
    publish_version(manifest, store, report=_passing_report(version.version))

    assert "no earlier published version" in rollback_version(store)


def test_version_history_is_written_into_the_knowledge_area(built):
    """Version history is part of the asset and renders as the Updates section."""
    manifest, store = built
    version = cut_version(manifest, store)
    publish_version(manifest, store, report=_passing_report(version.version))

    path = manifest.resolve("versions") / f"{version.version}.json"
    assert path.is_file()
    record = json.loads(path.read_text())
    assert record["evaluation"]["passed"] is True
    assert record["stats"]["chunks"] > 0


# -- end to end ---------------------------------------------------------


def test_real_evaluation_drives_a_real_publish(built):
    manifest, store = built
    version = cut_version(manifest, store)
    report = run_evaluation(manifest, store, persist=False)
    report_dict = report.as_dict()

    result = publish_version(manifest, store, report=report_dict)
    if report.passed:
        assert result.published
    else:
        assert not result.published
