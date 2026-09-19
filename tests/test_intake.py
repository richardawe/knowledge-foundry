"""The public ask path that needs no server (§16).

These tests are about the seam, not the specialist: a stranger's text arriving
as an issue body, and a durable record coming out the other end. The answering
itself is covered by the specialist tests.
"""

from __future__ import annotations

import pytest

from foundry.feedback import import_challenges, list_challenges, read_challenge_files
from foundry.intake import (
    MAX_QUESTION_CHARS,
    IntakeError,
    Submission,
    handle,
    parse_issue_form,
    render_comment,
    render_refusal,
    submission_from_body,
)

AREA = "kb-test-widgets"


def body(question="At what core temperature does widget overheating begin?",
         area=AREA, problem="_No response_", expected="_No response_"):
    return (
        "### Knowledge area\n\n"
        f"{area}\n\n"
        "### Your question\n\n"
        f"{question}\n\n"
        "### What do you think is wrong? (optional)\n\n"
        f"{problem}\n\n"
        "### What should it have said or done? (optional)\n\n"
        f"{expected}\n"
    )


# -- parsing -------------------------------------------------------------


def test_parse_reads_every_field_of_a_rendered_form():
    fields = parse_issue_form(body(problem="It cited the wrong document", expected="90 °C"))
    assert fields["knowledge_area"] == AREA
    assert fields["question"].startswith("At what core temperature")
    assert fields["problem"] == "It cited the wrong document"
    assert fields["expected"] == "90 °C"


def test_an_optional_field_left_blank_reads_as_empty_not_as_the_placeholder():
    """`_No response_` is GitHub's wording, not the person's."""
    fields = parse_issue_form(body())
    assert fields["problem"] == ""
    assert fields["expected"] == ""


def test_windows_line_endings_parse_identically():
    assert parse_issue_form(body().replace("\n", "\r\n")) == parse_issue_form(body())


def test_a_question_spanning_several_lines_is_collapsed(ka_dir):
    submission = submission_from_body(body(question="What is\nthe vent\npressure?"))
    assert submission.question == "What is the vent pressure?"


# -- refusals, phrased to be posted back ---------------------------------


def test_free_text_with_no_form_is_refused_with_an_actionable_reason(ka_dir):
    with pytest.raises(IntakeError) as exc:
        submission_from_body("I just typed a question into the box")
    assert "issue form" in str(exc.value)


def test_an_unknown_knowledge_area_is_refused_and_lists_the_real_ones(ka_dir):
    with pytest.raises(IntakeError) as exc:
        submission_from_body(body(area="kb-999-nonexistent"))
    assert AREA in str(exc.value)


def test_an_overlong_question_is_refused_before_any_model_is_called(ka_dir):
    with pytest.raises(IntakeError) as exc:
        submission_from_body(body(question="widget " * MAX_QUESTION_CHARS))
    assert str(MAX_QUESTION_CHARS) in str(exc.value)


def test_a_refusal_renders_as_a_reply_rather_than_silence():
    rendered = render_refusal(IntakeError("the reason"))
    assert "the reason" in rendered
    assert rendered.startswith("## ")


# -- the exchange becomes a durable record -------------------------------


def test_answering_records_the_challenge_in_the_store_and_in_the_repository(built):
    manifest, store = built
    submission = submission_from_body(
        body(problem="Too vague"), submitted_by="@someone", reference="https://example.test/1"
    )
    result = handle(manifest, store, submission, challenge_id="ch_issue_1")

    rows = list_challenges(store)
    assert [row["id"] for row in rows] == ["ch_issue_1"]
    assert rows[0]["submitted_by"] == "@someone"
    assert rows[0]["claimed_problem"] == "Too vague"
    # The answer is stored with the complaint: a challenge nobody can reproduce
    # later is not evidence of anything.
    assert rows[0]["system_answer"] == result.answer.answer

    files = read_challenge_files(manifest)
    assert [record["id"] for record in files] == ["ch_issue_1"]
    assert "example.test" in files[0]["notes"]


def test_replaying_the_same_submission_updates_rather_than_duplicates(built):
    """A workflow re-run must not turn one complaint into two."""
    manifest, store = built
    for problem in ("first wording", "second wording"):
        handle(
            manifest, store,
            submission_from_body(body(problem=problem)),
            challenge_id="ch_issue_7",
        )

    rows = list_challenges(store)
    assert len(rows) == 1
    assert rows[0]["claimed_problem"] == "second wording"
    assert len(read_challenge_files(manifest)) == 1


def test_a_challenge_survives_the_store_being_thrown_away(built):
    """`var/` is disposable; a stranger's question is not derivable from sources."""
    manifest, store = built
    handle(manifest, store, submission_from_body(body()), challenge_id="ch_issue_9")

    store.conn.execute("DELETE FROM challenges")
    store.conn.commit()
    assert list_challenges(store) == []

    assert import_challenges(manifest, store) == 1
    assert [row["id"] for row in list_challenges(store)] == ["ch_issue_9"]


def test_importing_challenges_is_idempotent(built):
    manifest, store = built
    handle(manifest, store, submission_from_body(body()), challenge_id="ch_issue_11")
    import_challenges(manifest, store)
    import_challenges(manifest, store)
    assert len(list_challenges(store)) == 1


def test_a_review_decision_in_the_repository_survives_a_rebuild(built):
    """The repository is the record, so it wins over whatever the store holds."""
    from foundry.feedback import write_challenge_file

    manifest, store = built
    handle(manifest, store, submission_from_body(body()), challenge_id="ch_issue_13")
    record = dict(list_challenges(store)[0])
    record["status"] = "accepted"
    write_challenge_file(manifest, record)

    import_challenges(manifest, store)
    assert list_challenges(store)[0]["status"] == "accepted"


# -- the reply -----------------------------------------------------------


def test_the_reply_names_what_produced_the_answer(built):
    """Whoever is asked to judge an answer needs to know what wrote it."""
    manifest, store = built
    result = handle(manifest, store, submission_from_body(body()), challenge_id="ch_issue_3")

    assert "local_extractive" in result.comment
    assert result.answer.knowledge_version in result.comment
    assert "ch_issue_3" in result.comment
    assert "## Evidence confidence" in result.comment


def test_a_fallback_answer_is_flagged_rather_than_passed_off_as_the_model(built):
    manifest, store = built
    result = handle(manifest, store, submission_from_body(body()), challenge_id="ch_issue_4")
    result.answer.llm = {
        "provider": "local_extractive (fell back from ollama)", "model": "x [refused]"
    }
    rendered = render_comment(manifest, result)
    assert "configured model did not answer" in rendered


def test_an_abstention_is_reported_as_a_decision_not_an_answer(built):
    manifest, store = built
    submission = Submission(
        knowledge_area=manifest.id, question="How should I treat a bacterial infection?"
    )
    result = handle(manifest, store, submission, challenge_id="ch_issue_5")

    assert result.answer.status != "answered"
    assert "declined to answer" in result.comment
    assert "designed outcome" in result.comment


def test_the_reply_lists_the_sources_the_answer_actually_cited(built):
    manifest, store = built
    result = handle(manifest, store, submission_from_body(body()), challenge_id="ch_issue_6")

    if [s for s in result.answer.sources if s.get("cited")]:
        assert "## Sources" in result.comment


# -- the invariant the worker already enforces ---------------------------


def test_a_keyless_provider_is_refused_when_a_real_model_is_required(built):
    """Same rule as the worker: a corpus extract must not wear a model's name."""
    from foundry.llm import KeylessProviderRefused

    manifest, store = built
    with pytest.raises(KeylessProviderRefused):
        handle(
            manifest, store, submission_from_body(body()),
            challenge_id="ch_issue_8", require_model=True,
        )


# -- the command the workflow actually runs ------------------------------
#
# The module-level tests above call `handle` directly, which is not what
# production does. A workflow runs the CLI, and the first end-to-end run found
# a missing flag that every test above was blind to. These cover the seam.


def _cli(*argv: str) -> int:
    from foundry.cli import main

    return main(list(argv))


def test_the_cli_answers_a_submission_and_writes_the_reply(built, tmp_path):
    manifest, store = built
    store.close()          # the CLI opens its own

    body_file = tmp_path / "body.md"
    body_file.write_text(body(), encoding="utf-8")
    comment = tmp_path / "comment.md"

    code = _cli(
        "intake",
        "--body-file", str(body_file),
        "--challenge-id", "ch_issue_100",
        "--author", "@sceptic",
        "--reference", "https://example.test/issues/100",
        "--comment-out", str(comment),
    )

    assert code == 0
    assert "## Answer" in comment.read_text(encoding="utf-8")
    assert "ch_issue_100" in comment.read_text(encoding="utf-8")


def test_the_cli_writes_a_reply_and_a_distinct_exit_code_for_a_refusal(built, tmp_path):
    """The workflow posts the file whatever happened, then fails the run."""
    manifest, store = built
    store.close()

    body_file = tmp_path / "body.md"
    body_file.write_text(body(area="kb-999-nonexistent"), encoding="utf-8")
    comment = tmp_path / "comment.md"

    code = _cli(
        "intake",
        "--body-file", str(body_file),
        "--challenge-id", "ch_issue_101",
        "--comment-out", str(comment),
    )

    assert code == 3
    assert "could not answer" in comment.read_text(encoding="utf-8")


def test_the_cli_declines_rather_than_letting_a_keyless_reply_stand_in(built, tmp_path):
    manifest, store = built
    store.close()

    body_file = tmp_path / "body.md"
    body_file.write_text(body(), encoding="utf-8")
    comment = tmp_path / "comment.md"

    code = _cli(
        "intake",
        "--body-file", str(body_file),
        "--challenge-id", "ch_issue_102",
        "--comment-out", str(comment),
        "--require-model",
    )

    assert code == 4
    rendered = comment.read_text(encoding="utf-8")
    # Worded for the person waiting, not for the operator who misconfigured it.
    assert "no model is reachable" in rendered
    assert "inference queue" not in rendered
