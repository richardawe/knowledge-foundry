"""The inference queue: deterministic Python here, model calls over there.

The properties that matter are about the boundary. A reply must be validated
against the evidence it was actually shown, never against evidence that moved
underneath it; the worker must not be able to mark its own work; and a gate's
refusal must never be sent to a model at all.
"""

from __future__ import annotations

import json

import pytest

from foundry.evaluation import run_evaluation
from foundry.inference import (
    AnsweredFromQueue,
    apply,
    pending,
    plan,
    prompt_hash,
    queue_root,
    queue_status,
    record_response,
    worker,
)
from foundry.llm import LLMResponse, ScriptedProvider
from foundry.specialist import ANSWERED, Specialist


class Model:
    """Stands in for the machine with Ollama."""

    name, model = "ollama", "llama3.1:8b"

    def __init__(self, text="ANSWER\nWidget overheating begins when the core "
                            "temperature exceeds 80 °C. [1]"):
        self.text = text
        self.calls = []

    def complete(self, system, user, **kwargs):
        self.calls.append((system, user))
        return LLMResponse(text=self.text, provider=self.name, model=self.model,
                           prompt_tokens=900, completion_tokens=30, stop_reason="stop")


@pytest.fixture
def queued(built, tmp_path):
    manifest, store = built
    report = plan(manifest, store, root=tmp_path)
    return manifest, store, report, tmp_path


# -- planning -----------------------------------------------------------


def test_plan_queues_a_prompt_per_question_that_needs_a_model(queued):
    manifest, _store, report, root = queued
    assert report.written > 0
    files = list((queue_root(manifest.id, root) / "requests").glob("req_*.json"))
    assert len(files) == report.written


def test_a_gate_refusal_is_never_sent_to_a_model(built, tmp_path):
    """There is nothing a model can add to "this is out of scope".

    Queueing refusals would spend inference on answers the system has already
    correctly declined to give.
    """
    manifest, store = built
    refused = "Can you give me a medical diagnosis?"     # matches a declared out-of-scope topic
    answerable = "At what core temperature does overheating begin?"

    report = plan(manifest, store, root=tmp_path,
                  questions=[("q-refused", refused), ("q-ok", answerable)])

    assert report.answered_without_model == 1
    assert report.written == 1
    queued_questions = {
        json.loads(path.read_text())["question"]
        for path in (queue_root(manifest.id, tmp_path) / "requests").glob("req_*.json")
    }
    assert queued_questions == {answerable}


def test_the_system_prompt_is_stored_once_not_per_request(queued):
    manifest, _store, _report, root = queued
    assert (queue_root(manifest.id, root) / "system.json").is_file()

    request = json.loads(
        next((queue_root(manifest.id, root) / "requests").glob("req_*.json")).read_text()
    )
    assert "system" not in request
    # But the worker still receives it.
    assert pending(manifest.id, root)[0]["system"]


def test_replanning_unchanged_work_writes_nothing(queued):
    """Otherwise every run would churn the repository."""
    manifest, store, first, root = queued
    second = plan(manifest, store, root=root)

    assert second.written == 0
    assert second.unchanged == first.written


# -- the worker ---------------------------------------------------------


def test_the_worker_answers_pending_prompts(queued):
    manifest, _store, _report, root = queued
    model = Model()

    summary = worker([manifest], provider=model, root=root, worker_name="test-mac")

    assert summary["answered"] == len(model.calls) > 0
    assert summary["failed"] == 0
    assert queue_status(manifest.id, root)["pending"] == 0


def test_the_worker_does_no_retrieval_and_no_scoring(queued):
    """It reads a prompt, calls a model, writes the reply. Nothing else."""
    manifest, _store, _report, root = queued
    model = Model()
    worker([manifest], provider=model, root=root)

    recorded = json.loads(
        next((queue_root(manifest.id, root) / "responses").glob("req_*.json")).read_text()
    )
    assert recorded["provider"] == "ollama"
    # No verdict, no score, no citation analysis -- those are the other side's.
    for forbidden in ("status", "confidence", "grounded_ratio", "passed", "sources"):
        assert forbidden not in recorded


def test_a_model_error_is_recorded_not_swallowed(queued):
    manifest, _store, _report, root = queued

    class Broken:
        name, model = "ollama", "llama3.1:8b"

        def complete(self, system, user, **kwargs):
            return LLMResponse(text="", provider=self.name, model=self.model,
                               error="connection refused")

    summary = worker([manifest], provider=Broken(), root=root)
    assert summary["failed"] > 0
    assert summary["errors"][0]["error"] == "connection refused"


def test_the_worker_never_falls_back_to_the_keyless_provider(built, tmp_path, monkeypatch):
    """A fallback reply would masquerade as the model's in the queue."""
    manifest, store = built
    manifest.specialist.llm_provider = "ollama"
    plan(manifest, store, root=tmp_path)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")

    summary = worker([manifest], root=tmp_path)

    assert summary["answered"] == 0
    assert summary["failed"] > 0
    recorded = json.loads(
        next((queue_root(manifest.id, tmp_path) / "responses").glob("req_*.json")).read_text()
    )
    assert recorded["provider"] == "ollama"
    assert recorded["error"]


# -- applying -----------------------------------------------------------


def test_apply_validates_and_scores_without_calling_a_model(queued):
    manifest, store, _report, root = queued
    worker([manifest], provider=Model(), root=root)

    report, answers = apply(manifest, store, root=root)

    assert report.applied > 0
    assert report.providers == {"ollama": report.applied}
    answer = next(iter(answers.values()))
    # The grading happened here, over the evidence, not on the worker.
    assert answer.citation_validity is not None
    assert answer.sources


def test_a_reply_is_rejected_when_the_corpus_moved_underneath_it(queued):
    """The integrity guard. Without it, answers get graded against passages
    they were never shown."""
    manifest, store, _report, root = queued
    worker([manifest], provider=Model(), root=root)

    # Forge a reply to a prompt that no longer matches what the corpus builds.
    responses = sorted((queue_root(manifest.id, root) / "responses").glob("req_*.json"))
    data = json.loads(responses[0].read_text())
    data["prompt_sha"] = "0" * 64
    responses[0].write_text(json.dumps(data))

    report, answers = apply(manifest, store, root=root)

    assert data["id"] in report.stale
    assert len(answers) == report.applied


def test_a_response_with_no_request_is_reported(queued):
    manifest, store, _report, root = queued
    orphan = queue_root(manifest.id, root) / "responses" / "req_orphan.json"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text(json.dumps({"id": "req_orphan", "text": "hello", "prompt_sha": "x"}))

    report, _answers = apply(manifest, store, root=root)
    assert "req_orphan" in report.missing_request


def test_a_hallucinated_reply_is_caught_on_this_side(queued):
    """The worker cannot smuggle a bad answer past validation by generating it."""
    manifest, store, _report, root = queued
    liar = Model("ANSWER\nWidget overheating begins at 512 °C under the Widget "
                 "Directive 1994/12/EC. [1]")
    worker([manifest], provider=liar, root=root)

    _report, answers = apply(manifest, store, root=root)
    statuses = {a.status for a in answers.values()}
    assert ANSWERED not in statuses or any(a.unsupported_claims for a in answers.values())


def test_empty_queue_applies_nothing(built, tmp_path):
    manifest, store = built
    report, answers = apply(manifest, store, root=tmp_path)
    assert report.applied == 0 and answers == {}


# -- the join back into evaluation --------------------------------------


def test_evaluation_uses_queued_answers_and_falls_through_for_the_rest(queued):
    manifest, store, _report, root = queued
    worker([manifest], provider=Model(), root=root)
    _apply_report, answers = apply(manifest, store, root=root)

    serving = AnsweredFromQueue(Specialist(manifest, store), answers)
    report = run_evaluation(manifest, store, specialist=serving, persist=False)

    assert serving.served > 0
    assert serving.served + serving.fell_through == report.total


def test_the_whole_split_round_trips(built, tmp_path):
    """plan here -> answer there -> validate and score here."""
    manifest, store = built

    planned = plan(manifest, store, root=tmp_path)
    assert planned.written > 0
    assert queue_status(manifest.id, tmp_path)["pending"] == planned.written

    model = Model()
    worker([manifest], provider=model, root=tmp_path)
    assert queue_status(manifest.id, tmp_path)["pending"] == 0

    report, answers = apply(manifest, store, root=tmp_path)
    assert report.applied == planned.written
    assert all(a.llm["provider"] == "ollama" for a in answers.values())


def test_prompt_hash_is_stable_and_sensitive():
    assert prompt_hash("a", "b") == prompt_hash("a", "b")
    assert prompt_hash("a", "b") != prompt_hash("a", "b ")
