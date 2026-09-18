"""The inference queue: deterministic Python here, model calls over there.

Answering splits cleanly in two (see ``specialist.agent``): retrieval, the
gates, prompt assembly, then validation, grounding and scoring are all
deterministic Python over the corpus. Only one step in the middle needs a
model.

This module carries that middle step across a machine boundary, through the
repository itself:

    GitHub Actions                        the machine with Ollama
    ──────────────                        ───────────────────────
    plan   build, retrieve, write
           inference/<ka>/requests/*  ──▶
                                          worker  reads them, calls Ollama,
                                   ◀──            writes responses/*
    apply  validate and score each
           reply against the evidence
           it was actually shown

No server, no tunnel, no open port: a queue of files, pushed and pulled like
any other change. It is the same shape as the sibling ``localtest`` project,
where the Mac runs the model and publishes the results as files.

**The integrity guard.** A response is only accepted if the prompt it answered
still hashes to the prompt the corpus produces now. If the corpus moved between
plan and apply, the model answered against different evidence than we would
validate it against, and a stale reply is discarded rather than scored. Without
that check this design would quietly grade answers against the wrong passages.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .manifest import Manifest
from .specialist import Specialist
from .specialist.agent import Prepared
from .storage import Store, utcnow

QUEUE_DIRNAME = "inference"


def queue_root(ka_id: str, root: Path | None = None) -> Path:
    """The queue lives in the repository, not in var/, because both sides push it."""
    base = Path(root) if root else paths.repo_root()
    return base / QUEUE_DIRNAME / ka_id


def prompt_hash(system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x1f")
    h.update(user.encode("utf-8"))
    return h.hexdigest()


def request_id(ka_id: str, question: str) -> str:
    digest = hashlib.sha256(f"{ka_id}:{question}".encode("utf-8")).hexdigest()[:20]
    return f"req_{digest}"


@dataclass
class PlanReport:
    knowledge_area: str
    written: int = 0
    unchanged: int = 0
    answered_without_model: int = 0
    already_answered: int = 0
    queue: str = ""

    def as_dict(self) -> dict:
        return {
            "knowledge_area": self.knowledge_area,
            "written": self.written,
            "unchanged": self.unchanged,
            "answered_without_model": self.answered_without_model,
            "already_answered": self.already_answered,
            "queue": self.queue,
        }

    def render(self) -> str:
        return (
            f"planned {self.knowledge_area}: {self.written} prompt(s) queued, "
            f"{self.unchanged} unchanged, {self.already_answered} already answered, "
            f"{self.answered_without_model} settled by a gate without a model\n"
            f"  queue: {self.queue}"
        )


@dataclass
class ApplyReport:
    knowledge_area: str
    applied: int = 0
    stale: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    missing_request: list[str] = field(default_factory=list)
    providers: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "knowledge_area": self.knowledge_area,
            "applied": self.applied,
            "stale": self.stale,
            "failed": self.failed,
            "missing_request": self.missing_request,
            "providers": self.providers,
        }

    def render(self) -> str:
        lines = [f"applied {self.applied} answer(s) to {self.knowledge_area}"]
        if self.providers:
            lines.append(
                "  produced by: "
                + ", ".join(f"{k} x{v}" for k, v in sorted(self.providers.items()))
            )
        if self.stale:
            lines.append(
                f"  discarded {len(self.stale)} stale answer(s): the corpus moved "
                "after they were generated"
            )
        if self.failed:
            lines.append(f"  {len(self.failed)} answer(s) recorded a model error")
        if self.missing_request:
            lines.append(f"  {len(self.missing_request)} response(s) had no matching request")
        return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> bool:
    """Write only when the content would change, so the queue does not churn."""
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def questions_for(manifest: Manifest) -> list[tuple[str, str]]:
    """(question_id, question) across the knowledge area's evaluation suites."""
    from .evaluation.dataset import all_questions, load_suites

    return [(q.id, q.question) for q in all_questions(load_suites(manifest))]


def plan(
    manifest: Manifest,
    store: Store,
    questions: list[tuple[str, str]] | None = None,
    root: Path | None = None,
    force: bool = False,
) -> PlanReport:
    """Write a prompt file for every question that needs a model.

    Questions a gate settles -- out of scope, under-specified, no sufficient
    evidence -- never reach the queue. There is nothing for a model to add to a
    refusal, and queueing them would spend inference on answers the system has
    already correctly declined to give.
    """
    directory = queue_root(manifest.id, root)
    requests_dir = directory / "requests"
    responses_dir = directory / "responses"
    report = PlanReport(knowledge_area=manifest.id, queue=str(directory))

    specialist = Specialist(manifest, store)
    version = store.get_meta("current_version", "unversioned")

    # The system prompt is the same for every request in a knowledge area, so
    # it is written once rather than copied into each of a hundred files.
    system_prompt = specialist._system_prompt
    _write_json(directory / "system.json", {
        "knowledge_area": manifest.id,
        "system": system_prompt,
        "knowledge_version": version,
    })

    for question_id, question in questions or questions_for(manifest):
        prepared = specialist.prepare(question)
        if not prepared.needs_model:
            report.answered_without_model += 1
            continue

        rid = request_id(manifest.id, question)
        sha = prompt_hash(prepared.system, prepared.user)

        if not force and (responses_dir / f"{rid}.json").exists():
            existing = _read_json(responses_dir / f"{rid}.json")
            if existing.get("prompt_sha") == sha:
                report.already_answered += 1
                continue

        payload = {
            "id": rid,
            "knowledge_area": manifest.id,
            "question_id": question_id,
            "question": question,
            "user": prepared.user,
            "temperature": prepared.temperature,
            "max_tokens": prepared.max_tokens,
            "prompt_sha": sha,
            "knowledge_version": version,
            "created_at": utcnow(),
        }
        # created_at would make every run look changed, so compare without it.
        previous = _read_json(requests_dir / f"{rid}.json")
        if previous:
            comparable = {k: v for k, v in previous.items() if k != "created_at"}
            if comparable == {k: v for k, v in payload.items() if k != "created_at"}:
                report.unchanged += 1
                continue
        if _write_json(requests_dir / f"{rid}.json", payload):
            report.written += 1
    return report


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def system_prompt_for(ka_id: str, root: Path | None = None) -> str:
    """The shared system prompt, stored once per queue."""
    return _read_json(queue_root(ka_id, root) / "system.json").get("system", "")


def pending(ka_id: str, root: Path | None = None) -> list[dict]:
    """Requests with no current answer -- the worker's to-do list."""
    directory = queue_root(ka_id, root)
    requests_dir = directory / "requests"
    responses_dir = directory / "responses"
    if not requests_dir.is_dir():
        return []

    system = system_prompt_for(ka_id, root)
    out = []
    for path in sorted(requests_dir.glob("req_*.json")):
        request = _read_json(path)
        if not request:
            continue
        response = _read_json(responses_dir / f"{request['id']}.json")
        # A response to an older prompt does not count: the evidence changed.
        if response and response.get("prompt_sha") == request.get("prompt_sha"):
            continue
        request.setdefault("system", system)
        out.append(request)
    return out


def record_response(
    ka_id: str,
    request: dict,
    response,
    worker: str = "",
    root: Path | None = None,
) -> Path:
    """Write one model reply into the queue."""
    directory = queue_root(ka_id, root) / "responses"
    payload = {
        "id": request["id"],
        "question_id": request.get("question_id", ""),
        "prompt_sha": request.get("prompt_sha", ""),
        "text": response.text,
        "provider": response.provider,
        "model": response.model,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "stop_reason": response.stop_reason,
        "error": response.error,
        "worker": worker,
        "created_at": utcnow(),
    }
    path = directory / f"{request['id']}.json"
    _write_json(path, payload)
    return path


def apply(
    manifest: Manifest,
    store: Store,
    root: Path | None = None,
) -> tuple[ApplyReport, dict]:
    """Validate and score every queued reply, returning answers by question id.

    The validation happens here, never on the worker: a machine that generates
    an answer must not also be the one that marks it.
    """
    from .llm import LLMResponse

    directory = queue_root(manifest.id, root)
    requests_dir = directory / "requests"
    responses_dir = directory / "responses"
    report = ApplyReport(knowledge_area=manifest.id)
    answers: dict[str, object] = {}

    if not responses_dir.is_dir():
        return report, answers

    specialist = Specialist(manifest, store)
    for path in sorted(responses_dir.glob("req_*.json")):
        response_data = _read_json(path)
        if not response_data:
            continue
        request = _read_json(requests_dir / f"{response_data['id']}.json")
        if not request:
            report.missing_request.append(response_data["id"])
            continue

        # Re-derive the prompt from the corpus as it stands now. If it no longer
        # matches what was answered, the evidence has moved and scoring this
        # reply would grade it against passages it never saw.
        prepared: Prepared = specialist.prepare(request["question"])
        if not prepared.needs_model:
            report.stale.append(response_data["id"])
            continue
        if prompt_hash(prepared.system, prepared.user) != response_data.get("prompt_sha"):
            report.stale.append(response_data["id"])
            continue

        llm = LLMResponse(
            text=response_data.get("text", ""),
            provider=response_data.get("provider", "unknown"),
            model=response_data.get("model", ""),
            prompt_tokens=int(response_data.get("prompt_tokens") or 0),
            completion_tokens=int(response_data.get("completion_tokens") or 0),
            stop_reason=response_data.get("stop_reason", ""),
            error=response_data.get("error"),
        )
        if llm.error:
            report.failed.append(response_data["id"])

        answer = specialist.judge(prepared, llm)
        answers[request["question"]] = answer
        report.applied += 1
        provider = llm.provider or "unknown"
        report.providers[provider] = report.providers.get(provider, 0) + 1

    return report, answers


def queue_status(ka_id: str, root: Path | None = None) -> dict:
    directory = queue_root(ka_id, root)
    requests_dir = directory / "requests"
    responses_dir = directory / "responses"
    requests = list(requests_dir.glob("req_*.json")) if requests_dir.is_dir() else []
    responses = list(responses_dir.glob("req_*.json")) if responses_dir.is_dir() else []
    return {
        "knowledge_area": ka_id,
        "queue": str(directory),
        "requests": len(requests),
        "responses": len(responses),
        "pending": len(pending(ka_id, root)),
    }


class AnsweredFromQueue:
    """A stand-in specialist that serves answers already produced elsewhere.

    The evaluation runner asks a specialist a question; this hands back the
    reply the worker generated and this machine validated, and falls through to
    a real specialist for anything the queue does not cover — questions a gate
    settled, or ones answered since the queue was written.
    """

    def __init__(self, specialist: Specialist, answers: dict) -> None:
        self.specialist = specialist
        self.answers = answers
        self.served = 0
        self.fell_through = 0

    def ask(self, question: str, top_k: int | None = None):
        answer = self.answers.get((question or "").strip())
        if answer is not None:
            self.served += 1
            return answer
        self.fell_through += 1
        return self.specialist.ask(question, top_k=top_k)


def worker(
    manifests: list[Manifest],
    provider=None,
    limit: int | None = None,
    root: Path | None = None,
    worker_name: str = "",
    on_progress=None,
) -> dict:
    """Answer queued prompts with a model. The only part that calls one.

    Runs on the machine that has the model. It does no retrieval, applies no
    gates and scores nothing: it reads a prompt, calls Ollama, writes the reply
    back. Everything that decides whether the reply is any good happens
    elsewhere, which is the point of the split.
    """
    from .llm import provider_for

    summary = {"answered": 0, "failed": 0, "by_area": {}, "errors": []}

    for manifest in manifests:
        queued = pending(manifest.id, root)
        if limit is not None:
            queued = queued[:limit]
        if not queued:
            summary["by_area"][manifest.id] = 0
            continue

        # Unwrapped and model-only on purpose: a worker that answered with the
        # keyless extractive provider -- by falling back, or because the machine
        # exports the CI runner's FOUNDRY_LLM_PROVIDER -- would fill the queue
        # with replies that are indistinguishable from the model's once applied.
        # If the model is unreachable, that is an error to report, not to paper
        # over with an answer nobody asked a model for.
        model = provider or provider_for(manifest, allow_fallback=False, require_model=True)

        answered = 0
        for request in queued:
            response = model.complete(
                request["system"],
                request["user"],
                temperature=float(request.get("temperature", 0.0)),
                max_tokens=int(request.get("max_tokens", 1200)),
            )
            record_response(manifest.id, request, response, worker=worker_name, root=root)
            if response.ok and response.text.strip():
                answered += 1
                summary["answered"] += 1
            else:
                summary["failed"] += 1
                summary["errors"].append(
                    {"id": request["id"], "error": response.error or "empty completion"}
                )
            if on_progress:
                on_progress(manifest.id, request, response)
        summary["by_area"][manifest.id] = answered
    return summary
