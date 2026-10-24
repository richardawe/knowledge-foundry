"""Research an instance on request: the whole flow behind one call, one contract.

Another system -- a product's research phase, say -- hands over a template, the
parameters that identify the instance, and the claims it has drafted about it.
This renders the knowledge area, collects its sources, builds it, answers the
template's research questions (register fields first), verifies every claim,
and returns one JSON document in the ``foundry.research/v1`` contract.

The contract carries two things a caller must not have to reconstruct:

* **Visibility.** Every piece of evidence says whether it may be shown to
  anyone (``public``: registers, the instance's own site) or only to whoever the
  instance belongs to (``claimant_only``: press, reviews, community). A caller
  that renders the result cannot show a third-party finding by accident.
* **Provenance.** Every answer and verdict names what produced it: register
  fields, extraction, the sentence rule, or a named model.

``serve`` exposes it over HTTP as an asynchronous job, because a run takes
tens of seconds of polite crawling and should not hold a request open.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue

import yaml

from . import paths

CONTRACT = "foundry.research/v1"
PUBLIC_TYPES = frozenset({"official_record", "self_published"})
ROLE_EMAIL = re.compile(r"^(sales|support|help|contact|info|hello|privacy|security|press|media|"
                        r"careers|jobs|legal|billing|enquiries|inquiries|dpo|abuse|team|customer|"
                        r"service|partners|investors|ir|accounts)", re.I)


class ResearchError(ValueError):
    """A request the contract does not allow."""


def visibility(source_type: str) -> str:
    return "public" if source_type in PUBLIC_TYPES else "claimant_only"


def validate_request(request: dict) -> dict:
    if request.get("contract", CONTRACT) != CONTRACT:
        raise ResearchError(f"unsupported contract {request.get('contract')!r}; expected {CONTRACT}")
    params = request.get("params") or {}
    if not isinstance(params, dict) or not params:
        raise ResearchError("params is required, e.g. {\"domain\": \"example.com\", \"name\": \"Example\"}")
    claims = request.get("claims") or []
    if not isinstance(claims, list):
        raise ResearchError("claims must be a list")
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict) or not str(claim.get("text", "")).strip():
            raise ResearchError(f"claims[{index}] needs a non-empty 'text'")
        claim.setdefault("id", f"c{index + 1}")
    return {"template": request.get("template", "company"), "params": params,
            "claims": claims, "options": request.get("options") or {}}


def _site_facts(facts: dict) -> dict:
    site = facts.get("site", {})
    return {
        "domain": site.get("domain"),
        "redirected_from": site.get("redirected_from"),
        "pages_read": site.get("pages_read", 0),
        "page_uris": site.get("page_uris", []),
        # Role addresses only: a privacy policy naming a person's mailbox is not
        # a contact route, and the result may be shown publicly.
        "emails": [e for e in site.get("emails", []) if ROLE_EMAIL.match(e["value"])],
        "phones": site.get("phones", []),
        "probes": site.get("probes", {}),
        "declared_names": [d["name"] for d in facts.get("declared_names", [])],
    }


def run(request: dict) -> dict:
    """Execute one research request; return the v1 result document."""
    from .collect import run_collectors, write_register
    from .ingest import ingest_knowledge_area
    from .instantiate import instantiate
    from .manifest import Manifest
    from .records import answer_from_fields, read_fields
    from .retrieval import build_semantic_index
    from .specialist import Specialist
    from .storage import Store
    from .verify import entailment_from_env, verify_claims

    req = validate_request(request)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timings: dict[str, float] = {}

    t0 = time.monotonic()
    directory = instantiate(req["template"], req["params"], force=True)
    config = yaml.safe_load((directory / "collectors.yaml").read_text(encoding="utf-8")) or {}
    report = run_collectors(config, req["params"])
    write_register(directory, report, req["params"])
    timings["collect"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    manifest = Manifest.load(directory)
    with Store(manifest.id) as store:
        ingest = ingest_knowledge_area(manifest, store)
        build_semantic_index(manifest, store)
        timings["build"] = round(time.monotonic() - t0, 1)

        t0 = time.monotonic()
        bank_path = directory / "research.yaml"
        bank = (yaml.safe_load(bank_path.read_text(encoding="utf-8")) or {}).get("questions", []) \
            if bank_path.is_file() else []
        specialist = Specialist(manifest, store)
        answers = []
        for item in bank:
            fielded = (answer_from_fields(store, manifest.id, item["question"], item["fields"])
                       if item.get("fields") else None)
            answer = fielded or specialist.ask(item["question"])
            cited = [s for s in answer.sources if s.get("cited")]
            answers.append({
                "id": item["id"], "question": item["question"], "status": answer.status,
                "answer": answer.answer,
                "produced_by": (answer.llm or {}).get("provider", "unknown"),
                "sources": [{"source_type": s["source_type"], "title": s.get("title", ""),
                             "uri": s.get("uri", ""), "visibility": visibility(s["source_type"])}
                            for s in cited],
                "visibility": ("public" if all(s["source_type"] in PUBLIC_TYPES for s in cited)
                               else "claimant_only"),
            })
        timings["answer"] = round(time.monotonic() - t0, 1)

        t0 = time.monotonic()
        subject = req["params"].get("name")
        model = entailment_from_env()
        verdicts = verify_claims(store, [c["text"] for c in req["claims"]], subject=subject,
                                 entailment=model)
        timings["verify"] = round(time.monotonic() - t0, 1)
        registers = read_fields(store, None)
        sources = [dict(row) for row in store.sources()]

    claims = []
    for claim, verdict in zip(req["claims"], verdicts):
        claims.append({
            **{k: claim[k] for k in claim if k != "text"},
            "text": claim["text"],
            "verdict": verdict.verdict,
            "note": verdict.note,
            "support": [{**s.as_dict(), "visibility": visibility(s.source_type)}
                        for s in verdict.support],
        })
    counts: dict[str, int] = {}
    for source in sources:
        counts[source["source_type"]] = counts.get(source["source_type"], 0) + 1
    return {
        "contract": CONTRACT,
        "knowledge_area": manifest.id,
        "params": req["params"],
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": timings,
        "identity": report.identity,
        "registers": [{"publisher": r["publisher"], "uri": r["uri"], "title": r["title"],
                       "fields": {label: values for label, values in r["fields"]},
                       "visibility": "public"} for r in registers],
        "site": _site_facts(report.facts),
        "claims": claims,
        "answers": answers,
        "sources": {"counts": counts,
                    "licences": sorted({s["licence"] for s in sources}),
                    "retired": ingest.retired,
                    "skipped": report.skipped},
        "provenance": {
            "verifier": f"sentence rule + {model.name}:{model.model}" if model else "sentence rule",
            "answers": "register fields first, then retrieval and extraction",
        },
    }


# -- HTTP service ------------------------------------------------------------

def _jobs_dir() -> Path:
    path = paths.var_dir() / "research-jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_job(job_id: str, record: dict) -> None:
    target = _jobs_dir() / f"{job_id}.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(target)


def _worker(queue: "Queue[tuple[str, dict]]", runner) -> None:
    # One job at a time: collection is deliberately slow and polite, and two
    # crawls of the same registers in parallel would only get both throttled.
    while True:
        job_id, request = queue.get()
        _write_job(job_id, {"job_id": job_id, "status": "running"})
        try:
            result = runner(request)
            _write_job(job_id, {"job_id": job_id, "status": "done", "result": result})
        except Exception as exc:  # reported to the caller, never swallowed
            _write_job(job_id, {"job_id": job_id, "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}"})
        queue.task_done()


def make_handler(queue, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {"error": "missing or wrong bearer token"})
                return False
            return True

        def do_POST(self):  # noqa: N802 (http.server naming)
            if not self._authorised():
                return
            if self.path.rstrip("/") != "/v1/research":
                return self._send(404, {"error": "POST /v1/research"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
                validate_request(request)
            except (ValueError, ResearchError) as exc:
                return self._send(400, {"error": str(exc)})
            job_id = uuid.uuid4().hex
            _write_job(job_id, {"job_id": job_id, "status": "queued"})
            queue.put((job_id, request))
            self._send(202, {"job_id": job_id, "status": "queued",
                             "status_url": f"/v1/research/{job_id}"})

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/v1/health":
                return self._send(200, {"ok": True, "contract": CONTRACT})
            if not self._authorised():
                return
            match = re.fullmatch(r"/v1/research/([0-9a-f]{32})", self.path.rstrip("/"))
            record = _jobs_dir() / f"{match.group(1)}.json" if match else None
            if record is None or not record.is_file():
                return self._send(404, {"error": "no such job"})
            self._send(200, json.loads(record.read_text(encoding="utf-8")))

        def log_message(self, fmt, *args):  # quiet; the job files are the log
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8100, runner=run) -> None:
    token = os.environ.get("FOUNDRY_API_TOKEN")
    if host not in ("127.0.0.1", "localhost") and not token:
        raise ResearchError("refusing to listen beyond localhost without FOUNDRY_API_TOKEN")
    queue: "Queue[tuple[str, dict]]" = Queue()
    threading.Thread(target=_worker, args=(queue, runner), daemon=True).start()
    server = ThreadingHTTPServer((host, port), make_handler(queue, token))
    print(f"research service on http://{host}:{port}  (POST /v1/research, GET /v1/research/<id>)")
    server.serve_forever()
