"""The research contract and its HTTP job service, without the network."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from queue import Queue

import pytest

from foundry import paths, research


def test_requests_are_validated_against_the_contract():
    with pytest.raises(research.ResearchError):
        research.validate_request({"contract": "foundry.research/v0", "params": {"name": "x"}})
    with pytest.raises(research.ResearchError):
        research.validate_request({"params": {}})
    with pytest.raises(research.ResearchError):
        research.validate_request({"params": {"domain": "x.com"}, "claims": [{"text": " "}]})
    ok = research.validate_request({"params": {"domain": "x.com", "name": "X"},
                                    "claims": [{"text": "Session replay"}]})
    assert ok["claims"][0]["id"] == "c1" and ok["template"] == "company"


def test_third_party_evidence_is_claimant_only():
    assert research.visibility("official_record") == "public"
    assert research.visibility("self_published") == "public"
    for kind in ("press", "review", "community"):
        assert research.visibility(kind) == "claimant_only"


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "var_dir", lambda: tmp_path / "var")
    queue: Queue = Queue()
    seen = []

    def runner(request):
        seen.append(request)
        return {"contract": research.CONTRACT, "params": request["params"]}

    threading.Thread(target=research._worker, args=(queue, runner), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), research.make_handler(queue, "s3cret"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown()


def _call(url, method="GET", body=None, token="s3cret"):
    request = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Authorization": f"Bearer {token}",
                                              "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_a_job_is_queued_then_answered(service):
    base, seen = service
    status, accepted = _call(f"{base}/v1/research", "POST",
                             {"params": {"domain": "x.com", "name": "X"}, "claims": []})
    assert status == 202 and accepted["status"] == "queued"
    for _ in range(50):
        status, job = _call(base + accepted["status_url"])
        if job["status"] == "done":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["result"]["params"] == {"domain": "x.com", "name": "X"}


def test_the_service_refuses_bad_tokens_and_bad_requests(service):
    base, seen = service
    assert _call(f"{base}/v1/research", "POST", {"params": {"name": "X"}}, token="wrong")[0] == 401
    assert _call(f"{base}/v1/research", "POST", {"params": {}})[0] == 400
    assert seen == []


def test_the_service_will_not_listen_publicly_without_a_token(monkeypatch):
    monkeypatch.delenv("FOUNDRY_API_TOKEN", raising=False)
    with pytest.raises(research.ResearchError):
        research.serve(host="0.0.0.0", port=0)


@pytest.mark.parametrize("kind", ["request", "result"])
def test_the_dropin_examples_satisfy_the_published_schemas(kind):
    jsonschema = pytest.importorskip("jsonschema")
    dropin = paths.repo_root() / "integrations" / "requstory" / "dropin"
    schema = json.loads((dropin / "schema" / f"{kind}.v1.json").read_text())
    example = dropin / "examples" / ("request.json" if kind == "request" else "result.public.json")
    jsonschema.validate(json.loads(example.read_text()), schema)


def test_a_public_example_carries_no_claimant_only_evidence():
    dropin = paths.repo_root() / "integrations" / "requstory" / "dropin"
    result = json.loads((dropin / "examples" / "result.public.json").read_text())
    assert all(s["visibility"] == "public" for c in result["claims"] for s in c["support"])
    assert all(a["visibility"] == "public" or not a["sources"] for a in result["answers"])
