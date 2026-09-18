"""Native Ollama provider and `foundry doctor`.

The provider talks to Ollama's own API rather than its OpenAI-compatible shim,
because that is the surface every other Ollama workload on a machine already
uses, and the only one an older daemon has at all.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from foundry.llm import get_provider
from foundry.llm.ollama import OllamaProvider, host


@pytest.fixture
def daemon(monkeypatch):
    """A faithful stand-in for the native Ollama API."""
    state = {"chat": True, "requests": [], "models": ["llama3.1:8b"]}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body):
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/api/tags":
                return self._send(200, {"models": [{"name": n} for n in state["models"]]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
            state["requests"].append((self.path, request))

            if self.path == "/api/chat" and not state["chat"]:
                return self._send(404, {"error": "not found"})
            if self.path not in ("/api/chat", "/api/generate"):
                return self._send(404, {"error": "not found"})
            if request.get("model") not in state["models"]:
                return self._send(404, {"error": "model '%s' not found" % request.get("model")})
            if self.path == "/api/chat":
                return self._send(200, {
                    "model": request["model"],
                    "message": {"role": "assistant", "content": "chat reply"},
                    "prompt_eval_count": 120, "eval_count": 7, "done_reason": "stop",
                })
            return self._send(200, {
                "model": request["model"], "response": "generate reply",
                "prompt_eval_count": 120, "eval_count": 7, "done_reason": "stop",
            })

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:%d" % server.server_address[1])
    yield state
    server.shutdown()


# -- host resolution ----------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("localhost:11434", "http://localhost:11434"),
        ("127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://box.local:11434", "http://box.local:11434"),
        ("box.local", "http://box.local:11434"),
        ("11434", "http://localhost:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
    ],
)
def test_ollama_host_is_resolved_the_way_the_cli_does(monkeypatch, value, expected):
    """A daemon the rest of the machine can find must be findable here too."""
    monkeypatch.setenv("OLLAMA_HOST", value)
    assert host() == expected


def test_default_host_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("FOUNDRY_OLLAMA_HOST", raising=False)
    assert host() == "http://localhost:11434"


def test_foundry_specific_override_wins(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://shared:11434")
    monkeypatch.setenv("FOUNDRY_OLLAMA_HOST", "http://dedicated:11434")
    assert host() == "http://dedicated:11434"


# -- the provider -------------------------------------------------------


def test_it_uses_the_native_chat_endpoint(daemon):
    response = OllamaProvider("llama3.1:8b").complete("sys", "user")

    assert response.ok and response.text == "chat reply"
    assert response.prompt_tokens == 120 and response.completion_tokens == 7
    assert daemon["requests"][-1][0] == "/api/chat"


def test_streaming_is_always_disabled(daemon):
    """The native API streams by default; a streamed body parses as garbage."""
    OllamaProvider("llama3.1:8b").complete("sys", "user")
    assert daemon["requests"][-1][1]["stream"] is False


def test_generation_options_use_native_names(daemon):
    """Ollama takes num_predict, not max_tokens."""
    OllamaProvider("llama3.1:8b").complete("sys", "user", temperature=0.3, max_tokens=256)
    assert daemon["requests"][-1][1]["options"] == {"temperature": 0.3, "num_predict": 256}


def test_an_older_daemon_falls_back_to_generate(daemon):
    """Daemons before /api/chat still have /api/generate -- and still work."""
    daemon["chat"] = False
    provider = OllamaProvider("llama3.1:8b")
    response = provider.complete("sys", "user")

    assert response.ok and response.text == "generate reply"
    assert daemon["requests"][-1][0] == "/api/generate"
    assert provider._use_generate


def test_the_missing_route_is_probed_only_once(daemon):
    """Not once per answer."""
    daemon["chat"] = False
    provider = OllamaProvider("llama3.1:8b")
    provider.complete("sys", "user")
    daemon["requests"].clear()
    provider.complete("sys", "user")

    assert [p for p, _ in daemon["requests"]] == ["/api/generate"]


def test_an_unpulled_model_says_how_to_pull_it(daemon):
    response = OllamaProvider("mistral:7b").complete("sys", "user")

    assert not response.ok
    assert "ollama pull mistral:7b" in response.error


def test_an_unreachable_daemon_names_the_address(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    response = OllamaProvider().complete("sys", "user")

    assert not response.ok
    assert "127.0.0.1:1" in response.error


def test_the_registry_knows_it():
    provider = get_provider("ollama", "llama3.1:8b")
    assert provider.name == "ollama"
    assert provider.model == "llama3.1:8b"


def test_it_is_wrapped_for_fallback(manifest):
    from foundry.llm import FallbackProvider, provider_for

    manifest.specialist.llm_provider = "ollama"
    assert isinstance(provider_for(manifest), FallbackProvider)


# -- doctor -------------------------------------------------------------


def test_doctor_reports_a_healthy_daemon(daemon):
    from foundry.doctor import check_ollama

    states = {c.name: c.state for c in check_ollama("llama3.1:8b")}
    assert states["Ollama daemon"] == "ok"
    assert states["Model llama3.1:8b"] == "ok"


def test_doctor_fails_on_a_model_that_was_never_pulled(daemon):
    from foundry.doctor import check_ollama

    checks = {c.name: c for c in check_ollama("mistral:7b")}
    assert checks["Model mistral:7b"].state == "fail"
    assert "ollama pull mistral:7b" in checks["Model mistral:7b"].fix


def test_doctor_warns_when_the_chat_route_is_absent(daemon):
    from foundry.doctor import check_ollama

    daemon["chat"] = False
    checks = {c.name: c for c in check_ollama("llama3.1:8b")}
    assert checks["Native API /api/chat"].state == "warn"
    assert "/api/generate" in checks["Native API /api/chat"].detail


def test_doctor_tells_you_how_to_start_a_missing_daemon(monkeypatch):
    from foundry.doctor import check_ollama

    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    checks = check_ollama("llama3.1:8b")
    assert checks[0].state == "fail"
    assert "ollama serve" in checks[0].fix


def test_doctor_checks_the_corpus_is_built(built):
    from foundry.doctor import check_knowledge_area

    manifest, _store = built
    states = {c.name: c.state for c in check_knowledge_area(manifest.id)}
    assert states["%s corpus" % manifest.id] == "ok"
    assert states["%s semantic index" % manifest.id] == "ok"


def test_doctor_reports_an_unbuilt_corpus_with_the_command_to_fix_it(manifest):
    from foundry.doctor import check_knowledge_area

    checks = {c.name: c for c in check_knowledge_area(manifest.id)}
    corpus = checks["%s corpus" % manifest.id]
    assert corpus.state == "fail"
    assert "foundry build %s" % manifest.id in corpus.fix


def test_doctor_exits_non_zero_only_on_a_real_failure(built, monkeypatch):
    from foundry.doctor import run_doctor

    manifest, _store = built
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    report = run_doctor(knowledge_areas=[manifest.id])
    assert report.failed
    assert any("Ollama" in c.name for c in report.failed)
