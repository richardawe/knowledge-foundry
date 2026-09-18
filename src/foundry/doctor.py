"""``foundry doctor`` -- diagnose the whole chain on the machine it runs on.

Every link in the chain fails in a way that looks like the others from the
outside: a daemon that is not running, a model that was never pulled, a
knowledge area that was never built, a server on a different port. This walks
each link in order and says which one is broken and what fixes it.

It exists because the interesting failures happen on someone else's laptop,
where nobody else can look.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .manifest import Manifest, list_knowledge_areas
from .storage import Store

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
_MARK = {OK: "PASS", WARN: "WARN", FAIL: "FAIL", INFO: "    "}


@dataclass
class Check:
    name: str
    state: str
    detail: str = ""
    fix: str = ""

    def render(self) -> str:
        line = f"  [{_MARK[self.state]}] {self.name}"
        if self.detail:
            line += f"\n         {self.detail}"
        if self.fix:
            line += f"\n         fix: {self.fix}"
        return line


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *checks: Check) -> None:
        self.checks.extend(checks)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.state == FAIL]

    def as_dict(self) -> dict:
        return {
            "checks": [
                {"name": c.name, "state": c.state, "detail": c.detail, "fix": c.fix}
                for c in self.checks
            ],
            "ok": not self.failed,
        }

    def render(self) -> str:
        lines = ["foundry doctor"]
        lines.extend(c.render() for c in self.checks)
        lines.append("")
        if self.failed:
            lines.append(f"{len(self.failed)} check(s) failed — see the fixes above.")
        else:
            lines.append("Everything needed to answer a question is in place.")
        return "\n".join(lines)


def _get_json(url: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_ollama(model: str | None = None) -> list[Check]:
    from .llm.ollama import host

    address = host()
    checks: list[Check] = []
    try:
        data = _get_json(f"{address}/api/tags")
    except Exception as exc:
        return [Check(
            "Ollama daemon", FAIL,
            f"not reachable at {address} ({type(exc).__name__})",
            "start it with `ollama serve`, or set OLLAMA_HOST if it listens elsewhere",
        )]

    names = [m.get("name", "") for m in data.get("models", [])]
    checks.append(Check("Ollama daemon", OK, f"reachable at {address}"))
    checks.append(Check(
        "Pulled models", OK if names else WARN,
        ", ".join(names) if names else "none pulled",
        "" if names else "ollama pull llama3.1:8b",
    ))

    if model:
        # Ollama resolves a bare name to its ":latest" tag.
        matched = any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
        checks.append(Check(
            f"Model {model}", OK if matched else FAIL,
            "available" if matched else "not pulled on this daemon",
            "" if matched else f"ollama pull {model}",
        ))

    # Which HTTP surfaces this daemon offers. The native one is what the
    # provider uses; the OpenAI shim is only informative.
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{address}/api/chat", data=b"{}", headers={"Content-Type": "application/json"},
        ), timeout=5)
        native = True
    except urllib.error.HTTPError as exc:
        native = exc.code != 404
    except Exception:
        native = False
    checks.append(Check(
        "Native API /api/chat", OK if native else WARN,
        "available" if native else "absent — the provider will use /api/generate instead",
    ))

    try:
        urllib.request.urlopen(f"{address}/v1/models", timeout=5)
        shim = True
    except urllib.error.HTTPError as exc:
        shim = exc.code != 404
    except Exception:
        shim = False
    checks.append(Check(
        "OpenAI shim /v1", INFO,
        "available" if shim else "absent (not needed — the native API is used)",
    ))
    return checks


def check_knowledge_area(ka_id: str) -> list[Check]:
    checks: list[Check] = []
    try:
        manifest = Manifest.load(ka_id)
    except Exception as exc:
        return [Check(f"Knowledge area {ka_id}", FAIL, str(exc))]

    provider = manifest.specialist.llm_provider
    model = manifest.specialist.llm_model
    checks.append(Check(
        f"{ka_id} manifest", OK,
        f"provider={provider} model={model or '-'} "
        f"fallback={'on' if manifest.specialist.llm_fallback else 'off'}",
    ))

    with Store(ka_id) as store:
        counts = store.counts()
        version = store.get_meta("current_version", "unbuilt")
        status = store.get_meta("evaluation_status", "never run")
        embedded = store.conn.execute(
            "SELECT COUNT(*) AS n FROM embeddings"
        ).fetchone()["n"]

    built = counts["chunks"] > 0
    checks.append(Check(
        f"{ka_id} corpus", OK if built else FAIL,
        f"{counts['indexed_sources']} sources, {counts['chunks']} passages, version {version}"
        if built else "not built — no passages stored",
        "" if built else f"foundry build {ka_id}",
    ))
    if built:
        checks.append(Check(
            f"{ka_id} semantic index", OK if embedded else WARN,
            f"{embedded} vectors" if embedded else "no embeddings; retrieval loses the semantic leg",
            "" if embedded else f"foundry index {ka_id}",
        ))
        checks.append(Check(f"{ka_id} evaluation", INFO, f"last run: {status}"))
    return checks


def check_server(port: int = 8000) -> list[Check]:
    url = f"http://localhost:{port}"
    try:
        data = _get_json(f"{url}/healthz")
    except Exception:
        return [Check(
            "Local server", WARN,
            f"nothing answering on {url}",
            f"foundry serve --port {port}  (needed only for the ask UI and API)",
        )]
    return [
        Check("Local server", OK,
              f"{url} serving {len(data.get('knowledge_areas', []))} knowledge area(s), "
              f"v{data.get('version', '?')}, cors={'on' if data.get('cors') else 'off'}"),
        Check("Ask in a browser", INFO,
              " ".join(f"{url}/k/{ka}/ask" for ka in data.get("knowledge_areas", [])[:2])),
    ]


def run_doctor(knowledge_areas: list[str] | None = None, port: int = 8000) -> Report:
    report = Report()
    areas = knowledge_areas or list_knowledge_areas()

    model = None
    for ka_id in areas:
        try:
            manifest = Manifest.load(ka_id)
        except Exception:
            continue
        if manifest.specialist.llm_provider in ("ollama", "openai_compatible"):
            model = manifest.specialist.llm_model
            break

    report.add(*check_ollama(model))
    for ka_id in areas:
        report.add(*check_knowledge_area(ka_id))
    report.add(*check_server(port))
    return report
