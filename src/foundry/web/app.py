"""Public evidence pages, the ask UI and the JSON API (§15, §16, §19).

Built on the standard library's ``http.server``. No framework, because the
whole surface is a dozen routes and a framework would be the largest dependency
in a project whose defining constraint is that it must run anywhere for almost
nothing.

Three things live here:

* the **public evidence page** (§15) -- what a knowledge area covers, where it
  came from, how it scores, what changed, what it does not know, and what
  people have found wrong with it;
* **"try to break it"** (§16) -- an ask box next to a challenge form, because
  expert scrutiny is meant to be part of the product rather than something that
  happens in private;
* the **API** (§19), returning the documented response shape.

Multi-tenancy (§18) is present in its cheapest useful form: a knowledge area
declares a visibility and a tenant, and non-public areas are served only to a
request carrying a token mapped to that tenant. Real isolation later reuses the
same layout unchanged.
"""

from __future__ import annotations

import json
import os
import traceback
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import paths
from ..evidence import knowledge_area_report, source_detail, transcript
from ..feedback import add_challenge, list_challenges
from ..manifest import Manifest, list_knowledge_areas
from ..specialist import Specialist
from ..storage import Store

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 64 * 1024

# Routes a browser on another origin may call. Deliberately the read/ask API
# only: the HTML form POSTs that write challenges are never cross-origin, so a
# page you did not open cannot record anything against your knowledge base.
_CORS_SAFE_PREFIXES = ("/api/", "/knowledge/", "/healthz")


def cors_origins() -> list[str]:
    """Origins allowed to call the JSON API from a browser.

    A published evidence page is static, so its ask box has to reach a running
    instance somewhere -- normally the reader's own machine. That is a
    cross-origin request, which needs this.

    The default is permissive because the server binds to 127.0.0.1 and serves
    only public knowledge-base content: there is nothing here that a hostile
    page could read that it could not fetch from the sources directly. Narrow
    it with FOUNDRY_CORS_ORIGINS when the instance is exposed beyond localhost.
    """
    raw = os.environ.get("FOUNDRY_CORS_ORIGINS", "*")
    return [o.strip() for o in raw.split(",") if o.strip()]


def cors_headers(origin: str | None, path: str) -> list[tuple[str, str]]:
    if not any(path.startswith(prefix) for prefix in _CORS_SAFE_PREFIXES):
        return []
    allowed = cors_origins()
    if "*" in allowed:
        value = "*"
    elif origin and origin in allowed:
        value = origin
    else:
        return []
    return [
        ("Access-Control-Allow-Origin", value),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Foundry-Token"),
        ("Access-Control-Max-Age", "600"),
    ]


def _tenant_tokens() -> dict[str, str]:
    """``FOUNDRY_API_TOKENS='tok1:acme,tok2:globex'`` -> {token: tenant}.

    Tokens live in the environment, never in a manifest, so a knowledge area
    stays publishable as plain text.
    """
    raw = os.environ.get("FOUNDRY_API_TOKENS", "")
    out = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            token, tenant = pair.split(":", 1)
            out[token.strip()] = tenant.strip()
    return out


@dataclass
class Registry:
    """Loaded knowledge areas, shared across requests.

    One process serves every knowledge area (§17): adding the tenth one adds a
    SQLite file, not a server.
    """

    manifests: dict[str, Manifest]

    @classmethod
    def load(cls) -> "Registry":
        manifests = {}
        for ka_id in list_knowledge_areas():
            try:
                manifests[ka_id] = Manifest.load(ka_id)
            except Exception:
                continue
        return cls(manifests=manifests)

    def visible(self, tenant: str | None) -> list[Manifest]:
        return [m for m in self.manifests.values() if self.may_view(m, tenant)]

    @staticmethod
    def may_view(manifest: Manifest, tenant: str | None) -> bool:
        if manifest.visibility == "public":
            return True
        return tenant is not None and tenant == manifest.tenant


def _environment(base_url: str = "", static: bool = False):
    """Build the template environment.

    ``base_url`` prefixes every internal link, which is what lets the same
    templates render both a live site at ``/`` and a GitHub Pages project site
    at ``/knowledge-foundry/``. ``static`` hides the forms that need a POST
    endpoint, so an exported page never offers a control that cannot work.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the web UI needs Jinja2: pip install 'knowledge-foundry[web]'"
        ) from exc
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda v: "-" if v is None else f"{float(v) * 100:.0f}%"
    env.filters["num"] = lambda v: "-" if v is None else f"{float(v):.3f}"
    prefix = base_url.rstrip("/")
    env.globals["url"] = lambda path: f"{prefix}{path}"
    env.globals["static"] = static
    return env


class Router:
    """Request handling, kept separate from the HTTP plumbing so it is testable."""

    def __init__(
        self,
        registry: Registry | None = None,
        base_url: str = "",
        static: bool = False,
    ) -> None:
        self.registry = registry or Registry.load()
        self.base_url = base_url
        self.static = static
        self.env = _environment(base_url=base_url, static=static)

    # -- helpers ---------------------------------------------------------

    def render(self, template: str, **context) -> bytes:
        context.setdefault("areas", list(self.registry.manifests.values()))
        return self.env.get_template(template).render(**context).encode("utf-8")

    def _open(self, manifest: Manifest) -> Store:
        return Store(manifest.id)

    def _resolve(self, ka_id: str, tenant: str | None) -> Manifest | None:
        manifest = self.registry.manifests.get(ka_id)
        if manifest is None or not self.registry.may_view(manifest, tenant):
            return None
        return manifest

    # -- routes ----------------------------------------------------------

    def index(self, tenant: str | None) -> tuple[int, str, bytes]:
        rows = []
        for manifest in self.registry.visible(tenant):
            with self._open(manifest) as store:
                rows.append({
                    "manifest": manifest,
                    "version": store.get_meta("published_version")
                    or store.get_meta("current_version", "unbuilt"),
                    "status": store.get_meta("evaluation_status", "unknown"),
                    "counts": store.counts(),
                })
        return 200, "text/html; charset=utf-8", self.render("index.html", rows=rows)

    def evidence(self, ka_id: str, tenant: str | None) -> tuple[int, str, bytes]:
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.not_found()
        with self._open(manifest) as store:
            report = knowledge_area_report(manifest, store)
        return 200, "text/html; charset=utf-8", self.render(
            "area.html", manifest=manifest, report=report
        )

    def sources(self, ka_id: str, tenant: str | None) -> tuple[int, str, bytes]:
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.not_found()
        with self._open(manifest) as store:
            report = knowledge_area_report(manifest, store)
        return 200, "text/html; charset=utf-8", self.render(
            "sources.html", manifest=manifest, report=report
        )

    def source(self, ka_id: str, source_id: str, tenant: str | None) -> tuple[int, str, bytes]:
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.not_found()
        with self._open(manifest) as store:
            detail = source_detail(store, source_id)
        if detail is None:
            return self.not_found()
        return 200, "text/html; charset=utf-8", self.render(
            "source.html", manifest=manifest, detail=detail
        )

    def evaluation(self, ka_id: str, tenant: str | None) -> tuple[int, str, bytes]:
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.not_found()
        with self._open(manifest) as store:
            report = knowledge_area_report(manifest, store)
        return 200, "text/html; charset=utf-8", self.render(
            "evaluation.html", manifest=manifest, report=report
        )

    def transcript(self, ka_id: str, tenant: str | None) -> tuple[int, str, bytes]:
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.not_found()
        with self._open(manifest) as store:
            data = transcript(store)
        return 200, "text/html; charset=utf-8", self.render(
            "transcript.html", manifest=manifest, transcript=data
        )

    def challenges(self, ka_id: str, tenant: str | None) -> tuple[int, str, bytes]:
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.not_found()
        with self._open(manifest) as store:
            rows = [dict(r) for r in list_challenges(store)]
        return 200, "text/html; charset=utf-8", self.render(
            "challenges.html", manifest=manifest, challenges=rows
        )

    def ask_page(
        self, ka_id: str, tenant: str | None, question: str = "", answer=None
    ) -> tuple[int, str, bytes]:
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.not_found()
        return 200, "text/html; charset=utf-8", self.render(
            "ask.html", manifest=manifest, question=question, answer=answer
        )

    def ask(self, ka_id: str, question: str, tenant: str | None) -> tuple[int, str, bytes]:
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.not_found()
        with self._open(manifest) as store:
            result = Specialist(manifest, store).ask(question)
        return self.ask_page(ka_id, tenant, question=question, answer=result.as_dict())

    def submit_challenge(self, ka_id: str, form: dict, tenant: str | None):
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.not_found()
        question = (form.get("question") or "").strip()
        if not question:
            return self.challenges(ka_id, tenant)
        with self._open(manifest) as store:
            add_challenge(
                manifest,
                store,
                question=question,
                claimed_problem=(form.get("problem") or "").strip(),
                submitted_by=(form.get("by") or "anonymous").strip() or "anonymous",
                expected=(form.get("expected") or "").strip(),
                system_answer=(form.get("system_answer") or "").strip(),
            )
        return self.challenges(ka_id, tenant)

    # -- API -------------------------------------------------------------

    def api_ask(self, ka_id: str, question: str, tenant: str | None) -> tuple[int, str, bytes]:
        """``POST /knowledge/{knowledge_area}/ask`` -- the §19 contract."""
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.json_error(404, f"unknown knowledge area {ka_id!r}")
        if not (question or "").strip():
            return self.json_error(400, "a 'question' field is required")
        with self._open(manifest) as store:
            result = Specialist(manifest, store).ask(question)
        payload = result.as_dict()
        return 200, "application/json", json.dumps(payload, default=str).encode()

    def api_area(self, ka_id: str, tenant: str | None) -> tuple[int, str, bytes]:
        manifest = self._resolve(ka_id, tenant)
        if manifest is None:
            return self.json_error(404, f"unknown knowledge area {ka_id!r}")
        with self._open(manifest) as store:
            report = knowledge_area_report(manifest, store)
        return 200, "application/json", json.dumps(report, default=str).encode()

    def api_index(self, tenant: str | None) -> tuple[int, str, bytes]:
        payload = [m.summary() for m in self.registry.visible(tenant)]
        return 200, "application/json", json.dumps(payload, default=str).encode()

    # -- errors ----------------------------------------------------------

    def not_found(self) -> tuple[int, str, bytes]:
        return 404, "text/html; charset=utf-8", self.render("error.html", code=404,
                                                            message="Not found")

    def json_error(self, code: int, message: str) -> tuple[int, str, bytes]:
        return code, "application/json", json.dumps({"error": message}).encode()

    # -- dispatch --------------------------------------------------------

    def handle(self, method: str, path: str, query: dict, body: dict, tenant: str | None):
        parts = [p for p in path.strip("/").split("/") if p]

        if not parts:
            return self.index(tenant)
        if parts == ["healthz"]:
            # The client uses `cors` and `version` to tell a healthy instance
            # from one running a build that predates cross-origin support --
            # a distinction that is otherwise invisible in the browser, because
            # a CORS rejection and an unreachable host both surface as the same
            # opaque network error.
            from .. import __version__

            return 200, "application/json", json.dumps({
                "ok": True,
                "version": __version__,
                "cors": True,
                "knowledge_areas": sorted(self.registry.manifests),
            }).encode()
        if parts[0] == "assets" and len(parts) == 2:
            asset = STATIC_DIR / parts[1]
            if asset.is_file() and asset.parent == STATIC_DIR:
                kind = "text/javascript" if asset.suffix == ".js" else "text/plain"
                return 200, f"{kind}; charset=utf-8", asset.read_bytes()
            return self.not_found()

        # JSON API
        if parts[0] == "api":
            if parts == ["api", "knowledge"]:
                return self.api_index(tenant)
            if len(parts) == 3 and parts[1] == "knowledge":
                return self.api_area(parts[2], tenant)
            return self.json_error(404, "unknown API route")
        if parts[0] == "knowledge" and len(parts) == 3 and parts[2] == "ask":
            if method != "POST":
                return self.json_error(405, "POST required")
            return self.api_ask(parts[1], str(body.get("question", "")), tenant)

        # HTML
        if parts[0] == "k" and len(parts) >= 2:
            ka_id = parts[1]
            rest = parts[2:]
            if not rest:
                return self.evidence(ka_id, tenant)
            if rest == ["sources"]:
                return self.sources(ka_id, tenant)
            if len(rest) == 2 and rest[0] == "source":
                return self.source(ka_id, rest[1], tenant)
            if rest == ["evaluation"]:
                return self.evaluation(ka_id, tenant)
            if rest == ["transcript"]:
                return self.transcript(ka_id, tenant)
            if rest == ["challenges"]:
                if method == "POST":
                    return self.submit_challenge(ka_id, body, tenant)
                return self.challenges(ka_id, tenant)
            if rest == ["ask"]:
                if method == "POST":
                    return self.ask(ka_id, str(body.get("question", "")), tenant)
                return self.ask_page(ka_id, tenant, question=str(query.get("q", "")))
        return self.not_found()


class Handler(BaseHTTPRequestHandler):
    server_version = "KnowledgeFoundry/0.1"
    router: Router

    def _tenant(self) -> str | None:
        tokens = _tenant_tokens()
        if not tokens:
            return None
        header = self.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        token = token or self.headers.get("X-Foundry-Token", "").strip()
        return tokens.get(token)

    def _respond(self, status: int, content_type: str, payload: bytes,
                 path: str = "") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for header, value in cors_headers(self.headers.get("Origin"), path):
            self.send_header(header, value)
        # A knowledge system that can be framed can be misattributed.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _dispatch(self, method: str) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        body: dict = {}
        if method == "POST":
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._respond(413, "application/json",
                              json.dumps({"error": "request body too large"}).encode(),
                              path=parsed.path)
                return
            raw = self.rfile.read(length) if length else b""
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if content_type == "application/json":
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    self._respond(400, "application/json",
                                  json.dumps({"error": "invalid JSON body"}).encode(),
                                  path=parsed.path)
                    return
            else:
                body = {
                    k: v[0]
                    for k, v in urllib.parse.parse_qs(raw.decode("utf-8", "replace")).items()
                }
        try:
            status, content_type, payload = self.router.handle(
                method, parsed.path, query, body, self._tenant()
            )
        except Exception:  # pragma: no cover - defensive
            traceback.print_exc()
            self._respond(500, "application/json",
                          json.dumps({"error": "internal error"}).encode(),
                          path=parsed.path)
            return
        self._respond(status, content_type, payload, path=parsed.path)

    def do_OPTIONS(self) -> None:
        """CORS preflight, which a cross-origin JSON POST always sends first."""
        parsed = urllib.parse.urlsplit(self.path)
        headers = cors_headers(self.headers.get("Origin"), parsed.path)
        self.send_response(204 if headers else 405)
        for header, value in headers:
            self.send_header(header, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}")


def build_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"router": Router()})
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = build_server(host, port)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"Knowledge Foundry serving on http://{shown}:{port}")
    print()
    print("  Ask here (same origin, always works):")
    for ka_id in sorted(Registry.load().manifests):
        print(f"    http://{shown}:{port}/k/{ka_id}/ask")
    print()
    print(f"  The published page at any origin will also find this instance at")
    print(f"  http://localhost:{port} — reload it and the ask box goes live.")
    print()
    print(f"  API: POST http://{shown}:{port}/knowledge/<knowledge-area>/ask")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
