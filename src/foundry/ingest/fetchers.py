"""Fetchers: get bytes for a source URI, politely and reproducibly.

Three schemes cover the MVP: ``http(s)`` for public documents, ``file`` for
local corpora and expert-supplied documents, and ``inline`` for content carried
in the source register itself.

Politeness is not optional. A knowledge lab that gets itself blocked by the
agencies publishing its primary sources has destroyed its own supply chain, so
robots.txt is respected, requests are rate limited per host, and the user agent
identifies the project.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "KnowledgeFoundry/0.1 (+https://github.com/richardawe/knowledge-foundry)"
DEFAULT_TIMEOUT = 30
MIN_HOST_INTERVAL = 1.0  # seconds between requests to the same host

_last_request: dict[str, float] = {}
_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


class FetchError(RuntimeError):
    """Raised when a source cannot be retrieved. Recorded, never swallowed."""


@dataclass
class FetchResult:
    content: bytes
    content_type: str
    final_uri: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def _throttle(host: str) -> None:
    last = _last_request.get(host)
    if last is not None:
        wait = MIN_HOST_INTERVAL - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _last_request[host] = time.monotonic()


def robots_allows(uri: str, user_agent: str = USER_AGENT) -> bool:
    """Check robots.txt, failing open only when robots.txt itself is unreachable."""
    parts = urllib.parse.urlsplit(uri)
    root = f"{parts.scheme}://{parts.netloc}"
    if root not in _robots_cache:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"{root}/robots.txt")
        try:
            request = urllib.request.Request(
                f"{root}/robots.txt", headers={"User-Agent": user_agent}
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                parser.parse(response.read().decode("utf-8", "replace").splitlines())
            _robots_cache[root] = parser
        except Exception:
            # No robots.txt, or it is unreachable: proceed, but still throttled.
            _robots_cache[root] = None
    parser = _robots_cache[root]
    return True if parser is None else parser.can_fetch(user_agent, uri)


def fetch_http(uri: str, timeout: int = DEFAULT_TIMEOUT, respect_robots: bool = True) -> FetchResult:
    parts = urllib.parse.urlsplit(uri)
    if respect_robots and not robots_allows(uri):
        raise FetchError(f"robots.txt disallows fetching {uri}")
    _throttle(parts.netloc)
    request = urllib.request.Request(
        uri,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "en",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
            return FetchResult(content, content_type or "application/octet-stream", response.geturl())
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code} fetching {uri}") from exc
    except Exception as exc:  # timeouts, DNS, TLS
        raise FetchError(f"{type(exc).__name__} fetching {uri}: {exc}") from exc


def fetch_file(uri: str) -> FetchResult:
    path = Path(urllib.parse.urlsplit(uri).path if uri.startswith("file:") else uri)
    if not path.is_file():
        raise FetchError(f"local file not found: {path}")
    suffix = path.suffix.lower()
    content_type = {
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".htm": "text/html",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".json": "application/json",
    }.get(suffix, "text/plain")
    return FetchResult(path.read_bytes(), content_type, str(path))


def fetch_inline(content: str) -> FetchResult:
    return FetchResult(content.encode("utf-8"), "text/markdown", "inline:")


def fetch(uri: str, inline_content: str | None = None, **kwargs) -> FetchResult:
    """Dispatch on URI scheme."""
    if inline_content is not None or uri.startswith("inline:"):
        if inline_content is None:
            raise FetchError(f"inline source {uri} has no 'content' field")
        return fetch_inline(inline_content)
    scheme = urllib.parse.urlsplit(uri).scheme
    if scheme in ("http", "https"):
        return fetch_http(uri, **kwargs)
    if scheme in ("file", ""):
        return fetch_file(uri)
    raise FetchError(f"unsupported URI scheme {scheme!r} for {uri}")
