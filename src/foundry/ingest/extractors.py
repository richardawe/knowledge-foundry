"""Extractors: bytes -> clean text, by content type.

HTML extraction uses the standard library's ``HTMLParser`` rather than
BeautifulSoup so the core install stays dependency-free. PDF extraction needs
``pypdf`` and degrades to a recorded failure rather than a crash when it is
absent -- an optional dependency must never take down the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO

from ..text import normalise

# Elements whose *text* is never document content. <head> is not listed: its
# <title> is the best available document title on most web sources.
_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "form", "iframe"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "tr", "br", "hr",
    "table", "blockquote", "pre", "dd", "dt",
}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Containers that usually hold the real article body.
_MAIN_TAGS = {"main", "article"}


class ExtractionError(RuntimeError):
    pass


@dataclass
class Extracted:
    title: str
    text: str
    extractor: str


class _HTMLTextExtractor(HTMLParser):
    """Collect readable text, preserving headings as Markdown.

    Headings are converted to ``#`` form so the chunker's section detection
    works identically on HTML and Markdown sources.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._heading: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag in _HEADING_TAGS:
            self._heading = "#" * int(tag[1])
            self.parts.append("\n\n")
            self.parts.append(self._heading + " ")
        elif tag in _BLOCK_TAGS or tag in _MAIN_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag in _HEADING_TAGS:
            self._heading = None
            self.parts.append("\n")
        elif tag in _BLOCK_TAGS or tag in _MAIN_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data.strip()
            return
        if data.strip():
            self.parts.append(data)

    def result(self) -> str:
        return "".join(self.parts)


def extract_html(content: bytes) -> Extracted:
    raw = content.decode("utf-8", "replace")
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    parser.close()
    text = normalise(parser.result())
    # Strip the navigation debris that survives on government and news sites:
    # short, repeated lines with no sentence punctuation.
    lines = [ln.rstrip() for ln in text.split("\n")]
    seen: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            seen[stripped] = seen.get(stripped, 0) + 1
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped and seen[stripped] > 2 and len(stripped) < 60 and not stripped.startswith("#"):
            continue
        kept.append(line)
    return Extracted(title=parser.title.strip(), text=normalise("\n".join(kept)), extractor="html")


def extract_pdf(content: bytes) -> Extracted:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ExtractionError(
            "PDF extraction requires the 'pdf' extra: pip install 'knowledge-foundry[pdf]'"
        ) from exc
    try:
        reader = PdfReader(BytesIO(content))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise ExtractionError(f"could not parse PDF: {exc}") from exc
    title = ""
    try:
        title = (reader.metadata.title or "") if reader.metadata else ""
    except Exception:
        title = ""
    # Page markers are kept: a citation that can name a page is far more useful
    # to someone checking a 200-page incident report.
    body = "\n\n".join(f"## Page {i + 1}\n{page.strip()}" for i, page in enumerate(pages) if page.strip())
    return Extracted(title=title.strip(), text=normalise(body), extractor="pdf")


def extract_text(content: bytes) -> Extracted:
    text = normalise(content.decode("utf-8", "replace"))
    title = ""
    for line in text.split("\n"):
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            title = stripped
            break
    return Extracted(title=title[:200], text=text, extractor="text")


def extract_json(content: bytes) -> Extracted:
    """Flatten a JSON document to readable key/value prose.

    Structured datasets (§7) become retrievable without a bespoke schema per
    dataset; the structured retriever handles precision, this handles recall.
    """
    try:
        data = json.loads(content.decode("utf-8", "replace"))
    except Exception as exc:
        raise ExtractionError(f"could not parse JSON: {exc}") from exc

    lines: list[str] = []

    def walk(node, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{prefix}[{i}]")
        else:
            lines.append(f"{prefix}: {node}")

    walk(data)
    return Extracted(title="", text=normalise("\n".join(lines)), extractor="json")


_EXTRACTORS = {
    "text/html": extract_html,
    "application/xhtml+xml": extract_html,
    "application/pdf": extract_pdf,
    "text/plain": extract_text,
    "text/markdown": extract_text,
    "application/json": extract_json,
}


def extract(content: bytes, content_type: str, uri: str = "") -> Extracted:
    """Choose an extractor by declared content type, then by sniffing."""
    handler = _EXTRACTORS.get(content_type.lower())
    if handler is None:
        head = content[:1024].lstrip()
        if head.startswith(b"%PDF"):
            handler = extract_pdf
        elif re.match(rb"\s*<(!doctype html|html)", head, re.IGNORECASE):
            handler = extract_html
        elif head.startswith(b"{") or head.startswith(b"["):
            handler = extract_json
        else:
            handler = extract_text
    return handler(content)
