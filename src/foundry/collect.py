"""Collectors: discover a knowledge area's sources instead of listing them by hand.

A hand-curated knowledge area names every source in ``sources.yaml``. That does
not scale to one knowledge area per *instance* of something -- per company, per
product, per site -- where the sources have to be found: crawled from a site,
looked up in a register, pulled from a feed. A template therefore ships a
``collectors.yaml`` and this module turns it into an ordinary source register,
with every document snapshotted inline so the build reads exactly what was
collected and nothing that changed since.

Nothing here knows what is being collected. Collector kinds are mechanical --
crawl a site, read a feed, read a JSON list, look an entity up in a register --
and the URLs, field paths, names and licences all come from configuration.

Two disciplines carry over from hand-curated ingestion and matter more here:

* **Identity before evidence.** A register lookup by name returns candidates,
  not answers. A candidate is accepted only on a normalised name match, and is
  marked *corroborated* only when the instance's own site mentions its register
  key or full legal name. Everything seen is recorded in ``identity.yaml`` so a
  wrong match can be traced and corrected by a person.
* **Absence is recorded as a finding with its scope.** "No llms.txt (HTTP 404)"
  and "no telephone number on the 25 pages read" are facts about what was read,
  and the facts document says how much was read.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

from .ingest.extractors import ExtractionError, extract
from .ingest.fetchers import FetchError, fetch_http

MIN_DOCUMENT_CHARS = 200  # the ingest pipeline rejects anything shorter

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<![\w/.-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,4}[\s.-]\d{3,4}[\s.-]\d{3,4}(?![\w/.-])")
_PHONE_CUE_RE = re.compile(r"\b(phone|tel|telephone|call|ring|fax|helpline|hotline|contact us)\b|☎|📞")
_TEL_HREF_RE = re.compile(r'href="tel:([^"]+)"', re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.S | re.I)


class CollectError(RuntimeError):
    """A collector configuration that cannot run. Source failures are recorded, not raised."""


@dataclass
class Document:
    id: str
    title: str
    uri: str
    text: str
    publisher: str
    source_type: str
    authority: int
    licence: str
    tier: str
    collector: str
    published_at: str | None = None
    tags: list[str] = field(default_factory=list)

    def as_source(self) -> dict:
        entry = {
            "id": self.id,
            "uri": self.uri,
            "title": self.title[:200],
            "publisher": self.publisher,
            "source_type": self.source_type,
            "authority": self.authority,
            "licence": self.licence,
            "tags": [self.tier, self.collector, *self.tags],
            "content": self.text,
        }
        if self.published_at:
            entry["published_at"] = self.published_at
        return entry


@dataclass
class CollectReport:
    documents: list[Document] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    identity: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for doc in self.documents:
            out[doc.tier] = out.get(doc.tier, 0) + 1
        return out


# -- small helpers -----------------------------------------------------------

def _fill(template: str, params: dict) -> str:
    """Substitute ``{name}`` placeholders; ``{name_q}`` is the URL-quoted form."""
    values = dict(params)
    for key, value in params.items():
        values[f"{key}_q"] = urllib.parse.quote(str(value))
    try:
        return template.format(**values)
    except KeyError as exc:
        raise CollectError(f"template {template!r} needs parameter {exc}") from exc


def _dig(node: Any, path: str) -> Any:
    """Follow a dotted path through dicts and lists (``hits.0.name``)."""
    for part in [p for p in path.split(".") if p]:
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return None
        if node is None:
            return None
    return node


def _flat(value: Any, skip: frozenset[str] = frozenset()) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return ", ".join(_flat(v, skip) for k, v in value.items() if k not in skip and _flat(v, skip))
    if isinstance(value, list):
        return ", ".join(_flat(v) for v in value if _flat(v))
    return str(value).strip()


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAG_RE.sub(" ", text or ""))).strip()


def _date(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError):
        pass
    match = re.match(r"\d{4}-\d{2}-\d{2}", value)
    return match.group(0) if match else None


def normalise_name(name: str, drop_words: set[str]) -> str:
    """Lower-case, strip punctuation and legal-form words, collapse spaces."""
    words = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower()).split()
    return " ".join(w for w in words if w not in drop_words)


def _fetch_text(uri: str, retries: int = 1) -> tuple[bytes, str, str]:
    """Fetch, retrying once on a server error: a register's 502 is not its answer."""
    for attempt in range(retries + 1):
        try:
            result = fetch_http(uri)
            return result.content, result.content_type, result.final_uri
        except FetchError as exc:
            if attempt == retries or not re.search(r"HTTP 5\d\d|timed out|Timeout", str(exc)):
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


# -- link discovery for site crawls ------------------------------------------

class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None


def _same_site(host: str, domain: str) -> bool:
    host = host.lower().split(":")[0]
    return host == domain or host.endswith("." + domain)


# -- collector kinds ----------------------------------------------------------

def _base(cfg: dict, report: CollectReport) -> dict:
    return {
        "source_type": cfg["source_type"],
        "authority": int(cfg.get("authority", 3)),
        "licence": cfg.get("licence", "unspecified"),
        "tier": cfg.get("tier", cfg["source_type"]),
        "collector": cfg["id"],
    }


def collect_site(cfg: dict, params: dict, report: CollectReport) -> None:
    """Crawl one site breadth-first, preferring pages whose links name a wanted topic."""
    domain = _fill(cfg["domain"], params).lower()
    start = _fill(cfg.get("start", "https://{domain}/"), {**params, "domain": domain})
    max_pages = int(cfg.get("max_pages", 20))
    prefer = [p.lower() for p in cfg.get("prefer", [])]
    base = _base(cfg, report)

    redirected_from: str | None = None
    tel_links: list[tuple[str, str]] = []
    queue: list[tuple[float, str]] = [(0.0, start)]
    seen: set[str] = set()
    pages: list[tuple[str, str, str, str]] = []  # (uri, title, text, markup as text)
    while queue and len(pages) < max_pages:
        queue.sort(key=lambda item: item[0])
        _, uri = queue.pop(0)
        key = uri.split("#")[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        try:
            content, ctype, final = _fetch_text(uri)
        except FetchError as exc:
            report.skipped.append({"collector": cfg["id"], "uri": uri, "reason": str(exc)})
            continue
        final_host = urllib.parse.urlsplit(final).netloc.lower().split(":")[0]
        if not _same_site(final_host, domain):
            if not pages and not redirected_from:
                # The start page itself moved: the subject now lives elsewhere
                # (a rebrand, an acquisition). Follow it once, and say so.
                redirected_from = domain
                domain = final_host.removeprefix("www.")
            else:
                report.skipped.append({"collector": cfg["id"], "uri": uri,
                                       "reason": f"redirected off-site to {final}"})
                continue
        try:
            extracted = extract(content, ctype, final)
        except ExtractionError as exc:
            report.skipped.append({"collector": cfg["id"], "uri": uri, "reason": str(exc)})
            continue
        # Contact details are read from the markup with tags replaced by spaces:
        # text extraction can glue a word to the address that follows it.
        markup = (_strip_html(_SCRIPT_RE.sub(" ", content.decode("utf-8", "replace")))
                  if "html" in ctype else extracted.text)
        pages.append((final, extracted.title or final, extracted.text, markup))
        if "html" in ctype:
            for raw in _TEL_HREF_RE.findall(content.decode("utf-8", "replace")):
                tel_links.append((final, urllib.parse.unquote(raw).strip()))
        if "html" in ctype:
            parser = _Links()
            try:
                parser.feed(content.decode("utf-8", "replace"))
            except Exception:  # malformed markup costs links, not the page
                pass
            for href, label in parser.links:
                absolute = urllib.parse.urljoin(final, href)
                parts = urllib.parse.urlsplit(absolute)
                if parts.scheme not in ("http", "https") or not _same_site(parts.netloc, domain):
                    continue
                if re.search(r"\.(png|jpe?g|gif|svg|webp|zip|mp4|css|js|ico|xml)$", parts.path, re.I):
                    continue
                haystack = f"{parts.path} {label}".lower()
                hits = sum(1 for word in prefer if word in haystack)
                depth = parts.path.strip("/").count("/")
                queue.append((depth - 3 * hits, urllib.parse.urlunsplit(parts._replace(fragment=""))))

    for index, (uri, title, text, _) in enumerate(pages, start=1):
        if len(text.strip()) < MIN_DOCUMENT_CHARS:
            report.skipped.append({"collector": cfg["id"], "uri": uri,
                                   "reason": f"only {len(text.strip())} characters of text"})
            continue
        report.documents.append(Document(
            id=f"{cfg['id']}-{index:02d}", title=title, uri=uri, text=text,
            publisher=_fill(cfg.get("publisher", "{domain}"), {**params, "domain": domain}),
            **base))

    # Rule-based facts about the site as read. Each absence states its scope.
    facts: dict[str, Any] = {"pages_read": len(pages), "page_uris": [p[0] for p in pages],
                             "domain": domain, "redirected_from": redirected_from}
    emails: dict[str, str] = {}
    phones: dict[str, str] = {}
    for uri, _, _, text in pages:
        for email in _EMAIL_RE.findall(text):
            if not re.search(r"\.(png|jpe?g|gif|svg|webp)$", email, re.I):
                emails.setdefault(email.lower(), uri)
        for match in _PHONE_RE.finditer(text):
            phone = match.group(0).strip()
            digits = re.sub(r"\D", "", phone)
            # A run of digits is a telephone number only where the page says so:
            # clause numbers, IDs and prices match the shape too.
            before = text[max(0, match.start() - 40):match.start()].lower()
            if (9 <= len(digits) <= 15 and not re.fullmatch(r"(19|20)\d{6,}", digits)
                    and (phone.startswith("+") or _PHONE_CUE_RE.search(before))):
                phones.setdefault(phone, uri)
    for uri, raw in tel_links:
        phones.setdefault(raw, uri)
    facts["emails"] = [{"value": k, "found_on": v} for k, v in list(emails.items())[:10]]
    facts["phones"] = [{"value": k, "found_on": v} for k, v in list(phones.items())[:10]]
    facts["probes"] = {}
    home = pages[0][0] if pages else start  # probe the host that actually serves the site
    declared_sitemaps: list[str] = []
    for probe in cfg.get("probes", []):
        uri = urllib.parse.urljoin(home, probe)
        try:
            content, ctype, final = _fetch_text(uri)
            body = content.decode("utf-8", "replace")
            looks_like_html = "<html" in body[:500].lower() and not probe.endswith((".html", "/"))
            facts["probes"][probe] = ({"present": False, "detail": "served an HTML page, not the file"}
                                      if looks_like_html else
                                      {"present": True, "detail": f"{len(body.splitlines())} lines", "uri": final})
            if probe.endswith("robots.txt") and not looks_like_html:
                declared_sitemaps = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", body)
        except FetchError as exc:
            facts["probes"][probe] = {"present": False, "detail": str(exc)}
        # A sitemap is wherever robots.txt says it is, not only /sitemap.xml.
        if probe.endswith("sitemap.xml") and not facts["probes"][probe]["present"] and declared_sitemaps:
            facts["probes"][probe] = {"present": True, "uri": declared_sitemaps[0],
                                      "detail": f"declared in robots.txt ({len(declared_sitemaps)} listed)"}
    report.facts[cfg["id"]] = facts

    lines = [f"Facts recorded by rule from {len(pages)} pages of {domain} read on "
             f"{datetime.now(timezone.utc).date().isoformat()}.", ""]
    if redirected_from:
        lines.append(f"{redirected_from} redirects to {domain}; the pages read are on {domain}.")
    if emails:
        lines += ["Email addresses published on the site:"] + [
            f"- {k} (on {v})" for k, v in list(emails.items())[:10]]
    else:
        lines.append(f"No email address appears on any of the {len(pages)} pages read.")
    if phones:
        lines += ["Telephone numbers published on the site:"] + [
            f"- {k} (on {v})" for k, v in list(phones.items())[:10]]
    else:
        lines.append(f"No telephone number appears on any of the {len(pages)} pages read.")
    for probe, result in facts["probes"].items():
        lines.append(f"{probe}: {'present' if result['present'] else 'not present'} ({result['detail']}).")
    lines.append("Pages read: " + ", ".join(p[0] for p in pages))
    report.documents.append(Document(
        id=f"{cfg['id']}-facts", title=f"Site facts for {domain}", uri=start,
        text="\n".join(lines), publisher="rule-based site check", **base))


def collect_feed(cfg: dict, params: dict, report: CollectReport) -> None:
    """Read an RSS feed; each item becomes one document."""
    uri = _fill(cfg["url"], params)
    try:
        content, _, _ = _fetch_text(uri)
        root = ET.fromstring(content)
    except (FetchError, ET.ParseError) as exc:
        report.skipped.append({"collector": cfg["id"], "uri": uri, "reason": str(exc)})
        return
    base = _base(cfg, report)
    mention = str(params.get(cfg.get("require_mention", ""), "")).lower()
    taken = 0
    for item in root.iter("item"):
        if taken >= int(cfg.get("take", 10)):
            break
        title = _strip_html(item.findtext("title") or "")
        link = item.findtext("link") or ""
        unwrap = cfg.get("unwrap_param")
        if unwrap:
            inner = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query).get(unwrap)
            link = inner[0] if inner else link
        description = _strip_html(item.findtext("description") or "")
        when = _date(item.findtext("pubDate"))
        publisher = urllib.parse.urlsplit(link).netloc.removeprefix("www.") or "unknown"
        if mention and mention not in f"{title} {description}".lower():
            report.skipped.append({"collector": cfg["id"], "uri": link, "reason": "does not mention the subject"})
            continue
        # Publisher, date and link live in the source's fields, not its text:
        # a hostname in the body ("finance.yahoo.com") would match claims.
        text = f"{title}.\n\n{description}"
        if len(text) < MIN_DOCUMENT_CHARS:
            report.skipped.append({"collector": cfg["id"], "uri": link, "reason": "item too short"})
            continue
        taken += 1
        item_base = dict(base)
        # A press release is the subject speaking, whoever carries it: an item
        # on a wire service, or one bearing a wire's dateline, is reclassified.
        if (publisher in set(cfg.get("self_published_hosts", []))
                or any(re.search(m, f"{link}\n{text}", re.I)
                       for m in cfg.get("self_published_markers", []))):
            item_base.update(source_type="self_published", tier="self_published")
        report.documents.append(Document(
            id=f"{cfg['id']}-{taken:02d}", title=title, uri=link, text=text,
            publisher=publisher, published_at=when, **item_base))


def collect_json_items(cfg: dict, params: dict, report: CollectReport) -> None:
    """Read a JSON list from an API; each item becomes one document."""
    uri = _fill(cfg["url"], params)
    try:
        content, _, _ = _fetch_text(uri)
        data = json.loads(content)
    except (FetchError, ValueError) as exc:
        report.skipped.append({"collector": cfg["id"], "uri": uri, "reason": str(exc)})
        return
    base = _base(cfg, report)
    fields = cfg["fields"]
    mention = str(params.get(cfg.get("require_mention", ""), "")).lower()
    taken = 0
    for item in _dig(data, cfg.get("items", "")) or []:
        if taken >= int(cfg.get("take", 10)):
            break
        body = next((_strip_html(_flat(_dig(item, p))) for p in fields["text"] if _dig(item, p)), "")
        title = _strip_html(_flat(_dig(item, fields.get("title", "")))) or body[:80]
        link = _fill(fields["link"], {k: _flat(v) for k, v in item.items() if not isinstance(v, (dict, list))})
        when = _date(_flat(_dig(item, fields.get("date", ""))))
        if mention and mention not in f"{title} {body}".lower():
            continue
        text = f"{title}.\n\n{body}"
        if len(body) < 120:
            continue
        taken += 1
        report.documents.append(Document(
            id=f"{cfg['id']}-{taken:02d}", title=f"{cfg.get('title_prefix', '')}{title}",
            uri=link, text=text, publisher=cfg.get("publisher", "unknown"), published_at=when, **base))


def _candidates(cfg: dict, content: bytes) -> list[dict]:
    if cfg.get("format", "json") == "json":
        data = json.loads(content)
        spec = cfg["candidates"]
        out = []
        for item in _dig(data, spec.get("items", "")) or []:
            out.append({"key": _flat(_dig(item, spec["key"])), "name": _flat(_dig(item, spec["name"]))})
        return out
    pattern = re.compile(cfg["candidates"]["pattern"], re.S)
    html = content.decode("utf-8", "replace")
    return [{"key": m.group("key"), "name": _strip_html(m.group("name"))} for m in pattern.finditer(html)]


def _render_json(data: Any, keep: list[dict]) -> str:
    lines = []
    for spec in keep:
        value = _flat(_dig(data, spec["path"]), frozenset(spec.get("skip", [])))
        if value:
            lines.append(f"{spec['label']}: {value}")
    return "\n".join(lines)


def _render_page(text: str, markup: str, detail: dict) -> str:
    """Turn a register's web page into ``Label: value`` lines, dropping the page furniture.

    ``fields`` are single values found by pattern in the page text. ``records``
    are repeated blocks (one per officer, say), read from the markup because
    text extraction drops the labels that say which blocks are current: the
    markup is split, blocks are kept or dropped by pattern, and each kept block
    is reduced to the groups of ``value``.
    """
    flat = re.sub(r"[ \t]+", " ", text)
    lines = []
    for spec in detail.get("fields", []):
        match = re.search(spec["pattern"], flat, re.S)
        if match:
            lines.append(f"{spec['label']}: {' '.join(match.group(1).split())}")
    spec = detail.get("records")
    if spec:
        found = 0
        for block in re.split(spec["split"], markup):
            if not re.search(spec["keep_if"], block) or (
                    spec.get("drop_if") and re.search(spec["drop_if"], block)):
                continue
            match = re.search(spec["value"], block, re.S)
            if match:
                lines.append(f"{spec['label']}: " + ", ".join(
                    " ".join(_strip_html(g).split()) for g in match.groups() if g))
                found += 1
        if not found and spec.get("none"):
            lines.append(f"{spec['label']}: {spec['none']}")
    return "\n".join(lines)


def collect_register(cfg: dict, params: dict, report: CollectReport, site_text: str,
                     declared: list[dict] | None = None) -> None:
    """Look an entity up in a register by name, accept a match, fetch its records.

    The trading name is searched first, then each legal name the site declares.
    """
    if cfg.get("requires_env") and not os.environ.get(cfg["requires_env"]):
        report.identity[cfg["id"]] = {
            "status": "not_run",
            "reason": f"this register requires {cfg['requires_env']} to be set "
                      f"({cfg.get('requires_reason', 'see collectors.yaml')})",
        }
        return
    record: dict[str, Any] = {"register": cfg.get("publisher", cfg["id"]), "queries": []}
    report.identity[cfg["id"]] = record
    drop = {w.lower() for w in cfg.get("legal_form_words", [])}
    clean = cfg.get("name_clean")
    accepted: list[dict] = []
    searches = [(params["name"], "trading name")] + [
        (d["name"], "declared on the site") for d in (declared or [])]
    for search_name, why in searches:
        query = _fill(cfg["search"], {**params, "name": search_name})
        attempt: dict[str, Any] = {"name": search_name, "why": why, "query": query}
        record["queries"].append(attempt)
        try:
            content, _, _ = _fetch_text(query)
            candidates = _candidates(cfg, content)
        except (FetchError, ValueError, KeyError) as exc:
            attempt["error"] = str(exc)
            continue
        wanted = normalise_name(search_name, drop)
        seen = []
        for candidate in candidates[: int(cfg.get("consider", 20))]:
            name = candidate["name"]
            while clean and re.sub(clean, "", name) != name:  # suffixes can stack: "/MA (PRGS)"
                name = re.sub(clean, "", name)
            match = bool(wanted) and normalise_name(name, drop) == wanted
            seen.append({"key": candidate["key"], "name": candidate["name"], "match": match})
            if match and candidate["key"] not in {a["key"] for a in accepted}:
                accepted.append({"key": candidate["key"], "name": name.strip(), "via": why})
        attempt["candidates_seen"] = seen[:8]

    if not accepted:
        if all("error" in q for q in record["queries"]):
            record.update(status="failed", reason="the register could not be reached: "
                          + record["queries"][0]["error"])
        else:
            record.update(status="unresolved", reason="no candidate's name matched "
                          + " or ".join(repr(n) for n, _ in searches))
        return
    # Prefer an entity the site itself declared.
    accepted.sort(key=lambda a: a["via"] != "declared on the site")
    if len(accepted) > int(cfg.get("max_entities", 1)):
        record["also_matched"] = accepted[int(cfg.get("max_entities", 1)):]
        accepted = accepted[: int(cfg.get("max_entities", 1))]

    # The subject's own word settles identity both ways: "register:key" entries
    # it confirmed are used, ones it rejected are dropped before anything else.
    confirmed = {str(k) for k in params.get("confirmed") or []}
    rejected = {str(k) for k in params.get("rejected") or []}
    accepted = [a for a in accepted if f"{cfg['id']}:{a['key']}" not in rejected]
    if not accepted:
        record.update(status="unresolved", reason="every match was rejected by the subject")
        return

    lowered_site = site_text.lower()
    base = _base(cfg, report)
    entities = []
    for entity in accepted:
        if f"{cfg['id']}:{entity['key']}" in confirmed:
            entity["via"] = "confirmed by the subject"
        expanded = entity["name"].lower()
        for short, full in (cfg.get("legal_form_expansions") or {}).items():
            expanded = re.sub(rf"\b{re.escape(short)}\b\.?", full, expanded)
        # A bare trading name on the site corroborates nothing -- the site says
        # its own name everywhere. Only a name carrying its legal form does.
        has_form = normalise_name(entity["name"], drop) != normalise_name(entity["name"], set())
        corroborated = (entity["via"] in ("declared on the site", "confirmed by the subject")
                        or entity["key"].lower() in lowered_site
                        or (has_form and (entity["name"].lower() in lowered_site
                                          or expanded in lowered_site)))
        entities.append({**entity, "corroborated": corroborated})
        if not corroborated:
            # A name match nobody has confirmed stays out of the corpus. Indexed,
            # it would let an answer cite another company's address as this one's.
            continue
        tags: list[str] = []
        authority = base["authority"]
        for index, detail in enumerate(cfg.get("detail", []), start=1):
            uri = _fill(detail["url"], {**params, "key": entity["key"]})
            try:
                content, ctype, final = _fetch_text(uri)
            except FetchError as exc:
                if "404" in str(exc) and detail.get("absent_means"):
                    text = (f"{detail.get('label', 'Record')} for {entity['name']} ({entity['key']}) "
                            f"from {cfg.get('publisher', cfg['id'])}.\n\n{detail['absent_means']}\n\n"
                            f"Looked up at {uri} on {datetime.now(timezone.utc).date().isoformat()}; "
                            f"the register returned no record.")
                else:
                    report.skipped.append({"collector": cfg["id"], "uri": uri, "reason": str(exc)})
                    continue
            else:
                if detail.get("keep"):
                    text = _render_json(json.loads(content), detail["keep"])
                elif detail.get("fields") or detail.get("records"):
                    markup = " ".join(content.decode("utf-8", "replace").split())
                    text = _render_page(extract(content, ctype, final).text, markup, detail)
                else:
                    text = extract(content, ctype, final).text
                # Where and when, in the record itself: a compact record of
                # labelled fields is otherwise short enough to be mistaken for a
                # failed extraction, and the lookup is worth stating anyway.
                text = (f"{detail.get('label', 'Record')} for {entity['name']} ({entity['key']}) "
                        f"from {cfg.get('publisher', cfg['id'])}.\n\n{text}\n\n"
                        f"Looked up at {uri} on {datetime.now(timezone.utc).date().isoformat()}.")
            if len(text) < MIN_DOCUMENT_CHARS:
                report.skipped.append({"collector": cfg["id"], "uri": uri, "reason": "record too short"})
                continue
            base_without_authority = {k: v for k, v in base.items() if k != "authority"}
            report.documents.append(Document(
                id=f"{cfg['id']}-{entity['key'].lower()}-{index}",
                title=f"{detail.get('label', 'Record')}: {entity['name']}",
                uri=uri, text=text, publisher=cfg.get("publisher", cfg["id"]),
                authority=authority, tags=tags, **base_without_authority))
    confirmed = any(e["corroborated"] for e in entities)
    record.update(status="resolved" if confirmed else "pending_confirmation", entities=entities)
    if not confirmed:
        record["reason"] = ("name matched, but nothing on the subject's own site confirms it; "
                            "held out of the corpus until the subject confirms it")


KINDS = {"site": collect_site, "feed": collect_feed, "json_items": collect_json_items}


def run_collectors(config: dict, params: dict) -> CollectReport:
    """Run every collector in order. Registers run last: they corroborate against the site."""
    report = CollectReport()
    collectors = config.get("collectors") or []
    for cfg in collectors:
        kind = cfg.get("kind")
        if kind == "register":
            continue
        if kind not in KINDS:
            raise CollectError(f"collector {cfg.get('id')!r} has unknown kind {kind!r}")
        KINDS[kind](cfg, params, report)
    site_text = "\n".join(d.text for d in report.documents if d.tier == "self_published")
    declared = declared_names(site_text, config.get("site_declared_names") or {})
    report.facts["declared_names"] = declared
    for cfg in collectors:
        if cfg.get("kind") == "register":
            collect_register(cfg, params, report, site_text, declared)
    return report


def declared_names(site_text: str, cfg: dict) -> list[dict]:
    """Legal names the subject's own pages declare, in a sentence that says so.

    A site naming its own legal entity ("registered in England as Example Ltd")
    is the strongest identity evidence there is: it needs no fuzzy matching and
    is corroborated by construction. Only sentences carrying a registration cue
    count, so a customer list full of other companies' names does not.
    """
    if not cfg.get("pattern"):
        return []
    name_re = re.compile(cfg["pattern"])
    cue_re = re.compile(cfg.get("cue", "."), re.I)
    found: dict[str, str] = {}
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", site_text):
        if not cue_re.search(sentence):
            continue
        for match in name_re.finditer(sentence):
            name = match.group("name").strip(" ,.")
            found.setdefault(name, sentence.strip()[:300])
    return [{"name": n, "sentence": s} for n, s in list(found.items())[: int(cfg.get("max", 4))]]


def write_register(directory: Path, report: CollectReport, params: dict) -> Path:
    """Write ``sources.yaml``, ``identity.yaml`` and ``facts.json`` for a collected instance."""
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    register = {
        "defaults": {"refresh_days": 3650, "licence": "unspecified"},
        "collected_at": collected_at,
        "sources": [doc.as_source() for doc in report.documents],
    }
    path = directory / "sources.yaml"
    path.write_text(yaml.safe_dump(register, sort_keys=False, allow_unicode=True, width=100),
                    encoding="utf-8")
    (directory / "identity.yaml").write_text(
        yaml.safe_dump({"collected_at": collected_at, "params": params, "registers": report.identity},
                       sort_keys=False, allow_unicode=True), encoding="utf-8")
    (directory / "collection.json").write_text(json.dumps({
        "collected_at": collected_at,
        "counts": report.counts(),
        "facts": report.facts,
        "skipped": report.skipped,
    }, indent=2), encoding="utf-8")
    return path
