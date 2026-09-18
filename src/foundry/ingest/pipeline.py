"""The ingestion pipeline (§7).

    sources.yaml -> fetch -> cache(sha256) -> extract -> clean -> chunk -> index -> claims

Two properties matter more than throughput:

**Incrementality.** Content is addressed by SHA-256. Re-ingesting an unchanged
source is a no-op and re-ingesting a changed one replaces exactly its documents
and chunks. That is what makes success criterion #7 -- update without rebuilding
the whole system -- true rather than aspirational.

**Legality.** Every source carries a licence, and sources marked ``mirrored:
false`` are registered as *pointers*: title, publisher, URI and dates are stored
and shown, the content never is. The knowledge area's ``known_limitations`` must
say so, and the public evidence page repeats it.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .. import paths
from ..chunking import chunk_document
from ..manifest import Manifest, ManifestError
from ..storage import Source, Store, utcnow
from .claims import extract_claims
from .extractors import ExtractionError, extract
from .fetchers import FetchError, fetch

REQUIRED_SOURCE_FIELDS = ("id", "title", "publisher", "source_type", "authority", "licence")
SOURCE_TYPES = (
    "regulatory", "standard", "investigation", "academic",
    "industry", "dataset", "expert", "reference",
)


@dataclass
class IngestReport:
    """What the pipeline did, in enough detail to debug a bad build."""

    knowledge_area: str
    started_at: str = field(default_factory=utcnow)
    finished_at: str | None = None
    registered: int = 0
    fetched: int = 0
    unchanged: int = 0
    skipped_fresh: int = 0
    pointers: int = 0
    indexed: int = 0
    documents: int = 0
    chunks: int = 0
    claims: int = 0
    failures: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "knowledge_area": self.knowledge_area,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "registered": self.registered,
            "fetched": self.fetched,
            "unchanged": self.unchanged,
            "skipped_fresh": self.skipped_fresh,
            "pointers": self.pointers,
            "indexed": self.indexed,
            "documents": self.documents,
            "chunks": self.chunks,
            "claims": self.claims,
            "failures": self.failures,
        }


def load_source_register(manifest: Manifest) -> list[dict]:
    """Read and validate ``sources.yaml``, applying register-level defaults."""
    path = manifest.sources_path
    if not path.is_file():
        raise ManifestError(f"source register not found at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults") or {}
    entries = data.get("sources") or []
    if not isinstance(entries, list):
        raise ManifestError("sources.yaml: 'sources' must be a list")

    seen: set[str] = set()
    out: list[dict] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"sources.yaml: entry {index} is not a mapping")
        merged = {**defaults, **entry}
        for key in REQUIRED_SOURCE_FIELDS:
            if not merged.get(key):
                raise ManifestError(
                    f"sources.yaml: entry {index} ({merged.get('id', '?')}) missing '{key}'"
                )
        if not merged.get("uri") and merged.get("content") is None:
            raise ManifestError(
                f"sources.yaml: {merged['id']} needs either 'uri' or inline 'content'"
            )
        if merged["source_type"] not in SOURCE_TYPES:
            raise ManifestError(
                f"sources.yaml: {merged['id']} has unknown source_type "
                f"{merged['source_type']!r}; expected one of {SOURCE_TYPES}"
            )
        authority = int(merged["authority"])
        if not 1 <= authority <= 5:
            raise ManifestError(f"sources.yaml: {merged['id']} authority must be 1-5")
        merged["authority"] = authority
        if merged["id"] in seen:
            raise ManifestError(f"sources.yaml: duplicate source id {merged['id']!r}")
        seen.add(merged["id"])
        out.append(merged)
    return out


def _needs_refresh(row, refresh_days: int) -> bool:
    if row is None or row["status"] not in ("indexed", "pointer"):
        return True
    if not row["retrieved_at"]:
        return True
    try:
        when = _dt.datetime.fromisoformat(row["retrieved_at"])
    except ValueError:
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    age = (_dt.datetime.now(_dt.timezone.utc) - when).days
    return age >= refresh_days


def _index_document(
    store: Store, manifest: Manifest, spec: dict, extracted, content_sha: str, report: IngestReport
) -> None:
    """Replace one source's document, chunks and claims atomically."""
    source_id = spec["id"]
    doc_id = f"doc_{hashlib.sha256(source_id.encode()).hexdigest()[:20]}"
    title = spec.get("title") or extracted.title or source_id

    store.replace_document(
        doc_id=doc_id,
        source_id=source_id,
        title=title,
        text=extracted.text,
        extractor=extracted.extractor,
        content_sha256=content_sha,
    )
    cfg = manifest.retrieval.chunk
    chunks = chunk_document(
        extracted.text,
        target_chars=cfg.target_chars,
        overlap_chars=cfg.overlap_chars,
        min_chars=cfg.min_chars,
    )
    rows = [c.as_row(doc_id, source_id) for c in chunks]
    store.add_chunks(rows)

    claim_count = 0
    with store.transaction():
        for row in rows:
            for claim in extract_claims(row["id"], row["text"]):
                store.add_claim(
                    claim_id=claim["id"],
                    chunk_id=row["id"],
                    source_id=source_id,
                    text=claim["text"],
                    kind=claim["kind"],
                    subject=row["section"],
                )
                claim_count += 1

    report.documents += 1
    report.chunks += len(rows)
    report.claims += claim_count
    report.indexed += 1


def ingest_source(
    store: Store,
    manifest: Manifest,
    spec: dict,
    report: IngestReport,
    force: bool = False,
    rebuild: bool = False,
    offline: bool = False,
) -> None:
    """Ingest one source, recording provenance and status at every step.

    ``force`` re-fetches regardless of the refresh window. ``rebuild`` also
    re-extracts and re-chunks even when the bytes are unchanged. They are
    separate because the content-hash short circuit *is* the incrementality
    guarantee -- "fetch it again now" must not silently mean "rebuild the whole
    corpus", or success criterion #7 quietly stops being true.
    """
    source_id = spec["id"]
    refresh_days = int(spec.get("refresh_days", 180))
    mirrored = bool(spec.get("mirrored", True))
    existing = store.get_source(source_id)

    source = Source(
        id=source_id,
        uri=spec.get("uri") or f"inline:{source_id}",
        title=spec["title"],
        publisher=spec["publisher"],
        source_type=spec["source_type"],
        authority=spec["authority"],
        licence=spec["licence"],
        published_at=str(spec["published_at"]) if spec.get("published_at") else None,
        doc_version=spec.get("doc_version"),
        mirrored=mirrored,
        refresh_days=refresh_days,
        notes=spec.get("notes", ""),
        tags=list(spec.get("tags") or []),
        retrieved_at=existing["retrieved_at"] if existing else None,
        content_sha256=existing["content_sha256"] if existing else None,
        raw_path=existing["raw_path"] if existing else None,
        status=existing["status"] if existing else "registered",
    )

    # Pointer-only sources: cited, never mirrored. Licence compliance by design.
    if not mirrored:
        source.status = "pointer"
        source.retrieved_at = utcnow()
        store.upsert_source(source)
        report.registered += 1
        report.pointers += 1
        return

    store.upsert_source(source)
    report.registered += 1

    if not force and not _needs_refresh(existing, refresh_days):
        report.skipped_fresh += 1
        return

    if offline and spec.get("content") is None and not str(spec.get("uri", "")).startswith("file"):
        report.skipped_fresh += 1
        return

    try:
        result = fetch(spec.get("uri") or "inline:", inline_content=spec.get("content"))
    except FetchError as exc:
        store.update_source(source_id, status="failed", error=str(exc))
        report.failures.append({"source": source_id, "stage": "fetch", "error": str(exc)})
        return

    report.fetched += 1
    content_sha = result.sha256
    raw_path = paths.raw_dir(manifest.id) / f"{content_sha}.bin"
    if not raw_path.exists():
        raw_path.write_bytes(result.content)

    store.update_source(
        source_id,
        retrieved_at=utcnow(),
        content_sha256=content_sha,
        raw_path=str(raw_path),
        status="fetched",
        error=None,
    )

    # Content-addressed short circuit: unchanged bytes need no reindexing.
    if (
        not rebuild
        and existing
        and existing["content_sha256"] == content_sha
        and existing["status"] == "indexed"
    ):
        store.update_source(source_id, status="indexed")
        report.unchanged += 1
        return

    try:
        extracted = extract(result.content, result.content_type, result.final_uri)
    except ExtractionError as exc:
        store.update_source(source_id, status="failed", error=str(exc))
        report.failures.append({"source": source_id, "stage": "extract", "error": str(exc)})
        return

    if len(extracted.text.strip()) < 200:
        message = f"extracted only {len(extracted.text.strip())} characters"
        store.update_source(source_id, status="failed", error=message)
        report.failures.append({"source": source_id, "stage": "extract", "error": message})
        return

    _index_document(store, manifest, spec, extracted, content_sha, report)
    store.update_source(source_id, status="indexed", error=None)


def load_ontology(manifest: Manifest, store: Store) -> tuple[int, int]:
    """Load ``ontology.yaml`` into the entities and triples tables (§8)."""
    if not manifest.ontology_path:
        return (0, 0)
    path = manifest.resolve(manifest.ontology_path)
    if not path.is_file():
        return (0, 0)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    known: set[str] = set()

    with store.transaction():
        for entity in entities:
            entity_id = entity["id"]
            store.upsert_entity(
                entity_id=entity_id,
                name=entity.get("name", entity_id),
                kind=entity.get("kind", "concept"),
                aliases=list(entity.get("aliases") or []),
                description=entity.get("description", ""),
            )
            known.add(entity_id)

        triple_count = 0
        for rel in relationships:
            subject, predicate, obj = rel.get("subject"), rel.get("predicate"), rel.get("object")
            if subject not in known or obj not in known:
                raise ManifestError(
                    f"ontology.yaml: relationship {subject!r} -{predicate}-> {obj!r} "
                    "references an undeclared entity"
                )
            store.add_triple(
                subject_id=subject,
                predicate=predicate,
                object_id=obj,
                source_id=rel.get("source"),
                confidence=float(rel.get("confidence", 1.0)),
            )
            triple_count += 1
    return (len(known), triple_count)


def ingest_knowledge_area(
    manifest: Manifest,
    store: Store | None = None,
    force: bool = False,
    rebuild: bool = False,
    offline: bool = False,
    only: list[str] | None = None,
) -> IngestReport:
    """Run the full pipeline for a knowledge area."""
    owns_store = store is None
    store = store or Store(manifest.id)
    report = IngestReport(knowledge_area=manifest.id)
    try:
        specs = load_source_register(manifest)
        if only:
            wanted = set(only)
            specs = [s for s in specs if s["id"] in wanted]
        for spec in specs:
            ingest_source(
                store, manifest, spec, report,
                force=force, rebuild=rebuild, offline=offline,
            )
        load_ontology(manifest, store)
        report.finished_at = utcnow()
        store.set_meta("last_ingest", report.finished_at)
    finally:
        if owns_store:
            store.close()
    return report
