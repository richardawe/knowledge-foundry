"""SQLite storage: the knowledge area's derived index, with provenance built in.

Implements the data model in IMPLEMENTATION_PLAN.md §4. One file per knowledge
area, so a knowledge area backs up with ``cp`` and adding the tenth one adds
disk rather than a server (§17).

The central design decision is that **provenance is structural**: a chunk cannot
exist without a document, which cannot exist without a source carrying a URI,
publisher, dates, licence and content hash. "Where did this answer come from?"
(§7) is therefore a SQL join, and cannot silently stop being true.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import paths

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Every piece of knowledge traces back to a row here (§7).
CREATE TABLE IF NOT EXISTS sources (
    id             TEXT PRIMARY KEY,
    uri            TEXT NOT NULL,
    title          TEXT NOT NULL,
    publisher      TEXT NOT NULL,
    source_type    TEXT NOT NULL,          -- regulatory|standard|investigation|academic|industry|dataset|expert
    authority      INTEGER NOT NULL,       -- 1 (weak) .. 5 (primary authority)
    published_at   TEXT,
    retrieved_at   TEXT,
    doc_version    TEXT,
    licence        TEXT NOT NULL,
    mirrored       INTEGER NOT NULL DEFAULT 1,  -- 0 = pointer only, content not stored
    content_sha256 TEXT,
    raw_path       TEXT,
    refresh_days   INTEGER NOT NULL DEFAULT 180,
    status         TEXT NOT NULL DEFAULT 'registered',  -- registered|fetched|extracted|indexed|failed|skipped
    error          TEXT,
    notes          TEXT,
    tags           TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);

CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    title        TEXT NOT NULL,
    text         TEXT NOT NULL,
    char_count   INTEGER NOT NULL,
    extractor    TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    content_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);

CREATE TABLE IF NOT EXISTS chunks (
    id             TEXT PRIMARY KEY,
    document_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_id      TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    ordinal        INTEGER NOT NULL,
    section        TEXT NOT NULL DEFAULT '',
    text           TEXT NOT NULL,
    char_start     INTEGER NOT NULL,
    char_end       INTEGER NOT NULL,
    token_estimate INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);

-- BM25 over chunk text (§8, keyword retrieval). Contentless external-content
-- table: chunks stays the single source of truth for text.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'concept',
    aliases     TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

-- Entity -> relationship -> entity (§8). A table, not a graph database (§23).
CREATE TABLE IF NOT EXISTS triples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate  TEXT NOT NULL,
    object_id  TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    source_id  TEXT REFERENCES sources(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    UNIQUE(subject_id, predicate, object_id)
);
CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject_id);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object_id);

-- "claims derived from source" (§7): the unit the grounding checker verifies
-- answer sentences against.
CREATE TABLE IF NOT EXISTS claims (
    id           TEXT PRIMARY KEY,
    chunk_id     TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    source_id    TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    text         TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'fact',  -- fact|range|requirement|definition
    subject      TEXT NOT NULL DEFAULT '',
    extracted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_chunk ON claims(chunk_id);

CREATE TABLE IF NOT EXISTS kb_versions (
    version      TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    manifest_sha TEXT NOT NULL,
    corpus_sha   TEXT NOT NULL,
    index_sha    TEXT NOT NULL,
    eval_run_id  TEXT,
    status       TEXT NOT NULL DEFAULT 'draft',   -- draft|published|rolled_back
    notes        TEXT NOT NULL DEFAULT '',
    stats        TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id          TEXT PRIMARY KEY,
    version     TEXT,
    suite       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    metrics     TEXT NOT NULL DEFAULT '{}',
    passed      INTEGER NOT NULL DEFAULT 0,
    kind        TEXT NOT NULL DEFAULT 'evaluation'  -- evaluation|redteam
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id      TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id TEXT NOT NULL,
    question    TEXT NOT NULL,
    qtype       TEXT NOT NULL,
    passed      INTEGER NOT NULL,
    scores      TEXT NOT NULL DEFAULT '{}',
    answer      TEXT NOT NULL DEFAULT '{}',
    notes       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, question_id)
);

-- Expert challenges (§14); promoted challenges become permanent tests (§24.9).
CREATE TABLE IF NOT EXISTS challenges (
    id              TEXT PRIMARY KEY,
    submitted_by    TEXT NOT NULL DEFAULT 'anonymous',
    question        TEXT NOT NULL,
    claimed_problem TEXT NOT NULL DEFAULT '',
    expected        TEXT NOT NULL DEFAULT '',
    system_answer   TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'open',  -- open|accepted|rejected|promoted
    created_at      TEXT NOT NULL,
    promoted_to     TEXT,
    notes           TEXT NOT NULL DEFAULT ''
);
"""


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class Source:
    """A registered source with the provenance §7 requires."""

    id: str
    uri: str
    title: str
    publisher: str
    source_type: str
    authority: int
    licence: str
    published_at: str | None = None
    retrieved_at: str | None = None
    doc_version: str | None = None
    mirrored: bool = True
    content_sha256: str | None = None
    raw_path: str | None = None
    refresh_days: int = 180
    status: str = "registered"
    error: str | None = None
    notes: str = ""
    tags: list[str] | None = None

    def as_row(self) -> dict:
        return {
            "id": self.id,
            "uri": self.uri,
            "title": self.title,
            "publisher": self.publisher,
            "source_type": self.source_type,
            "authority": int(self.authority),
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "doc_version": self.doc_version,
            "licence": self.licence,
            "mirrored": 1 if self.mirrored else 0,
            "content_sha256": self.content_sha256,
            "raw_path": self.raw_path,
            "refresh_days": int(self.refresh_days),
            "status": self.status,
            "error": self.error,
            "notes": self.notes,
            "tags": json.dumps(self.tags or []),
        }


class Store:
    """Thin, explicit wrapper over the knowledge area's SQLite database.

    Deliberately not an ORM. The queries are the interesting part -- hiding them
    behind a mapper would hide the retrieval behaviour this project is supposed
    to be able to explain.
    """

    def __init__(self, ka_id: str, path: Path | None = None) -> None:
        self.ka_id = ka_id
        self.path = Path(path) if path else paths.db_path(ka_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
            ("knowledge_area", self.ka_id),
        )
        self.conn.commit()

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- meta ------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    # -- sources ---------------------------------------------------------

    def upsert_source(self, source: Source) -> None:
        row = source.as_row()
        cols = ", ".join(row)
        placeholders = ", ".join(f":{c}" for c in row)
        updates = ", ".join(f"{c} = excluded.{c}" for c in row if c != "id")
        self.conn.execute(
            f"INSERT INTO sources ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            row,
        )
        self.conn.commit()

    def update_source(self, source_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE sources SET {assignments} WHERE id = ?",
            (*fields.values(), source_id),
        )
        self.conn.commit()

    def get_source(self, source_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()

    def sources(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return list(
                self.conn.execute(
                    "SELECT * FROM sources WHERE status = ? ORDER BY authority DESC, id",
                    (status,),
                )
            )
        return list(
            self.conn.execute("SELECT * FROM sources ORDER BY authority DESC, id")
        )

    def pointer_sources(self) -> list[sqlite3.Row]:
        """Sources the register names but whose text is not held.

        The licence guarantee depends on this list being consultable without
        going through retrieval: what may be reproduced is a property of the
        register, and asking the evidence instead only ever finds whatever
        happens to mention the thing.
        """
        return list(
            self.conn.execute(
                "SELECT * FROM sources WHERE mirrored = 0 ORDER BY authority DESC, id"
            )
        )

    # -- documents & chunks ----------------------------------------------

    def replace_document(
        self,
        doc_id: str,
        source_id: str,
        title: str,
        text: str,
        extractor: str,
        content_sha256: str,
    ) -> None:
        """Insert or replace a document, cascading away its old chunks.

        Replacing rather than mutating is what makes incremental re-ingestion
        safe: a changed source cannot leave orphaned chunks behind.
        """
        with self.transaction() as conn:
            self._delete_chunk_fts_for_document(doc_id)
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.execute(
                "INSERT INTO documents"
                "(id, source_id, title, text, char_count, extractor, extracted_at, content_sha256)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, source_id, title, text, len(text), extractor, utcnow(), content_sha256),
            )

    def _delete_chunk_fts_for_document(self, doc_id: str) -> None:
        rows = self.conn.execute(
            "SELECT rowid, text, section FROM chunks WHERE document_id = ?", (doc_id,)
        ).fetchall()
        for row in rows:
            self.conn.execute(
                "INSERT INTO chunks_fts(chunks_fts, rowid, text, section) VALUES('delete', ?, ?, ?)",
                (row["rowid"], row["text"], row["section"]),
            )

    def add_chunks(self, chunks: Iterable[dict]) -> int:
        count = 0
        with self.transaction() as conn:
            for chunk in chunks:
                cur = conn.execute(
                    "INSERT INTO chunks"
                    "(id, document_id, source_id, ordinal, section, text,"
                    " char_start, char_end, token_estimate)"
                    " VALUES (:id, :document_id, :source_id, :ordinal, :section, :text,"
                    " :char_start, :char_end, :token_estimate)",
                    chunk,
                )
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, text, section) VALUES (?, ?, ?)",
                    (cur.lastrowid, chunk["text"], chunk["section"]),
                )
                count += 1
        return count

    def chunk(self, chunk_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()

    def chunks(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM chunks ORDER BY source_id, ordinal"))

    def chunks_with_provenance(self, chunk_ids: Iterable[str]) -> dict[str, sqlite3.Row]:
        """The join that answers 'where did this come from?' (§7)."""
        ids = list(chunk_ids)
        if not ids:
            return {}
        placeholders = ", ".join("?" * len(ids))
        rows = self.conn.execute(
            f"""
            SELECT c.id AS chunk_id, c.text, c.section, c.ordinal,
                   d.id AS document_id, d.title AS document_title,
                   s.id AS source_id, s.uri, s.title AS source_title, s.publisher,
                   s.source_type, s.authority, s.published_at, s.retrieved_at,
                   s.doc_version, s.licence, s.mirrored
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN sources   s ON s.id = c.source_id
            WHERE c.id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        return {row["chunk_id"]: row for row in rows}

    # -- embeddings ------------------------------------------------------

    def set_embedding(self, chunk_id: str, model: str, dim: int, vector: bytes) -> None:
        self.conn.execute(
            "INSERT INTO embeddings(chunk_id, model, dim, vector) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET model=excluded.model, "
            "dim=excluded.dim, vector=excluded.vector",
            (chunk_id, model, dim, vector),
        )

    def embeddings(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM embeddings"))

    def clear_embeddings(self) -> None:
        self.conn.execute("DELETE FROM embeddings")
        self.conn.commit()

    # -- ontology --------------------------------------------------------

    def upsert_entity(
        self, entity_id: str, name: str, kind: str, aliases: list[str], description: str
    ) -> None:
        self.conn.execute(
            "INSERT INTO entities(id, name, kind, aliases, description) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, kind=excluded.kind, "
            "aliases=excluded.aliases, description=excluded.description",
            (entity_id, name, kind, json.dumps(aliases), description),
        )

    def add_triple(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        source_id: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO triples(subject_id, predicate, object_id, source_id, confidence)"
            " VALUES (?, ?, ?, ?, ?)",
            (subject_id, predicate, object_id, source_id, confidence),
        )

    def entities(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM entities ORDER BY name"))

    def triples(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT t.*, e1.name AS subject_name, e2.name AS object_name "
                "FROM triples t JOIN entities e1 ON e1.id = t.subject_id "
                "JOIN entities e2 ON e2.id = t.object_id ORDER BY e1.name, t.predicate"
            )
        )

    # -- claims ----------------------------------------------------------

    def add_claim(
        self,
        claim_id: str,
        chunk_id: str,
        source_id: str,
        text: str,
        kind: str = "fact",
        subject: str = "",
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO claims"
            "(id, chunk_id, source_id, text, kind, subject, extracted_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (claim_id, chunk_id, source_id, text, kind, subject, utcnow()),
        )

    def claims_for_chunks(self, chunk_ids: Iterable[str]) -> list[sqlite3.Row]:
        ids = list(chunk_ids)
        if not ids:
            return []
        placeholders = ", ".join("?" * len(ids))
        return list(
            self.conn.execute(
                f"SELECT * FROM claims WHERE chunk_id IN ({placeholders})", ids
            )
        )

    # -- statistics ------------------------------------------------------

    def counts(self) -> dict[str, int]:
        tables = ("sources", "documents", "chunks", "entities", "triples", "claims", "challenges")
        out = {}
        for table in tables:
            out[table] = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        out["indexed_sources"] = self.conn.execute(
            "SELECT COUNT(*) AS n FROM sources WHERE status = 'indexed'"
        ).fetchone()["n"]
        return out

    def corpus_hash(self) -> str:
        """Hash of document content hashes -- the corpus identity for versioning."""
        import hashlib

        rows = self.conn.execute(
            "SELECT id, content_sha256 FROM documents ORDER BY id"
        ).fetchall()
        h = hashlib.sha256()
        for row in rows:
            h.update(row["id"].encode())
            h.update(row["content_sha256"].encode())
        return h.hexdigest()

    def freshness_days(self) -> float | None:
        """Median age of retrieved sources, in days -- the §12 freshness metric."""
        rows = self.conn.execute(
            "SELECT retrieved_at FROM sources WHERE retrieved_at IS NOT NULL"
        ).fetchall()
        if not rows:
            return None
        now = _dt.datetime.now(_dt.timezone.utc)
        ages = []
        for row in rows:
            try:
                when = _dt.datetime.fromisoformat(row["retrieved_at"])
            except ValueError:
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=_dt.timezone.utc)
            ages.append((now - when).total_seconds() / 86400.0)
        if not ages:
            return None
        ages.sort()
        mid = len(ages) // 2
        return ages[mid] if len(ages) % 2 else (ages[mid - 1] + ages[mid]) / 2
