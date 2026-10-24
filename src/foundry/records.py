"""Answer from register fields, not from retrieved prose.

Some questions have exactly one kind of right source. "What is the registered
office?" is settled by a register, and the pilot showed text retrieval getting it
wrong: the register record was in the corpus, and the answer came from a privacy
policy that also mentions an address. Retrieval ranks passages by resemblance to
the question; it has no notion that one source is *the* authority on it.

So a question can name the fields that answer it. Records written as
``Label: value`` lines (what the collectors produce for registers) are read
directly, and the answer is those values, each cited to its record. When no
record carries a named field the caller falls back to ordinary retrieval -- this
is a shortcut for questions with an authority, not a replacement for the gates.
"""

from __future__ import annotations

import re
import time

from .specialist.answer import ANSWERED, CONFIDENCE_HIGH, Answer
from .storage import Store

_FIELD_RE = re.compile(r"^([A-Za-z][^:\n]{1,60}):\s*(.+?)\s*$", re.M)


def read_fields(store: Store, labels: list[str] | None,
                source_types: tuple[str, ...] = ("official_record",)) -> list[dict]:
    """Every record of ``source_types`` that carries at least one of ``labels`` (all, if None)."""
    wanted = [label.lower() for label in labels] if labels is not None else None
    found = []
    for source in store.sources():
        if source["source_type"] not in source_types:
            continue
        rows = store.conn.execute(
            "SELECT text FROM documents WHERE source_id = ?", (source["id"],)).fetchall()
        values: dict[str, list[str]] = {}
        for row in rows:
            for label, value in _FIELD_RE.findall(row["text"]):
                if "http" in label.lower():  # "Looked up at https://..." is prose, not a field
                    continue
                if wanted is None or label.lower() in wanted:
                    values.setdefault(label, [])
                    if value not in values[label]:
                        values[label].append(value)
        if values:
            ordered = (list(values.items()) if wanted is None else
                       sorted(values.items(), key=lambda kv: wanted.index(kv[0].lower())))
            found.append({
                "source_id": source["id"], "title": source["title"], "uri": source["uri"],
                "publisher": source["publisher"], "source_type": source["source_type"],
                "authority": source["authority"], "licence": source["licence"],
                "retrieved_at": source["retrieved_at"], "fields": ordered,
            })
    return found


def answer_from_fields(store: Store, knowledge_area: str, question: str,
                       labels: list[str]) -> Answer | None:
    """An answer built only from register fields, or ``None`` when no record has them."""
    started = time.monotonic()
    records = read_fields(store, labels)
    if not records:
        return None
    lines, sources = [], []
    for rank, record in enumerate(records, start=1):
        parts = []
        for label, values in record["fields"]:
            shown = values if len(values) <= 8 else values[:8] + [f"and {len(values) - 8} more"]
            parts.append(f"{label}: {'; '.join(shown)}")
        # Values often end in a full stop of their own ("INC.", "...register.").
        sentences = [part if part.endswith(".") else part + "." for part in parts]
        lines.append(f"{record['publisher']} records {' '.join(sentences)} [{rank}]")
        sources.append({
            "rank": rank, "source_id": record["source_id"], "title": record["title"],
            "citation": f"{record['title']} — {record['publisher']} — {record['uri']}",
            "uri": record["uri"], "publisher": record["publisher"],
            "source_type": record["source_type"], "authority": record["authority"],
            "licence": record["licence"], "retrieved_at": record["retrieved_at"],
            "cited": True,
        })
    return Answer(
        question=question, knowledge_area=knowledge_area, status=ANSWERED,
        answer="\n".join(lines),
        reasoning="Read directly from register fields; no passage retrieval and no inference.",
        sources=sources, confidence=CONFIDENCE_HIGH, confidence_score=1.0,
        llm={"provider": "register_fields"},
        retrieval={"fields": labels, "records": len(records)},
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )
