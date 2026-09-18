"""The public evidence report (§15).

Assembles everything a sceptic needs to judge a knowledge area: what it covers,
where the knowledge came from, how it performs against its own tests, what
changed between versions, what it does not know, and what people have found
wrong with it.

This is the differentiator the brief describes, so it is built from live state
rather than written by hand. A limitation that has to be remembered is a
limitation that eventually goes stale; this one is recomputed on every request.
"""

from __future__ import annotations

import json

from . import paths
from .feedback import challenge_stats
from .manifest import Manifest
from .storage import Store
from .versioning import version_history


def _latest_report(manifest: Manifest, name: str = "latest.json") -> dict | None:
    path = paths.reports_dir(manifest.id) / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def source_register(store: Store) -> list[dict]:
    """The full source list, pointers included and marked as such."""
    rows = store.conn.execute(
        """
        SELECT s.*, COUNT(c.id) AS chunk_count
        FROM sources s
        LEFT JOIN chunks c ON c.source_id = s.id
        GROUP BY s.id
        ORDER BY s.authority DESC, s.publisher, s.title
        """
    ).fetchall()
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "publisher": row["publisher"],
            "uri": row["uri"],
            "source_type": row["source_type"],
            "authority": row["authority"],
            "published_at": row["published_at"],
            "retrieved_at": row["retrieved_at"],
            "licence": row["licence"],
            "mirrored": bool(row["mirrored"]),
            "status": row["status"],
            "error": row["error"],
            "notes": row["notes"],
            "tags": json.loads(row["tags"] or "[]"),
            "chunk_count": row["chunk_count"],
        }
        for row in rows
    ]


def coverage(store: Store) -> dict:
    """What the corpus is actually made of, by type, authority and tag."""
    by_type = {
        row["source_type"]: row["n"]
        for row in store.conn.execute(
            "SELECT source_type, COUNT(*) AS n FROM sources GROUP BY source_type"
        )
    }
    by_authority = {
        str(row["authority"]): row["n"]
        for row in store.conn.execute(
            "SELECT authority, COUNT(*) AS n FROM sources GROUP BY authority ORDER BY authority DESC"
        )
    }
    chunks_by_type = {
        row["source_type"]: row["n"]
        for row in store.conn.execute(
            "SELECT s.source_type, COUNT(c.id) AS n FROM chunks c "
            "JOIN sources s ON s.id = c.source_id GROUP BY s.source_type"
        )
    }
    tags: dict[str, int] = {}
    for row in store.conn.execute("SELECT tags FROM sources"):
        for tag in json.loads(row["tags"] or "[]"):
            tags[tag] = tags.get(tag, 0) + 1

    return {
        "sources_by_type": by_type,
        "sources_by_authority": by_authority,
        "chunks_by_type": chunks_by_type,
        "tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])),
        "entities": store.counts()["entities"],
        "relationships": store.counts()["triples"],
    }


def knowledge_area_report(manifest: Manifest, store: Store) -> dict:
    """Everything the public evidence page renders (§15)."""
    counts = store.counts()
    evaluation = _latest_report(manifest)
    redteam = _latest_report(manifest, "redteam_latest.json")
    register = source_register(store)
    freshness = store.freshness_days()

    pointers = [s for s in register if not s["mirrored"]]
    failed = [s for s in register if s["status"] == "failed"]

    return {
        "knowledge_area": manifest.summary(),
        "about": {
            "description": manifest.description,
            "in_scope": manifest.scope.in_scope,
            "out_of_scope": manifest.scope.out_of_scope,
            "disclaimer": manifest.governance.disclaimer,
        },
        "version": {
            "current": store.get_meta("current_version", "unbuilt"),
            "published": store.get_meta("published_version") or None,
            "last_ingest": store.get_meta("last_ingest"),
            "last_evaluation": store.get_meta("last_evaluation"),
            "last_redteam": store.get_meta("last_redteam"),
            "evaluation_status": store.get_meta("evaluation_status", "unknown"),
        },
        "corpus": {
            "sources": counts["sources"],
            "indexed_sources": counts["indexed_sources"],
            "pointer_sources": len(pointers),
            "failed_sources": len(failed),
            "documents": counts["documents"],
            "chunks": counts["chunks"],
            "claims": counts["claims"],
            "freshness_days": round(freshness, 1) if freshness is not None else None,
            "embedding_model": store.get_meta("embedding_model"),
        },
        "coverage": coverage(store),
        "sources": register,
        "evaluation": {
            "run_id": (evaluation or {}).get("run_id"),
            "total": (evaluation or {}).get("total", 0),
            "passed_count": (evaluation or {}).get("passed_count", 0),
            "pass_rate": (evaluation or {}).get("pass_rate", 0.0),
            "by_type": (evaluation or {}).get("by_type", {}),
            "metrics": (evaluation or {}).get("metrics", {}),
            "passed": (evaluation or {}).get("passed"),
            "threshold_failures": (evaluation or {}).get("threshold_failures", []),
            # Failures are published, not hidden. That is the entire point (§25).
            "failures": [
                r for r in (evaluation or {}).get("results", []) if not r.get("passed")
            ],
        },
        "redteam": {
            "run_id": (redteam or {}).get("run_id"),
            "total": (redteam or {}).get("total", 0),
            "broke": (redteam or {}).get("broke", 0),
            "survival_rate": (redteam or {}).get("survival_rate"),
            "by_failure_mode": (redteam or {}).get("by_failure_mode", {}),
            "by_strategy": (redteam or {}).get("by_strategy", {}),
            "breaks": [v for v in (redteam or {}).get("verdicts", []) if v.get("broke")][:25],
        },
        "updates": version_history(manifest),
        "limitations": {
            "declared": manifest.governance.known_limitations,
            "pointer_sources": [
                {"title": s["title"], "publisher": s["publisher"], "licence": s["licence"]}
                for s in pointers
            ],
            "failed_sources": [
                {"id": s["id"], "title": s["title"], "error": s["error"]} for s in failed
            ],
        },
        "governance": {
            "review_status": manifest.governance.review_status,
            "confidence": manifest.governance.confidence,
            "reviewers": manifest.governance.reviewers,
        },
        "challenges": challenge_stats(store),
    }


def source_detail(store: Store, source_id: str) -> dict | None:
    """One source with its stored passages -- the source viewer (§22)."""
    row = store.get_source(source_id)
    if row is None:
        return None
    chunks = list(
        store.conn.execute(
            "SELECT id, ordinal, section, text, char_start, char_end FROM chunks "
            "WHERE source_id = ? ORDER BY ordinal",
            (source_id,),
        )
    )
    claims = list(
        store.conn.execute(
            "SELECT text, kind FROM claims WHERE source_id = ? ORDER BY id LIMIT 200",
            (source_id,),
        )
    )
    return {
        "source": {
            "id": row["id"],
            "title": row["title"],
            "publisher": row["publisher"],
            "uri": row["uri"],
            "source_type": row["source_type"],
            "authority": row["authority"],
            "published_at": row["published_at"],
            "retrieved_at": row["retrieved_at"],
            "licence": row["licence"],
            "mirrored": bool(row["mirrored"]),
            "status": row["status"],
            "notes": row["notes"],
            "content_sha256": row["content_sha256"],
        },
        "chunks": [dict(c) for c in chunks],
        "claims": [dict(c) for c in claims],
    }
