"""Versioning, the publish gate and rollback (§12, §15).

A version identifies **a set of inputs**, not a point in time. It is derived
from a content hash of the manifest, the system prompt, the corpus and the
index configuration, so:

* rebuilding identical inputs produces the identical version, and
* "what changed between versions" is answerable mechanically rather than from
  someone's memory.

The bump rule encodes what kind of change happened. A configuration or prompt
change bumps the minor number, because it can change every answer. New or
updated source content bumps the patch number, because it changes what is
known without changing how the system behaves.

**Publishing is gated, not ceremonial.** A build is published only if its
evaluation passed its absolute thresholds AND nothing regressed against the
last published version. Otherwise it stays a draft and the previously published
version keeps serving -- which is the whole point of §12's PASS/FAIL diagram.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .evaluation.metrics import METRICS, is_regression
from .manifest import Manifest
from .storage import Store, utcnow

_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


@dataclass
class Version:
    version: str
    created_at: str
    manifest_sha: str
    corpus_sha: str
    index_sha: str
    status: str = "draft"
    notes: str = ""
    stats: dict = field(default_factory=dict)
    reused: bool = False

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "manifest_sha": self.manifest_sha[:16],
            "corpus_sha": self.corpus_sha[:16],
            "index_sha": self.index_sha[:16],
            "status": self.status,
            "notes": self.notes,
            "stats": self.stats,
            "reused": self.reused,
        }


@dataclass
class PublishResult:
    version: str
    published: bool
    reasons: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    newly_failing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "published": self.published,
            "reasons": self.reasons,
            "regressions": self.regressions,
            "newly_failing": self.newly_failing,
        }

    def render(self) -> str:
        if self.published:
            return f"published {self.version}"
        lines = [f"HELD {self.version}: not published"]
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        for regression in self.regressions:
            lines.append(f"  - regression: {regression}")
        if self.newly_failing:
            lines.append(f"  - newly failing questions: {', '.join(self.newly_failing[:20])}")
        lines.append("  The previously published version continues to serve.")
        return "\n".join(lines)


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def compute_hashes(manifest: Manifest, store: Store) -> tuple[str, str, str]:
    """(manifest_sha, corpus_sha, index_sha) -- the three axes of a version."""
    try:
        prompt = manifest.system_prompt()
    except Exception:
        prompt = ""
    manifest_sha = _sha(manifest.content_hash(), prompt)
    corpus_sha = store.corpus_hash()
    index_sha = _sha(
        manifest.retrieval.semantic_provider,
        str(manifest.retrieval.semantic_dim),
        store.get_meta("embedding_model", "") or "",
        store.get_meta("embedding_dim", "") or "",
    )
    return manifest_sha, corpus_sha, index_sha


# Timestamps have second resolution, so two versions cut in the same second
# tie. Insertion order (rowid) breaks the tie; a version *string* cannot, since
# lexical ordering puts v0.10.0 before v0.9.0.
_VERSION_ORDER = "ORDER BY created_at DESC, rowid DESC"


def _latest_row(store: Store):
    return store.conn.execute(
        f"SELECT rowid, * FROM kb_versions {_VERSION_ORDER} LIMIT 1"
    ).fetchone()


def _next_version(previous: str | None, bump: str) -> str:
    if previous is None:
        return "v0.1.0"
    match = _VERSION_RE.match(previous)
    if not match:
        return "v0.1.0"
    major, minor, patch = (int(g) for g in match.groups())
    if bump == "minor":
        return f"v{major}.{minor + 1}.0"
    return f"v{major}.{minor}.{patch + 1}"


def cut_version(manifest: Manifest, store: Store, notes: str = "") -> Version:
    """Record a draft version for the current build.

    Rebuilding identical inputs returns the existing version rather than
    creating a duplicate -- otherwise the version log would fill with entries
    that mean nothing and "what changed" would stop being answerable.
    """
    manifest_sha, corpus_sha, index_sha = compute_hashes(manifest, store)

    identical = store.conn.execute(
        "SELECT rowid, * FROM kb_versions "
        "WHERE manifest_sha = ? AND corpus_sha = ? AND index_sha = ? "
        f"{_VERSION_ORDER} LIMIT 1",
        (manifest_sha, corpus_sha, index_sha),
    ).fetchone()
    if identical:
        # Reusing a version means "this build *is* that version", so the
        # serving pointer must move to it. Without this, a rebuild that
        # restores an earlier corpus keeps serving under the newer version
        # number and the version log stops describing what is deployed.
        store.set_meta("current_version", identical["version"])
        return Version(
            version=identical["version"],
            created_at=identical["created_at"],
            manifest_sha=manifest_sha,
            corpus_sha=corpus_sha,
            index_sha=index_sha,
            status=identical["status"],
            notes=identical["notes"],
            stats=json.loads(identical["stats"] or "{}"),
            reused=True,
        )

    previous = _latest_row(store)
    if previous is None:
        bump = "initial"
    elif previous["manifest_sha"] != manifest_sha or previous["index_sha"] != index_sha:
        bump = "minor"   # behaviour may have changed for every question
    else:
        bump = "patch"   # only what is known has changed
    version = _next_version(previous["version"] if previous else None, bump)

    stats = store.counts()
    created = utcnow()
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO kb_versions"
            "(version, created_at, manifest_sha, corpus_sha, index_sha, status, notes, stats)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (version, created, manifest_sha, corpus_sha, index_sha, "draft", notes,
             json.dumps(stats)),
        )
    store.set_meta("current_version", version)
    return Version(
        version=version,
        created_at=created,
        manifest_sha=manifest_sha,
        corpus_sha=corpus_sha,
        index_sha=index_sha,
        status="draft",
        notes=notes,
        stats=stats,
    )


def list_versions(store: Store) -> list:
    return list(store.conn.execute(f"SELECT rowid, * FROM kb_versions {_VERSION_ORDER}"))


def published_version(store: Store) -> str | None:
    row = store.conn.execute(
        f"SELECT rowid, version FROM kb_versions WHERE status = 'published' {_VERSION_ORDER} LIMIT 1"
    ).fetchone()
    return row["version"] if row else None


def check_regressions(
    manifest: Manifest, report: dict, baseline: dict | None
) -> tuple[list[str], list[str]]:
    """Compare a report against the last published one (§12).

    Returns (metric regressions, newly failing question ids). Naming the
    newly-failing questions is what turns "investigate" into a concrete list
    rather than a feeling.
    """
    if not baseline:
        return [], []

    regressions: list[str] = []
    for name, current in report.get("metrics", {}).items():
        spec = METRICS.get(name)
        if spec is None or not spec.gated:
            continue
        if name not in baseline.get("metrics", {}):
            continue
        previous = baseline["metrics"][name]
        tolerance = manifest.evaluation.tolerance_for(name)
        if is_regression(name, current, previous, tolerance):
            direction = "fell" if spec.direction == "higher" else "rose"
            regressions.append(
                f"{name} {direction} from {previous:.3f} to {current:.3f} "
                f"(tolerance {tolerance:.3f})"
            )

    previously_passing = {r["id"] for r in baseline.get("results", []) if r.get("passed")}
    now_failing = sorted(
        r["id"] for r in report.get("results", [])
        if not r.get("passed") and r["id"] in previously_passing
    )
    return regressions, now_failing


def publish_version(
    manifest: Manifest,
    store: Store,
    version: str | None = None,
    force: bool = False,
    report: dict | None = None,
) -> PublishResult:
    """Publish a draft, if and only if the gate allows it."""
    from .evaluation.runner import load_baseline

    version = version or store.get_meta("current_version")
    if not version:
        return PublishResult("(none)", False, ["no version has been built yet"])

    row = store.conn.execute("SELECT * FROM kb_versions WHERE version = ?", (version,)).fetchone()
    if row is None:
        return PublishResult(version, False, [f"version {version} is not in the version log"])

    if report is None:
        latest = paths.reports_dir(manifest.id) / "latest.json"
        if not latest.is_file():
            return PublishResult(
                version, False,
                ["no evaluation report found; run `foundry eval` before publishing"],
            )
        report = json.loads(latest.read_text(encoding="utf-8"))

    result = PublishResult(version, published=False)

    # The report must be *this* version's. Accepting a report from another
    # build -- including a pre-versioning "unversioned" one left in var/ --
    # would let a stale pass authorise a new release, which is exactly the
    # failure the gate exists to prevent.
    if report.get("version") != version:
        result.reasons.append(
            f"the latest evaluation ran against {report.get('version')!r}, not {version}; "
            "run `foundry eval` against this build first"
        )
    if not report.get("passed"):
        failures = report.get("threshold_failures", [])
        if failures:
            result.reasons.extend(f"threshold not met: {f}" for f in failures)
        else:
            result.reasons.append("evaluation did not pass")

    baseline = load_baseline(manifest)
    result.regressions, result.newly_failing = check_regressions(manifest, report, baseline)

    blocked = bool(result.reasons or result.regressions or result.newly_failing)
    if blocked and not force:
        return result

    if blocked and force:
        result.reasons.append("PUBLISHED WITH --force despite the gate")

    with store.transaction() as conn:
        conn.execute(
            "UPDATE kb_versions SET status = 'published', eval_run_id = ? WHERE version = ?",
            (report.get("run_id"), version),
        )
    store.set_meta("current_version", version)
    store.set_meta("published_version", version)
    store.set_meta("evaluation_status", "passed" if report.get("passed") else "forced")

    _write_version_log(manifest, store, version, report)
    _save_baseline(manifest, report)

    result.published = True
    return result


def _save_baseline(manifest: Manifest, report: dict) -> None:
    path = paths.ka_var_dir(manifest.id) / "baseline.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def _write_version_log(manifest: Manifest, store: Store, version: str, report: dict) -> Path:
    """Write the published record into the knowledge area itself.

    Versions live in git alongside the sources, because the version history is
    part of the knowledge asset and renders as the *Updates* section of the
    public evidence page (§15).
    """
    directory = manifest.resolve("versions")
    directory.mkdir(parents=True, exist_ok=True)
    row = store.conn.execute("SELECT * FROM kb_versions WHERE version = ?", (version,)).fetchone()
    record = {
        "version": version,
        "published_at": utcnow(),
        "created_at": row["created_at"],
        "manifest_sha": row["manifest_sha"][:16],
        "corpus_sha": row["corpus_sha"][:16],
        "index_sha": row["index_sha"][:16],
        "notes": row["notes"],
        "stats": json.loads(row["stats"] or "{}"),
        "evaluation": {
            "run_id": report.get("run_id"),
            "passed": report.get("passed"),
            "total": report.get("total"),
            "passed_count": report.get("passed_count"),
            "pass_rate": report.get("pass_rate"),
            "metrics": report.get("metrics", {}),
            "by_type": report.get("by_type", {}),
        },
    }
    path = directory / f"{version}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def rollback_version(store: Store, version: str | None = None) -> str:
    """Repoint the serving version at a previously published build.

    Rollback is cheap because nothing was destroyed: the database still holds
    every version's record, and the index is shared.
    """
    if version is None:
        rows = [
            r for r in store.conn.execute(
                "SELECT rowid, * FROM kb_versions "
                f"WHERE status IN ('published', 'rolled_back') {_VERSION_ORDER}"
            )
        ]
        current = store.get_meta("published_version")
        candidates = [r for r in rows if r["version"] != current]
        if not candidates:
            return "no earlier published version to roll back to"
        version = candidates[0]["version"]
        with store.transaction() as conn:
            conn.execute(
                "UPDATE kb_versions SET status = 'rolled_back' WHERE version = ?", (current,)
            )
    row = store.conn.execute("SELECT * FROM kb_versions WHERE version = ?", (version,)).fetchone()
    if row is None:
        return f"version {version} is not in the version log"

    with store.transaction() as conn:
        conn.execute("UPDATE kb_versions SET status = 'published' WHERE version = ?", (version,))
    store.set_meta("current_version", version)
    store.set_meta("published_version", version)
    return f"rolled back to {version}"


def version_history(manifest: Manifest) -> list[dict]:
    """Published version records, newest first -- the Updates section (§15)."""
    directory = manifest.resolve("versions")
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("v*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    records.sort(key=lambda r: r.get("published_at", ""), reverse=True)
    return records
