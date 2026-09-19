"""Expert challenges (§14).

The lab does not need the founder to hold a PhD in every domain. It needs the
system to be challengeable, and it needs a challenge to go somewhere.

So a challenge follows the same path a red-team break does: it is recorded with
the answer it disputes, and when accepted it is **promoted into the evaluation
suite** as a graded test case. Expert feedback becomes a dataset rather than an
inbox, and the same complaint cannot be made twice about a live system.
"""

from __future__ import annotations

import datetime as _dt
import hashlib

import yaml

from .manifest import Manifest
from .storage import Store, utcnow

CHALLENGE_SUITE = "evaluation/regression_from_challenges.yaml"

OPEN = "open"
ACCEPTED = "accepted"
REJECTED = "rejected"
PROMOTED = "promoted"


def add_challenge(
    manifest: Manifest,
    store: Store,
    question: str,
    claimed_problem: str = "",
    submitted_by: str = "anonymous",
    expected: str = "",
    system_answer: str = "",
    challenge_id: str | None = None,
) -> str:
    """Record a challenge, capturing what the system said at the time.

    The answer is stored with the complaint because a challenge that cannot be
    reproduced later is not evidence of anything -- and the system's answer
    will change as the knowledge area is rebuilt.

    ``challenge_id`` lets a caller supply a stable id derived from wherever the
    challenge arrived from. A submission that can be replayed -- a workflow
    re-run, a webhook delivered twice -- must land on the same row rather than
    accumulating near-duplicate complaints about one question.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("a challenge needs a question")

    if not challenge_id:
        digest = hashlib.sha256(f"{question}{submitted_by}{utcnow()}".encode()).hexdigest()[:12]
        challenge_id = f"ch_{digest}"
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO challenges"
            "(id, submitted_by, question, claimed_problem, expected, system_answer,"
            " status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET"
            " submitted_by=excluded.submitted_by, question=excluded.question,"
            " claimed_problem=excluded.claimed_problem, expected=excluded.expected,"
            " system_answer=excluded.system_answer",
            (
                challenge_id, submitted_by or "anonymous", question,
                claimed_problem, expected, system_answer, OPEN, utcnow(),
            ),
        )
    return challenge_id


def list_challenges(store: Store, status: str | None = None) -> list:
    if status:
        return list(
            store.conn.execute(
                "SELECT * FROM challenges WHERE status = ? ORDER BY created_at DESC", (status,)
            )
        )
    return list(store.conn.execute("SELECT * FROM challenges ORDER BY created_at DESC"))


def set_status(store: Store, challenge_id: str, status: str, notes: str = "") -> None:
    if status not in (OPEN, ACCEPTED, REJECTED, PROMOTED):
        raise ValueError(f"unknown challenge status {status!r}")
    store.conn.execute(
        "UPDATE challenges SET status = ?, notes = ? WHERE id = ?",
        (status, notes, challenge_id),
    )
    store.conn.commit()


def _graders_for(expected: str | None, claimed_problem: str) -> list[dict]:
    """Choose graders from what the expert said was wrong.

    A challenge that does not say what the right behaviour is can still be
    recorded, but it becomes a citation-discipline test rather than a content
    test -- asserting nothing would be worse than asserting something modest.
    """
    text = f"{expected or ''} {claimed_problem}".lower()
    if expected:
        if any(word in text for word in ("should not answer", "should decline", "no evidence",
                                         "not knowable", "unanswerable", "out of scope")):
            return [{"kind": "must_abstain"}]
        return [
            {"kind": "must_include", "any_of": [expected.strip()]},
            {"kind": "citations_resolve", "value": 1.0},
        ]
    if any(word in text for word in ("hallucinat", "made up", "fabricat", "invented")):
        return [{"kind": "max_unsupported", "value": 0.0}]
    if any(word in text for word in ("wrong source", "citation", "cited")):
        return [{"kind": "citations_resolve", "value": 1.0}]
    return [{"kind": "citations_resolve", "value": 1.0}, {"kind": "max_unsupported", "value": 0.34}]


def promote_challenge(
    manifest: Manifest, store: Store, challenge_id: str, expected: str | None = None
) -> str:
    """Turn an accepted challenge into a permanent, graded test case (§24.9)."""
    row = store.conn.execute(
        "SELECT * FROM challenges WHERE id = ?", (challenge_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"no challenge {challenge_id!r}")

    expected = expected if expected is not None else (row["expected"] or "")
    path = manifest.resolve(CHALLENGE_SUITE)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = {
            "suite": "regression_from_challenges",
            "description": (
                "Questions submitted by external experts and practitioners that the "
                "system got wrong (§14). Promoted so the same complaint cannot be "
                "made twice about a live system."
            ),
            "questions": [],
        }
    questions = data.setdefault("questions", [])
    if any(q["question"].strip().lower() == row["question"].strip().lower() for q in questions):
        set_status(store, challenge_id, PROMOTED, "already present in the challenge suite")
        return str(path)

    questions.append({
        "id": f"chq-{challenge_id[3:]}",
        "type": "adversarial",
        "question": row["question"],
        "origin": "challenge",
        "notes": (
            f"Submitted by {row['submitted_by']} on {row['created_at'][:10]}; "
            f"promoted {_dt.date.today().isoformat()}. "
            f"Reported problem: {row['claimed_problem'] or 'not stated'}."
        ),
        "graders": _graders_for(expected, row["claimed_problem"] or ""),
    })
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )

    with store.transaction() as conn:
        conn.execute(
            "UPDATE challenges SET status = ?, promoted_to = ? WHERE id = ?",
            (PROMOTED, str(path), challenge_id),
        )
    return str(path)


def challenge_stats(store: Store) -> dict:
    rows = store.conn.execute(
        "SELECT status, COUNT(*) AS n FROM challenges GROUP BY status"
    ).fetchall()
    counts = {row["status"]: row["n"] for row in rows}
    counts["total"] = sum(counts.values())
    return counts


# -- repository-durable challenges (§14) ---------------------------------
#
# A challenge submitted from outside must outlive the build cache. The store
# lives in ``var/``, which is disposable by design -- it is rebuilt from the
# source register and thrown away between runners. A complaint from a person
# who knows the domain is not derivable from the sources, so if it lives only
# there, a cache eviction silently destroys the one dataset the factory cannot
# regenerate.
#
# So the repository holds the record and the store holds a copy. Same reasoning
# as the inference queue: state that arrives from outside travels in the
# repository, because that is the only thing here that is actually durable.

CHALLENGE_DIRNAME = "challenges"
_CHALLENGE_FIELDS = (
    "id", "submitted_by", "question", "claimed_problem", "expected",
    "system_answer", "status", "created_at", "promoted_to", "notes",
)


def challenges_dir(manifest: Manifest):
    return manifest.resolve(CHALLENGE_DIRNAME)


def write_challenge_file(manifest: Manifest, record: dict) -> str:
    """Persist one challenge as repository state, one file per challenge."""
    directory = challenges_dir(manifest)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['id']}.yaml"
    payload = {key: record.get(key) or "" for key in _CHALLENGE_FIELDS}
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    return str(path)


def read_challenge_files(manifest: Manifest) -> list[dict]:
    directory = challenges_dir(manifest)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("id") and data.get("question"):
            records.append(data)
    return records


def import_challenges(manifest: Manifest, store: Store) -> int:
    """Restore repository-held challenges into a freshly built store.

    Idempotent, and the repository wins: a rebuild must not resurrect a
    challenge that was deleted, nor drop the status a review has since given
    one. Called from ``build`` so the public page survives a cold runner.
    """
    records = read_challenge_files(manifest)
    if not records:
        return 0
    with store.transaction() as conn:
        for record in records:
            conn.execute(
                "INSERT INTO challenges"
                "(id, submitted_by, question, claimed_problem, expected, system_answer,"
                " status, created_at, promoted_to, notes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET"
                " submitted_by=excluded.submitted_by, question=excluded.question,"
                " claimed_problem=excluded.claimed_problem, expected=excluded.expected,"
                " system_answer=excluded.system_answer, status=excluded.status,"
                " promoted_to=excluded.promoted_to, notes=excluded.notes",
                (
                    record["id"], record.get("submitted_by") or "anonymous",
                    record["question"], record.get("claimed_problem") or "",
                    record.get("expected") or "", record.get("system_answer") or "",
                    record.get("status") or OPEN, record.get("created_at") or utcnow(),
                    record.get("promoted_to") or None, record.get("notes") or "",
                ),
            )
    return len(records)
