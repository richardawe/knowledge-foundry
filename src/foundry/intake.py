"""Questions that arrive from outside, answered with no server running (§16).

The public evidence pages are static, so the ask box on them cannot answer:
retrieval needs the corpus, the gates and the validator, all of which are
Python. The usual fix is to run a host. This module takes the other route --
the one the inference queue already proved -- and treats the repository's own
automation as the backend.

    a person opens an issue
        -> the issue body is parsed into a submission
        -> the corpus answers it, through the same gates as everything else
        -> the answer is validated and scored exactly as the suite's answers are
        -> the exchange is recorded as a challenge, in the repository
        -> the reply is posted back, and the public page shows it

Nothing about this is live. It takes as long as a workflow takes, and that is
the honest trade: no host, no key in a browser, no unlogged conversations.

What it buys is worth more than the latency. §14 asks that a challenge *go
somewhere* -- that expert feedback become a dataset rather than an inbox. A
chat box produces an exchange nobody can audit afterwards; this produces a
durable, citable record with the evidence the answer rested on, sitting in the
same repository as the tests it can be promoted into.

**The question is untrusted input.** It reaches a model, so it can attempt to
instruct one. That is not a new hole: the answer is still validated against the
retrieved passages afterwards, so an answer talked into ignoring the evidence
fails grounding and is withheld like any other. The guards here are the ones
validation cannot provide -- a length bound, a known knowledge area, and a
refusal to answer at all when no real model is configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import provider_for
from .manifest import Manifest, list_knowledge_areas
from .specialist import Specialist
from .specialist.answer import ANSWERED, Answer
from .storage import Store

# Long enough for a genuinely technical question with its conditions spelled
# out; short enough that nobody pastes a document into the prompt.
MAX_QUESTION_CHARS = 1500
MAX_FIELD_CHARS = 600

# What GitHub writes into the body for an optional field left blank.
_NO_RESPONSE = "_no response_"

# Issue-form labels, normalised, mapped to submission fields. Matching on a
# prefix keeps the form's wording editable without breaking the parser.
_FIELD_PREFIXES = (
    ("knowledge area", "knowledge_area"),
    ("your question", "question"),
    ("question", "question"),
    ("what do you think is wrong", "problem"),
    ("what should it have said", "expected"),
)

_HEADING_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


class IntakeError(RuntimeError):
    """A submission that cannot be answered, with a reason fit to post back."""


@dataclass
class Submission:
    knowledge_area: str
    question: str
    problem: str = ""
    expected: str = ""
    submitted_by: str = "anonymous"
    reference: str = ""          # where it came from, e.g. an issue URL


@dataclass
class IntakeResult:
    submission: Submission
    answer: Answer
    challenge_id: str
    challenge_path: str = ""
    comment: str = ""

    def as_dict(self) -> dict:
        return {
            "knowledge_area": self.submission.knowledge_area,
            "question": self.submission.question,
            "submitted_by": self.submission.submitted_by,
            "challenge_id": self.challenge_id,
            "challenge_path": self.challenge_path,
            "answer": self.answer.as_dict(),
        }


# -- parsing -------------------------------------------------------------


def parse_issue_form(body: str) -> dict:
    """Split a rendered issue form into ``{field: value}``.

    GitHub renders each form field as an ``### Label`` heading followed by the
    value, so the body is parsed rather than the API queried: the workflow
    already has the body, and one less API call is one less thing to fail.
    """
    fields: dict[str, str] = {}
    body = (body or "").replace("\r\n", "\n")
    headings = list(_HEADING_RE.finditer(body))
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        label = match.group(1).strip().lower().rstrip("?:").strip()
        value = body[match.end():end].strip()
        if value.lower() == _NO_RESPONSE:
            value = ""
        for prefix, name in _FIELD_PREFIXES:
            if label.startswith(prefix) and name not in fields:
                fields[name] = value
                break
    return fields


def submission_from_body(
    body: str, *, submitted_by: str = "anonymous", reference: str = ""
) -> Submission:
    """Validate a rendered issue form into something answerable, or refuse it.

    Every refusal here is phrased to be posted back to the person, so the
    reason has to be actionable rather than a stack trace.
    """
    fields = parse_issue_form(body)
    question = " ".join((fields.get("question") or "").split())
    if not question:
        raise IntakeError(
            "I could not find a question in this issue. Please use the "
            "**Ask the specialist** issue form so the question lands in a field "
            "I can read."
        )
    if len(question) > MAX_QUESTION_CHARS:
        raise IntakeError(
            f"That question is {len(question)} characters and the limit is "
            f"{MAX_QUESTION_CHARS}. Please ask one question at a time -- a long "
            "prompt retrieves worse evidence, not more."
        )

    area = (fields.get("knowledge_area") or "").strip()
    known = list_knowledge_areas()
    if area not in known:
        listed = "\n".join(f"* `{item}`" for item in known)
        raise IntakeError(
            f"`{area or '(none given)'}` is not a knowledge area in this repository. "
            f"The ones that exist are:\n\n{listed}"
        )

    return Submission(
        knowledge_area=area,
        question=question,
        problem=(fields.get("problem") or "")[:MAX_FIELD_CHARS].strip(),
        expected=(fields.get("expected") or "")[:MAX_FIELD_CHARS].strip(),
        submitted_by=submitted_by or "anonymous",
        reference=reference,
    )


# -- answering -----------------------------------------------------------


def handle(
    manifest: Manifest,
    store: Store,
    submission: Submission,
    *,
    challenge_id: str,
    require_model: bool = False,
    provider=None,
) -> IntakeResult:
    """Answer a submission, record it, and render the reply.

    ``require_model`` refuses the keyless providers, for the same reason the
    worker does: an answer published under someone's question, in reply to
    them, should not be a corpus extract wearing a model's name. It is a
    legitimate answer -- it is what the whole published site is built from --
    but it has to say so, and the safest way to guarantee that is to make the
    caller choose it deliberately.
    """
    from .feedback import add_challenge, write_challenge_file

    if provider is None:
        provider = provider_for(manifest, require_model=require_model)

    answer = Specialist(manifest, store, provider=provider).ask(submission.question)

    add_challenge(
        manifest,
        store,
        question=submission.question,
        claimed_problem=submission.problem,
        submitted_by=submission.submitted_by,
        expected=submission.expected,
        system_answer=answer.answer,
        challenge_id=challenge_id,
    )
    row = store.conn.execute(
        "SELECT * FROM challenges WHERE id = ?", (challenge_id,)
    ).fetchone()
    record = dict(row)
    if submission.reference:
        record["notes"] = f"Submitted via {submission.reference}"
        store.conn.execute(
            "UPDATE challenges SET notes = ? WHERE id = ?", (record["notes"], challenge_id)
        )
        store.conn.commit()
    path = write_challenge_file(manifest, record)

    result = IntakeResult(
        submission=submission, answer=answer, challenge_id=challenge_id, challenge_path=path
    )
    result.comment = render_comment(manifest, result)
    return result


# -- the reply -----------------------------------------------------------


def _provenance(answer: Answer) -> str:
    """Name what produced the answer, and flag it when that was not the model.

    ``FallbackProvider`` writes the substitution into the provider string
    rather than a flag, so that a stored answer stays self-describing. Read it
    the same way here: this reply is being handed to someone who is about to
    judge it, and "which model said this" is the first thing they need.
    """
    llm = answer.llm or {}
    provider = llm.get("provider") or "unknown"
    model = llm.get("model") or "unknown"
    line = f"`{provider}` / `{model}`"
    if "fell back" in provider.lower() or llm.get("error"):
        line += (
            "\n\n> ⚠️ **The configured model did not answer this.** A fallback "
            "provider did, and it is named above. Judge the answer accordingly."
        )
    return line


def render_comment(manifest: Manifest, result: IntakeResult) -> str:
    """The reply, in the §10 evidence-first shape, as issue markdown.

    Provenance is not a footnote here. Whoever reads this is being asked to
    judge the answer, and they cannot do that without knowing what produced it,
    which corpus version it saw, and whether that version passes its own tests.
    """
    answer = result.answer
    out: list[str] = []

    if answer.status != ANSWERED:
        out.append(
            f"**The system declined to answer.** Status: `{answer.status}`.\n\n"
            "That is a designed outcome, not an error -- three gates can end a "
            "question without an answer. If you think this one should have been "
            "answerable from the sources, say so below: that disagreement is "
            "more useful than the answer would have been."
        )
        out.append("")

    out.append("## Answer")
    out.append("")
    out.append(answer.answer.strip() or "_(empty)_")

    if answer.reasoning.strip():
        out.append("")
        out.append("## Reasoning")
        out.append("")
        out.append(answer.reasoning.strip())

    cited = [s for s in answer.sources if s.get("cited")]
    if cited:
        out.append("")
        out.append("## Sources")
        out.append("")
        for source in cited:
            title = source.get("source_title") or source.get("source_id") or "source"
            uri = source.get("uri") or ""
            publisher = source.get("publisher") or ""
            label = f"[{title}]({uri})" if uri else title
            detail = " — ".join(x for x in (publisher, source.get("published_at") or "") if x)
            out.append(f"- `[{source['rank']}]` {label}{(' — ' + detail) if detail else ''}")

    out.append("")
    out.append("## Evidence confidence")
    out.append("")
    out.append(
        f"**{answer.confidence}** (score {answer.confidence_score:.2f}) — "
        f"citations valid {answer.citation_validity:.0%}, "
        f"claims grounded {answer.grounded_ratio:.0%}"
    )

    if answer.unsupported_claims:
        out.append("")
        out.append("## Claims that could not be traced to the evidence")
        out.append("")
        for claim in answer.unsupported_claims:
            out.append(f"- {claim['sentence']}  _({claim['reason']})_")

    if answer.contradictions:
        out.append("")
        out.append("## Contradictions between sources")
        out.append("")
        for item in answer.contradictions:
            out.append(f"- {item['description']}")

    if answer.limitations:
        out.append("")
        out.append("## Limitations")
        out.append("")
        for limitation in answer.limitations:
            out.append(f"- {limitation}")

    out.append("")
    out.append("---")
    out.append("")
    out.append(
        f"Answered by **{manifest.name}** `{answer.knowledge_version}` "
        f"(evaluation: `{answer.evaluation_status}`) using {_provenance(answer)}."
    )
    out.append("")
    out.append(
        f"Recorded as challenge `{result.challenge_id}`. **If this is wrong, say so in a "
        "reply** — state what it should have said. An accepted challenge is promoted "
        "into the evaluation suite, which means the same mistake cannot be made twice "
        "without a test failing."
    )
    return "\n".join(out)


def render_refusal(error: Exception) -> str:
    """A refusal, in the same shape, so nobody gets silence."""
    return (
        "## I could not answer this\n\n"
        f"{error}\n\n"
        "---\n\n"
        "Edit the issue and I will try again, or open a new one with the "
        "**Ask the specialist** form."
    )
