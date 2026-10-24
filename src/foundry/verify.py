"""Verify claims made elsewhere against a knowledge area, and say who supports them.

``Specialist.ask`` answers a question. This answers a different one: *somebody
else* has said something -- another system's summary, a generated map, an
LLM's draft -- and the question is whether the corpus bears it out, and on whose
word. The answer is a verdict per claim:

* ``corroborated`` -- supported by a source that is not the subject speaking
  about itself (a register, the press, users).
* ``self_reported`` -- supported only by the subject's own publication.
* ``label_only`` -- the words appear, but only as a short label (a menu item,
  a button), never in a sentence. Present is not the same as meant.
* ``not_found`` -- nothing in the corpus supports it.

The check is exhaustive over the knowledge area's chunks rather than retrieval:
a verifier that misses support because of a ranking cut-off reports a gap that
is not there, and a per-instance corpus is small enough to read whole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .storage import Store
from .text import content_terms, coverage

SELF_TYPES = frozenset({"self_published"})
# Sources accepted as being about the subject by construction: a register record
# is collected only once the entity is matched. Anything else -- press, users --
# corroborates only in a passage that names the subject, or a comment about some
# other company's support team would "confirm" this one's.
ABOUT_SUBJECT_TYPES = frozenset({"self_published", "official_record"})
LABEL_MAX_WORDS = 5


@dataclass
class Support:
    source_id: str
    source_type: str
    title: str
    uri: str
    support: float
    quote: str
    unconfirmed_identity: bool = False

    def as_dict(self) -> dict:
        return {"source_id": self.source_id, "source_type": self.source_type,
                "title": self.title, "uri": self.uri, "support": round(self.support, 3),
                "quote": self.quote, "unconfirmed_identity": self.unconfirmed_identity}


@dataclass
class Verdict:
    claim: str
    verdict: str
    support: list[Support] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {"claim": self.claim, "verdict": self.verdict, "note": self.note,
                "support": [s.as_dict() for s in self.support]}


def _quote(text: str, terms: set[str], width: int = 240) -> str:
    """The sentence (or line) of ``text`` that carries most of ``terms``."""
    best, best_score = "", -1.0
    for piece in re.split(r"(?<=[.!?])\s+|\n+", text):
        piece = piece.strip()
        if not piece:
            continue
        score = coverage(terms, content_terms(piece))
        if score > best_score:
            best, best_score = piece, score
    return best[:width]


def _phrase_lines(text: str, phrase: str) -> list[str]:
    needle = phrase.lower()
    return [line.strip() for line in text.splitlines() if needle in line.lower()]


def threshold_for(terms: set[str]) -> float:
    """Short claims must be carried whole; long ones mostly."""
    return 1.0 if len(terms) <= 3 else 0.6


def _words(text: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def _sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+|\s+\|\s+", text) if p.strip()]


def stating_sentence(text: str, claim: str, terms: set[str], subject: str | None) -> str | None:
    """The sentence of ``text`` that states ``claim`` about ``subject``, if there is one.

    Corroboration is a property of a sentence, not of a passage: a passage that
    mentions the subject in one place and the claim's words in another (a job ad
    listing many employers, a news roundup) states nothing about the subject. A
    short claim must appear as a phrase; a longer one must have most of its
    content in the sentence. Either way the sentence must name the subject.
    """
    phrase = _words(claim)
    for sentence in _sentences(text):
        if subject and _words(subject) not in _words(sentence):
            continue
        if len(terms) <= 3:
            if phrase in _words(sentence):
                return sentence
        elif coverage(terms, content_terms(sentence)) >= 0.6:
            return sentence
    return None


ENTAILMENT_SYSTEM = (
    "You check whether a passage states a claim. Answer with exactly one word: "
    "YES if the passage itself states the claim about the named company, NO otherwise. "
    "Mentioning the same words is not stating the claim."
)


def model_confirms(provider, subject: str | None, claim: str, sentence: str) -> tuple[bool, str]:
    """Ask a model whether the sentence states the claim. Any failure counts as no."""
    about = f" about {subject}" if subject else ""
    response = provider.complete(
        ENTAILMENT_SYSTEM,
        f"Claim{about}: {claim}\n\nPassage: {sentence}\n\nDoes the passage state the claim?",
        temperature=0.0, max_tokens=3)
    if response.error:
        return False, f"model error: {response.error}"
    return response.text.strip().upper().startswith("YES"), f"{response.provider}:{response.model}"


def entailment_from_env():
    """The model for the entailment check, from ``FOUNDRY_VERIFY_PROVIDER`` / ``_MODEL``.

    Returns ``None`` when unset. A keyless provider is refused outright: it
    would answer every check without reading anything, and its "YES" would be
    recorded as a model's judgement.
    """
    import os

    from .llm import get_provider

    name = os.environ.get("FOUNDRY_VERIFY_PROVIDER")
    if not name:
        return None
    if name in ("local_extractive", "scripted"):
        raise ValueError(f"FOUNDRY_VERIFY_PROVIDER={name!r} calls no model; refusing it as a verifier")
    return get_provider(name, os.environ.get("FOUNDRY_VERIFY_MODEL"))


def verify_claims(store: Store, claims: list[str], max_support: int = 3,
                  subject: str | None = None, entailment=None) -> list[Verdict]:
    """One verdict per claim.

    ``entailment`` is an optional model provider. Without one, corroboration
    rests on the sentence rule alone; with one, each candidate sentence must
    also be confirmed by the model, and the verdict says which model.
    """
    rows = store.chunks_with_provenance([row["id"] for row in store.chunks()])
    unconfirmed = {
        row["id"] for row in store.sources()
        if "identity-unconfirmed" in (row["tags"] or "")
    }
    # A heading is stored as the chunk's section, not its text; it is still
    # something the page says, and the clearest case of a label.
    chunks = [(row, content_terms(f"{row['section'] or ''} {row['text']}")) for row in rows.values()]

    verdicts = []
    for claim in claims:
        terms = content_terms(claim)
        if not terms:
            verdicts.append(Verdict(claim, "not_found", note="nothing checkable in the claim"))
            continue
        needed = threshold_for(terms)
        best: dict[str, Support] = {}
        for row, chunk_terms in chunks:
            score = coverage(terms, chunk_terms)
            if score < needed:
                continue
            if (subject and row["source_type"] not in ABOUT_SUBJECT_TYPES
                    and subject.lower() not in row["text"].lower()):
                continue
            current = best.get(row["source_id"])
            if current is None or score > current.support:
                best[row["source_id"]] = Support(
                    source_id=row["source_id"], source_type=row["source_type"],
                    title=row["source_title"], uri=row["uri"], support=score,
                    quote=_quote(row["text"], terms),
                    unconfirmed_identity=row["source_id"] in unconfirmed)
        supports = sorted(best.values(), key=lambda s: (s.source_type in SELF_TYPES, -s.support))
        own = [s for s in supports if s.source_type in SELF_TYPES]
        independent, checked_by = [], "sentence rule"
        for candidate in (s for s in supports if s.source_type not in SELF_TYPES):
            text = "\n".join(r["text"] for r, _ in chunks if r["source_id"] == candidate.source_id)
            sentence = stating_sentence(text, claim, terms, subject)
            if sentence is None:
                continue
            if entailment is not None:
                confirmed, by = model_confirms(entailment, subject, claim, sentence)
                checked_by = f"sentence rule + {by}"
                if not confirmed:
                    continue
            candidate.quote = sentence[:240]
            independent.append(candidate)

        # A one-word claim ("Finance", "Research") is in nearly every article;
        # finding the word elsewhere corroborates nothing.
        if independent and len(terms) > 1:
            verdicts.append(Verdict(claim, "corroborated", independent[:max_support] + own[:1],
                                    note=f"checked by {checked_by}"))
            continue
        if not own:
            note = "a single word, found only in third-party sources" if independent else ""
            verdicts.append(Verdict(claim, "not_found", note=note))
            continue
        # Present in the subject's own pages. In a sentence, or only as a label?
        if len(terms) <= 3:
            lines = [line for s in own for line in _phrase_lines("\n".join(
                f"{r['section'] or ''}\n{r['text']}" for r, _ in chunks
                if r["source_id"] == s.source_id), claim)]
            if lines and all(len(line.split()) <= LABEL_MAX_WORDS for line in lines):
                verdicts.append(Verdict(claim, "label_only", own[:max_support],
                                        note=f"appears only as a short label ({len(lines)} times)"))
                continue
        verdicts.append(Verdict(claim, "self_reported", own[:max_support]))
    return verdicts
