"""Claim extraction: "claims derived from source" (§7).

A deliberately conservative, rule-based extractor. It does not paraphrase and it
does not infer -- a claim is a *verbatim sentence* from a chunk, classified by
shape. That matters: the claims table is used to check whether an answer is
supported, so a claim that has been reworded by a model would make the check
circular.

An LLM-backed extractor can be added behind the same signature later; the
rule-based one stays as the zero-cost default and as the CI baseline.
"""

from __future__ import annotations

import hashlib
import re

from ..text import numbers_with_units, sentences

_REQUIREMENT_RE = re.compile(
    r"\b(shall|must|is required to|are required to|may not|shall not|must not|"
    r"is prohibited|required by|mandates?|prohibits?)\b",
    re.IGNORECASE,
)
_DEFINITION_RE = re.compile(
    r"\b(is defined as|means|refers to|is the term|is a process|is a|are a)\b", re.IGNORECASE
)
_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|–|to|and)\s*(\d+(?:\.\d+)?)\s*(°C|°F|%|V|Ah|Wh|kWh|MWh|kW|MW|bar|ppm)",
    re.IGNORECASE,
)
# Sentences that are navigational or meta rather than substantive.
_NOISE_RE = re.compile(
    r"^(see also|contents|table of contents|figure \d|page \d|references?|"
    r"copyright|all rights reserved|cookies?|skip to)\b",
    re.IGNORECASE,
)

MIN_CLAIM_CHARS = 40
MAX_CLAIM_CHARS = 500


def classify(sentence: str) -> str | None:
    """Return the claim kind, or None if the sentence is not claim-bearing."""
    if _NOISE_RE.match(sentence.strip()):
        return None
    if _RANGE_RE.search(sentence):
        return "range"
    if _REQUIREMENT_RE.search(sentence):
        return "requirement"
    if numbers_with_units(sentence):
        return "fact"
    if _DEFINITION_RE.search(sentence):
        return "definition"
    return None


def extract_claims(chunk_id: str, text: str) -> list[dict]:
    """Extract claim-bearing sentences from one chunk."""
    out: list[dict] = []
    for sentence in sentences(text):
        sentence = sentence.strip()
        if not (MIN_CLAIM_CHARS <= len(sentence) <= MAX_CLAIM_CHARS):
            continue
        kind = classify(sentence)
        if kind is None:
            continue
        digest = hashlib.sha256(f"{chunk_id}:{sentence}".encode()).hexdigest()[:20]
        out.append({"id": f"cl_{digest}", "text": sentence, "kind": kind})
    return out
