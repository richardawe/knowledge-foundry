"""Shared text utilities.

Kept in one place because tokenisation choices leak into retrieval scores,
grounding decisions and evaluation grading alike. When they drift apart, the
metrics stop meaning what they say.
"""

from __future__ import annotations

import re
import unicodedata

# Words that carry no discriminative content. Small and deliberate: an
# aggressive stop list hurts technical domains, where "state of charge" and
# "end of life" are terms of art built almost entirely from stop words.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here
    is are was were be been being am do does did doing done
    have has had having will would shall should can could may might must
    of in on at to for from by with without within into onto over under
    as it its it's their they them he she his her you your we our us i
    not no nor so such about between during above below up down out off again
    further once each few more most other some any all both very
    what which who whom whose when where why how
    also however therefore thus hence per via e.g i.e etc
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-/+']*|\d+(?:[.,]\d+)*")
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%|°C|°F|C|K|V|A|Ah|Wh|kWh|MWh|kW|MW|W|mm|cm|m|km|kg|g|s|min|h|bar|kPa|MPa|psi|ppm)?", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])")
_QUOTED_RE = re.compile(r'"([^"]{4,120})"')


def normalise(text: str) -> str:
    """Unicode-normalise and collapse whitespace without destroying structure."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, hyphenated technical terms kept whole."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def content_terms(text: str) -> set[str]:
    """Tokens that carry meaning -- the unit of grounding comparison."""
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}


def sentences(text: str) -> list[str]:
    """Split into sentences.

    A regex splitter, not a parser. It is good enough for grounding checks on
    generated prose and has no dependencies; a mis-split only ever costs a
    marginally stricter or looser support check on one sentence.
    """
    out: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for part in _SENTENCE_RE.split(block):
            part = part.strip()
            if part:
                out.append(part)
    return out


def numbers_with_units(text: str) -> set[str]:
    """Extract numeric values with optional units, normalised for comparison.

    This is the cheapest effective hallucination trap in a domain full of
    thresholds and temperatures: if the answer states a number, that number had
    better appear in the evidence it cites.
    """
    found = set()
    for match in _NUMBER_RE.finditer(text):
        value = match.group(1).replace(",", "")
        try:
            numeric = float(value)
        except ValueError:
            continue
        unit = (match.group(2) or "").lower()
        # Render integers without a trailing .0 so "80" and "80.0" agree.
        rendered = str(int(numeric)) if numeric.is_integer() else str(numeric)
        found.add(f"{rendered}{unit}")
    return found


def quoted_spans(text: str) -> set[str]:
    """Double-quoted spans, which an answer implicitly claims are verbatim."""
    return {m.group(1).strip().lower() for m in _QUOTED_RE.finditer(text)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def coverage(needle: set[str], haystack: set[str]) -> float:
    """Fraction of ``needle`` present in ``haystack``.

    Asymmetric on purpose: grounding asks "is this sentence's content present in
    the evidence", not "are the two similar".
    """
    if not needle:
        return 1.0
    return len(needle & haystack) / len(needle)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate; ~4 characters per token is close enough for budgets."""
    return max(1, len(text) // 4)


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix
