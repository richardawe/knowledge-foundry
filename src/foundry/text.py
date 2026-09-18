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
# A number, optionally followed by a unit written as a symbol ("100 Wh"), as a
# word ("100 watt hours"), or not at all ("10 fatalities").
_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*"
    r"(°\s?[CF]|µ?[A-Za-z]+(?:[- ][A-Za-z]+)?)?"
)

# Unit spellings that mean the same quantity, mapped to one canonical form.
#
# This table exists because of a real bug: the previous pattern matched unit
# symbols without a trailing word boundary, so "100 watt hours" yielded "100w"
# (the W of "watt") while "100 Wh" yielded "100wh". The grounding checker then
# read a correct answer as a fabricated number and withheld it -- the worst
# direction for that check to fail in. Units are now resolved by whole token,
# and spelled-out forms resolve to the same key as their symbol.
_UNIT_ALIASES = {
    "%": "%", "percent": "%", "pct": "%",
    "c": "°c", "°c": "°c", "degc": "°c", "celsius": "°c",
    "degrees celsius": "°c", "degree celsius": "°c", "deg c": "°c",
    "f": "°f", "°f": "°f", "fahrenheit": "°f", "degrees fahrenheit": "°f",
    "k": "k", "kelvin": "k",
    "v": "v", "volt": "v", "volts": "v",
    "a": "a", "amp": "a", "amps": "a", "ampere": "a", "amperes": "a",
    "ah": "ah", "amp hour": "ah", "amp hours": "ah",
    "ampere hour": "ah", "ampere hours": "ah",
    "wh": "wh", "watt hour": "wh", "watt hours": "wh", "watt-hour": "wh",
    "watt-hours": "wh", "watthour": "wh", "watthours": "wh",
    "kwh": "kwh", "kilowatt hour": "kwh", "kilowatt hours": "kwh",
    "mwh": "mwh", "megawatt hour": "mwh", "megawatt hours": "mwh",
    "w": "w", "watt": "w", "watts": "w",
    "kw": "kw", "kilowatt": "kw", "kilowatts": "kw",
    "mw": "mw", "megawatt": "mw", "megawatts": "mw",
    "mm": "mm", "millimetre": "mm", "millimetres": "mm",
    "millimeter": "mm", "millimeters": "mm",
    "cm": "cm", "centimetre": "cm", "centimetres": "cm",
    "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "km": "km", "kilometre": "km", "kilometres": "km",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g", "gramme": "g", "grammes": "g",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "s": "s", "sec": "s", "second": "s", "seconds": "s",
    "min": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hour": "h", "hours": "h",
    "bar": "bar", "kpa": "kpa", "mpa": "mpa", "pa": "pa",
    "psi": "psi", "ppm": "ppm", "ppb": "ppb",
    "µm": "µm", "micron": "µm", "microns": "µm", "micrometre": "µm",
}


def _canonical_unit(raw: str | None) -> str:
    """Resolve a unit spelling to its canonical key, or "" if it is not a unit."""
    if not raw:
        return ""
    cleaned = re.sub(r"\s+", " ", raw.strip().lower()).replace("° ", "°")
    if cleaned in _UNIT_ALIASES:
        return _UNIT_ALIASES[cleaned]
    # The pattern captures up to two words so compounds like "watt hours" and
    # "degrees celsius" resolve. When the second word is just the next word of
    # the sentence ("grams of", "m between"), fall back to the first token.
    first = cleaned.split(" ")[0].split("-")[0]
    return _UNIT_ALIASES.get(first, "")
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


def stem(token: str) -> str:
    """Reduce a token to a form that different inflections agree on.

    Not a linguistics project, and not Porter: a compact suffix stripper chosen
    so that the forms this domain actually mixes converge on one key --

        gas / gases            -> gas
        vent / vents / venting -> vent
        charge / charging      -> charg
        battery / batteries    -> battery

    The final bare-"e" strip is what makes plurals converge. Without it
    "gases" reduces to "gase" while "gas" stays "gas", and a question about
    vent gases retrieves nothing about vent gas -- which is precisely the bug
    this replaced.

    It over-stems slightly ("fire" -> "fir"). That costs a little precision and
    buys a lot of recall, and it is applied to both sides of every comparison,
    so its errors are consistent rather than biased.
    """
    if not token or not token[0].isalpha():
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        token = token[:-2]
    elif len(token) > 5 and token.endswith("ing"):
        stripped = token[:-3]
        # "running" -> "run"; the length guard keeps "string" intact.
        if len(stripped) > 2 and stripped[-1] == stripped[-2]:
            stripped = stripped[:-1]
        token = stripped
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


def content_terms(text: str, stemmed: bool = True) -> set[str]:
    """Tokens that carry meaning -- the unit of grounding comparison."""
    terms = {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}
    if stemmed:
        return {stem(t) for t in terms}
    return terms


_CITATION_ONLY_RE = re.compile(r"^(?:\[\d+\]\s*)+$")
_LEADING_CITATION_RE = re.compile(r"^((?:\[\d+\]\s*)+)(.*)$", re.DOTALL)


def sentences(text: str) -> list[str]:
    """Split into sentences, keeping trailing citation markers attached.

    A regex splitter, not a parser. It is good enough for grounding checks on
    generated prose and has no dependencies; a mis-split only ever costs a
    marginally stricter or looser support check on one sentence.

    The one case that is *not* cosmetic: models write "...exceeds 80 C. [1]",
    and a naive split turns "[1]" into its own fragment, leaving the claim
    sentence apparently uncited. The grounding checker would then grade every
    properly cited sentence against no evidence at all. Citation-only fragments
    are therefore folded back into the sentence they belong to.
    """
    out: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for part in _SENTENCE_RE.split(block):
            part = part.strip()
            if not part:
                continue
            if out and _CITATION_ONLY_RE.match(part):
                out[-1] = f"{out[-1]} {part}"
                continue
            leading = _LEADING_CITATION_RE.match(part)
            if out and leading:
                # "...80 C. [1] Venting follows." -- the marker trails the claim
                # it supports, so it belongs to the sentence before it.
                out[-1] = f"{out[-1]} {leading.group(1).strip()}"
                remainder = leading.group(2).strip()
                if remainder:
                    out.append(remainder)
                continue
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
        unit = _canonical_unit(match.group(2))
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
