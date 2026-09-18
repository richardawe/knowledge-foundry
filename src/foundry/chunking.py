"""Section-aware chunking.

Splits on document structure first and only then on size. In standards and
incident reports the heading carries most of the retrievable context -- a chunk
that says "shall not exceed 50 kW" is useless without knowing it sits under
"6.3 Indoor installation". So the section rides along with every chunk and is
indexed as its own FTS column.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .text import estimate_tokens, normalise

# Markdown ATX headings, plus the numbered-clause style that standards and
# regulations use ("4.3.2 Ventilation requirements").
_HEADING_RE = re.compile(
    r"^(?:(#{1,6})\s+(?P<md>.+)"
    r"|(?P<num>\d+(?:\.\d+){0,3})\.?\s+(?P<numtitle>[A-Z][^\n]{2,90}))$"
)
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class Chunk:
    """A retrievable unit with its position in the source document."""

    ordinal: int
    section: str
    text: str
    char_start: int
    char_end: int

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)

    def id_for(self, document_id: str) -> str:
        digest = hashlib.sha256(f"{document_id}:{self.ordinal}:{self.text}".encode()).hexdigest()
        return f"ch_{digest[:20]}"

    def as_row(self, document_id: str, source_id: str) -> dict:
        return {
            "id": self.id_for(document_id),
            "document_id": document_id,
            "source_id": source_id,
            "ordinal": self.ordinal,
            "section": self.section,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_estimate": self.token_estimate,
        }


@dataclass
class _Block:
    section: str
    text: str
    start: int


def _split_into_sections(text: str) -> list[_Block]:
    blocks: list[_Block] = []
    section = ""
    buffer: list[str] = []
    buffer_start = 0
    offset = 0

    def flush() -> None:
        if buffer:
            body = "\n".join(buffer).strip()
            if body:
                blocks.append(_Block(section=section, text=body, start=buffer_start))
        buffer.clear()

    for line in text.split("\n"):
        match = _HEADING_RE.match(line.strip())
        if match:
            flush()
            if match.group("md"):
                section = match.group("md").strip()
            else:
                section = f"{match.group('num')} {match.group('numtitle')}".strip()
            buffer_start = offset + len(line) + 1
        else:
            if not buffer:
                buffer_start = offset
            buffer.append(line)
        offset += len(line) + 1
    flush()
    return blocks


def _pack_sentences(
    body: str, target: int, overlap: int, min_chars: int
) -> list[tuple[str, int, int]]:
    """Pack sentences to the target size, never splitting mid-sentence.

    Returns (text, relative_start, relative_end) triples.
    """
    parts = [p for p in _SENT_BOUNDARY_RE.split(body) if p.strip()]
    if not parts:
        return []

    # Track each part's offset within the body so chunk spans stay real.
    offsets: list[int] = []
    cursor = 0
    for part in parts:
        index = body.find(part, cursor)
        if index < 0:
            index = cursor
        offsets.append(index)
        cursor = index + len(part)

    out: list[tuple[str, int, int]] = []
    current: list[int] = []  # indices into parts
    current_len = 0

    def emit() -> None:
        if not current:
            return
        start = offsets[current[0]]
        last = current[-1]
        end = offsets[last] + len(parts[last])
        out.append((body[start:end].strip(), start, end))

    for i, part in enumerate(parts):
        # A single oversized sentence becomes its own chunk rather than being cut.
        if current and current_len + len(part) > target:
            emit()
            if overlap > 0:
                carry: list[int] = []
                carried = 0
                for j in reversed(current):
                    if carried + len(parts[j]) > overlap:
                        break
                    carry.insert(0, j)
                    carried += len(parts[j])
                current = carry
                current_len = carried
            else:
                current, current_len = [], 0
        current.append(i)
        current_len += len(part) + 1
    emit()

    # Fold a trailing scrap into its predecessor rather than emitting a stub.
    if len(out) > 1 and len(out[-1][0]) < min_chars:
        text, start, _ = out[-2]
        tail_text, _, tail_end = out[-1]
        out[-2] = (f"{text} {tail_text}".strip(), start, tail_end)
        out.pop()
    return out


def chunk_document(
    text: str, target_chars: int = 1200, overlap_chars: int = 150, min_chars: int = 200
) -> list[Chunk]:
    """Split a document into section-aware, sentence-aligned chunks."""
    text = normalise(text)
    if not text:
        return []

    chunks: list[Chunk] = []
    ordinal = 0
    for block in _split_into_sections(text):
        for body, rel_start, rel_end in _pack_sentences(
            block.text, target_chars, overlap_chars, min_chars
        ):
            if not body:
                continue
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    section=block.section,
                    text=body,
                    char_start=block.start + rel_start,
                    char_end=block.start + rel_end,
                )
            )
            ordinal += 1
    return chunks
