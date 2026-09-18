"""Prompt construction.

The knowledge area owns its *system* prompt -- it is domain knowledge and lives
in ``knowledge_areas/<id>/prompts/system.md``. This module owns the *evidence*
prompt, which is domain-agnostic machinery: how passages are numbered, what
provenance travels with them, and the standing instructions that make an answer
checkable.
"""

from __future__ import annotations

from typing import Sequence

from ..retrieval.base import Evidence
from ..text import truncate

# Appended to every knowledge area's own system prompt. These rules are not
# domain knowledge -- they are the contract the validator enforces, so they
# belong to the factory rather than to any one knowledge area.
STANDING_RULES = """
--- STANDING RULES (enforced by automated validation after you answer) ---
1. Use ONLY the numbered evidence below. If it is not in the evidence, you do
   not know it.
2. Cite with [n] after every sentence that makes a factual claim. Sentences
   without a citation are reported as unsupported claims.
3. Never state a number, threshold, date or quotation that does not appear in
   the evidence you cite for it.
4. Separate what the evidence SAYS from what you INFER. Mark inference
   explicitly, for example: "Inference (not stated in the sources): ...".
5. If sources disagree, say so and cite both. Do not silently pick one.
6. If the evidence is insufficient, say exactly that and state what would be
   needed. An honest "the evidence does not establish this" is a correct
   answer; a confident unsupported answer is a failure.
7. Answer the question that was asked. If it rests on a false premise, say so
   before answering.
"""


def format_evidence(evidence: Sequence[Evidence], max_chars_per_item: int = 1600) -> str:
    """Number the passages and attach the provenance a reader would want."""
    if not evidence:
        return "(no evidence retrieved)"
    blocks = []
    for item in evidence:
        header = f"[{item.rank}] {item.source_title or item.document_title} — {item.publisher}"
        details = []
        if item.published_at:
            details.append(f"published {item.published_at}")
        if item.section:
            details.append(f"section: {item.section}")
        details.append(f"type: {item.source_type}")
        details.append(f"authority: {item.authority}/5")
        blocks.append(
            f"{header}\n({'; '.join(details)})\n{truncate(item.text, max_chars_per_item)}"
        )
    return "\n\n".join(blocks)


def build_system_prompt(knowledge_area_prompt: str) -> str:
    return f"{knowledge_area_prompt.strip()}\n{STANDING_RULES}"


def build_user_prompt(question: str, evidence: Sequence[Evidence]) -> str:
    return (
        f"QUESTION\n{question}\n\n"
        f"EVIDENCE\n{format_evidence(evidence)}\n\n"
        "Answer the question using only the evidence above, citing with [n]."
    )
