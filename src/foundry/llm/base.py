"""LLM provider interface.

Deliberately small. Anything richer -- tool calling, streaming, structured
output -- would make providers hard to swap, and swappability is the point:
§4 of the brief wants frontier models used as knowledge-engineering workers
while production inference stays cheap and replaceable.

``GenerationContext`` carries the retrieved evidence *structurally* alongside
the rendered prompt. Remote providers ignore it (the evidence is already in the
prompt text); the local extractive provider uses it to compose an answer without
a model at all. That is what lets the entire loop -- including CI and the
regression gate -- run with no API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..retrieval.base import Evidence


@dataclass
class GenerationContext:
    question: str
    evidence: Sequence[Evidence] = field(default_factory=list)
    knowledge_area: str = ""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        context: GenerationContext | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> LLMResponse: ...
