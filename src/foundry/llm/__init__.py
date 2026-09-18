"""LLM provider registry.

The default is keyless and local. Choosing a hosted model is a one-line manifest
change, and nothing else in the system knows the difference.
"""

from __future__ import annotations

from ..manifest import Manifest
from .base import GenerationContext, LLMProvider, LLMResponse
from .local import ABSTENTION, LocalExtractiveProvider, ScriptedProvider

_REGISTRY = {
    "local_extractive": LocalExtractiveProvider,
    "scripted": ScriptedProvider,
}


def get_provider(provider: str = "local_extractive", model: str | None = None):
    if provider in _REGISTRY:
        cls = _REGISTRY[provider]
        return cls(model=model) if model else cls()
    if provider == "anthropic":
        from .remote import AnthropicProvider

        return AnthropicProvider(model=model or "claude-sonnet-5")
    if provider == "openai_compatible":
        from .remote import OpenAICompatibleProvider

        return OpenAICompatibleProvider(model=model or "llama3.1:8b")
    raise ValueError(
        f"unknown LLM provider {provider!r}; expected one of: "
        "local_extractive, scripted, anthropic, openai_compatible"
    )


def provider_for(manifest: Manifest):
    return get_provider(manifest.specialist.llm_provider, manifest.specialist.llm_model)


__all__ = [
    "ABSTENTION",
    "GenerationContext",
    "LLMProvider",
    "LLMResponse",
    "LocalExtractiveProvider",
    "ScriptedProvider",
    "get_provider",
    "provider_for",
]
