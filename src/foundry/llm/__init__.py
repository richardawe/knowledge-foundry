"""LLM provider registry.

The default is keyless and local. Choosing a hosted model is a one-line manifest
change, and nothing else in the system knows the difference.
"""

from __future__ import annotations

import os

from ..manifest import Manifest
from .base import GenerationContext, LLMProvider, LLMResponse
from .local import ABSTENTION, FallbackProvider, LocalExtractiveProvider, ScriptedProvider

_REGISTRY = {
    "local_extractive": LocalExtractiveProvider,
    "scripted": ScriptedProvider,
}


def get_provider(provider: str = "local_extractive", model: str | None = None):
    if provider in _REGISTRY:
        cls = _REGISTRY[provider]
        return cls(model=model) if model else cls()
    if provider == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(model=model or "llama3.1:8b")
    if provider == "anthropic":
        from .remote import AnthropicProvider

        return AnthropicProvider(model=model or "claude-sonnet-5")
    if provider == "openai_compatible":
        from .remote import OpenAICompatibleProvider

        return OpenAICompatibleProvider(model=model or "llama3.1:8b")
    raise ValueError(
        f"unknown LLM provider {provider!r}; expected one of: "
        "local_extractive, scripted, ollama, openai_compatible, anthropic"
    )


class KeylessProviderRefused(RuntimeError):
    """Raised when a keyless provider is asked to stand in for a real model."""


# Providers that reach the network and can therefore be unavailable.
REMOTE_PROVIDERS = frozenset({"ollama", "openai_compatible", "anthropic"})

# Providers that need no key, no daemon and no network. They cannot fail, which
# is exactly why nothing may quietly substitute them for a model.
KEYLESS_PROVIDERS = frozenset({"local_extractive", "scripted"})


def provider_name_for(manifest: Manifest) -> tuple[str, str]:
    """The provider and model this process will actually use, after overrides."""
    cfg = manifest.specialist
    return (
        os.environ.get("FOUNDRY_LLM_PROVIDER", "").strip() or cfg.llm_provider,
        os.environ.get("FOUNDRY_LLM_MODEL", "").strip() or cfg.llm_model,
    )


def provider_for(manifest: Manifest, allow_fallback: bool = True, require_model: bool = False):
    """Build the knowledge area's provider, wrapped so an outage degrades rather than breaks.

    A knowledge area pointing at a local Ollama should keep working -- with a
    weaker answer and a recorded note -- on a machine where Ollama is not
    running. Set ``specialist.llm.fallback: false`` in the manifest to make an
    unreachable model a hard error instead.
    """
    cfg = manifest.specialist
    # An environment override, for machines that cannot run the knowledge
    # area's chosen model. A CI runner has no Ollama, and saying so up front is
    # better than a hundred connection failures that each look like an outage.
    name, model = provider_name_for(manifest)
    if require_model and name in KEYLESS_PROVIDERS:
        raise KeylessProviderRefused(
            f"{name!r} answers from the corpus without calling a model, so it cannot "
            "fill the inference queue -- its replies would be indistinguishable from "
            f"{cfg.llm_provider!r}'s once applied. Unset FOUNDRY_LLM_PROVIDER on the "
            "machine running the worker, or point it at a real model."
        )
    provider = get_provider(name, model)
    if allow_fallback and cfg.llm_fallback and name in REMOTE_PROVIDERS:
        return FallbackProvider(provider)
    return provider


__all__ = [
    "ABSTENTION",
    "FallbackProvider",
    "KEYLESS_PROVIDERS",
    "KeylessProviderRefused",
    "REMOTE_PROVIDERS",
    "GenerationContext",
    "LLMProvider",
    "LLMResponse",
    "LocalExtractiveProvider",
    "ScriptedProvider",
    "get_provider",
    "provider_for",
    "provider_name_for",
]
