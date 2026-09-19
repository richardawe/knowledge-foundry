"""Provider registry and the Ollama fallback.

The knowledge areas name a local Ollama as their provider. The suite, CI and
any machine without Ollama must still answer, evaluate and publish -- and it
must be visible afterwards which model actually produced an answer.
"""

from __future__ import annotations

import json

import pytest

from foundry.llm import (
    FallbackProvider,
    GenerationContext,
    LLMResponse,
    ScriptedProvider,
    get_provider,
    provider_for,
)
from foundry.retrieval.base import Evidence
from foundry.specialist import Specialist


class Unreachable:
    name, model = "openai_compatible", "llama3.1:8b"

    def complete(self, system, user, **kwargs):
        return LLMResponse(text="", provider=self.name, model=self.model,
                           error="[Errno 111] Connection refused")


class Empty:
    name, model = "openai_compatible", "llama3.1:8b"

    def complete(self, system, user, **kwargs):
        return LLMResponse(text="   ", provider=self.name, model=self.model)


def _context() -> GenerationContext:
    return GenerationContext(
        question="At what core temperature does overheating begin?",
        evidence=[Evidence(rank=1, chunk_id="c1", score=1.0,
                           text="Widget overheating begins when the core temperature exceeds 80 °C.",
                           section="1")],
    )


def test_registry_knows_the_ollama_shaped_provider():
    provider = get_provider("openai_compatible", "llama3.1:8b")
    assert provider.name == "openai_compatible"
    assert provider.model == "llama3.1:8b"
    # Ollama's OpenAI-compatible endpoint is the default target.
    assert provider.base_url.endswith(":11434/v1")


def test_unknown_provider_is_a_loud_error():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        get_provider("telepathy")


def test_manifest_provider_is_wrapped_for_fallback(manifest):
    manifest.specialist.llm_provider = "openai_compatible"
    manifest.specialist.llm_model = "llama3.1:8b"
    provider = provider_for(manifest)

    assert isinstance(provider, FallbackProvider)
    assert provider.primary.name == "openai_compatible"


def test_local_provider_is_not_wrapped(manifest):
    manifest.specialist.llm_provider = "local_extractive"
    assert not isinstance(provider_for(manifest), FallbackProvider)


def test_fallback_can_be_switched_off(manifest):
    """So an operator can make an unreachable model a hard failure."""
    manifest.specialist.llm_provider = "openai_compatible"
    manifest.specialist.llm_fallback = False
    assert not isinstance(provider_for(manifest), FallbackProvider)


def test_an_unreachable_model_degrades_rather_than_breaks():
    provider = FallbackProvider(Unreachable())
    response = provider.complete("sys", "user", context=_context())

    assert response.ok
    assert "80" in response.text
    assert provider.fell_back


def test_the_fallback_is_recorded_not_hidden():
    """An evaluation run that quietly used the weaker model must stay identifiable."""
    provider = FallbackProvider(Unreachable())
    response = provider.complete("sys", "user", context=_context())

    assert "fell back from openai_compatible" in response.provider
    assert "Connection refused" in response.model


def test_an_empty_completion_also_triggers_the_fallback():
    provider = FallbackProvider(Empty())
    response = provider.complete("sys", "user", context=_context())
    assert provider.fell_back and response.text.strip()


def test_a_working_model_is_used_untouched():
    primary = ScriptedProvider("ANSWER\nOverheating begins above 80 °C. [1]")
    provider = FallbackProvider(primary)
    response = provider.complete("sys", "user", context=_context())

    assert not provider.fell_back
    assert response.provider == "scripted"


def test_the_fallback_survives_a_full_ask(built):
    """End to end: the knowledge area names Ollama, no Ollama is running."""
    manifest, store = built
    manifest.specialist.llm_provider = "openai_compatible"
    manifest.specialist.llm_model = "llama3.1:8b"

    answer = Specialist(manifest, store).ask(
        "At what core temperature does overheating begin?"
    )
    assert answer.status == "answered"
    assert "fell back" in answer.llm["provider"]
    # And it is durable in the stored record, not just the console.
    assert "fell back" in json.dumps(answer.as_dict())


def test_a_keyless_provider_never_wears_a_model_name_it_did_not_run():
    """The pair that produced a wrong provenance line on a real answer.

    A runner overrode the provider because it had no hosted key, while the
    model variable still named the hosted model. The keyless provider took the
    name and the published reply credited a model that never ran -- the same
    class of fabrication the mock-Ollama incident produced, arriving by a
    different route.
    """
    from foundry.llm import get_provider

    provider = get_provider("local_extractive", "meta-llama/llama-3.1-70b-instruct")
    assert provider.name == "local_extractive"
    assert "llama" not in provider.model

    scripted = get_provider("scripted", "claude-opus-5")
    assert scripted.model == "scripted"


def test_a_real_provider_still_takes_the_model_it_is_given():
    from foundry.llm import get_provider

    assert get_provider("openai_compatible", "some/model").model == "some/model"
