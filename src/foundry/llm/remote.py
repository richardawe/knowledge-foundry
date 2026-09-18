"""Hosted providers: Anthropic and any OpenAI-compatible endpoint.

Both are optional. Between them they cover Claude, GPT, Llama via Ollama or
vLLM, Mistral, Together, OpenRouter and anything else that speaks the
OpenAI chat-completions shape -- which is the practical definition of "not
locked in".

Keys come from the environment, never from the manifest, so a knowledge area
stays publishable as plain text.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import GenerationContext, LLMResponse


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    def complete(self, system: str, user: str, *, context: GenerationContext | None = None,
                 temperature: float = 0.0, max_tokens: int = 1200) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="", provider=self.name, model=self.model,
                error="ANTHROPIC_API_KEY is not set",
            )
        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/v1/messages",
            data=payload,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return LLMResponse(text="", provider=self.name, model=self.model, error=str(exc))

        text = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
            stop_reason=data.get("stop_reason", ""),
        )


class OpenAICompatibleProvider:
    """Any ``/v1/chat/completions`` endpoint.

    Defaults to a local Ollama, so the "cheap production inference" path in §4
    needs no account anywhere.
    """

    name = "openai_compatible"

    def __init__(self, model: str = "llama3.1:8b") -> None:
        self.model = model
        self.base_url = os.environ.get("FOUNDRY_LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.environ.get("FOUNDRY_LLM_API_KEY", "not-needed")

    def complete(self, system: str, user: str, *, context: GenerationContext | None = None,
                 temperature: float = 0.0, max_tokens: int = 1200) -> LLMResponse:
        payload = json.dumps({
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return LLMResponse(text="", provider=self.name, model=self.model, error=str(exc))

        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage", {})
        return LLMResponse(
            text=choice.get("message", {}).get("content", ""),
            provider=self.name,
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            stop_reason=choice.get("finish_reason", ""),
        )
