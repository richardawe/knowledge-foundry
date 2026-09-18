"""Native Ollama provider.

Ollama exposes two HTTP surfaces: its own API at ``/api/chat`` and
``/api/generate``, and an OpenAI-compatible shim at ``/v1``. This provider uses
the native one, because that is what actually works everywhere:

* the shim only exists from Ollama 0.1.24 onward, so on an older daemon the
  OpenAI path 404s while the native path is fine;
* the native API is what the ``ollama`` Python package calls, so a machine
  already running Ollama workloads is proven against exactly this surface;
* ``OLLAMA_HOST`` is the variable that package honours, so a daemon on a
  non-default host or port is found the same way here as there.

The native API streams by default, which is why every request sets
``stream: false`` -- a streamed body would otherwise arrive as newline-delimited
JSON fragments and parse as garbage.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import GenerationContext, LLMResponse

DEFAULT_HOST = "http://localhost:11434"
# A model answering a retrieval-augmented prompt on a machine that is also
# running other Ollama jobs can take a while. Generous by default, because a
# timeout here looks exactly like "the model is broken".
DEFAULT_TIMEOUT = 300


def host() -> str:
    """Resolve the daemon address the same way the ``ollama`` package does."""
    value = (
        os.environ.get("FOUNDRY_OLLAMA_HOST")
        or os.environ.get("OLLAMA_HOST")
        or DEFAULT_HOST
    ).strip().rstrip("/")
    # OLLAMA_HOST is commonly set bare: "localhost:11434", "0.0.0.0", or a
    # port on its own. All three are valid for the ollama CLI, so all three
    # have to resolve here too.
    if not value.startswith(("http://", "https://")):
        if value.isdigit():
            value = f"http://localhost:{value}"
        elif ":" in value:
            value = f"http://{value}"
        else:
            value = f"http://{value}:11434"
    return value


def _post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def list_models(timeout: int = 10) -> list[str]:
    """Model names the daemon has pulled. Raises if it is not reachable."""
    with urllib.request.urlopen(f"{host()}/api/tags", timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return [m.get("name", "") for m in data.get("models", [])]


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str = "llama3.1:8b", timeout: int | None = None) -> None:
        self.model = model
        self.timeout = timeout or int(os.environ.get("FOUNDRY_LLM_TIMEOUT", DEFAULT_TIMEOUT))
        self.base_url = host()
        # Set once /api/chat is known to be unavailable, so an older daemon is
        # probed for it only on the first call rather than on every answer.
        self._use_generate = False

    def _chat(self, system: str, user: str, temperature: float, max_tokens: int) -> dict:
        return _post(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            self.timeout,
        )

    def _generate(self, system: str, user: str, temperature: float, max_tokens: int) -> dict:
        return _post(
            f"{self.base_url}/api/generate",
            {
                "model": self.model,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            self.timeout,
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        context: GenerationContext | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        try:
            if self._use_generate:
                data = self._generate(system, user, temperature, max_tokens)
                text = data.get("response", "")
            else:
                try:
                    data = self._chat(system, user, temperature, max_tokens)
                    text = (data.get("message") or {}).get("content", "")
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                    # Pre-0.3 daemons have /api/generate but not /api/chat.
                    self._use_generate = True
                    data = self._generate(system, user, temperature, max_tokens)
                    text = data.get("response", "")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error", "")
            except Exception:
                pass
            if exc.code == 404 and "model" in detail.lower():
                detail = f"{detail} — pull it with: ollama pull {self.model}"
            return LLMResponse(
                text="", provider=self.name, model=self.model,
                error=f"HTTP {exc.code} from {self.base_url}: {detail or exc.reason}",
            )
        except Exception as exc:
            return LLMResponse(
                text="", provider=self.name, model=self.model,
                error=f"{type(exc).__name__} talking to {self.base_url}: {exc}",
            )

        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            stop_reason=str(data.get("done_reason", "") or ""),
        )
