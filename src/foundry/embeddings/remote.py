"""Optional embedders: local transformers and any OpenAI-compatible endpoint.

Neither is required. Both exist so that a knowledge area whose evaluation shows
LSA is the bottleneck can switch provider with one manifest line, and so that
the project never depends on a single vendor: the OpenAI-compatible client
speaks to Ollama, vLLM, LM Studio, Together, OpenRouter and OpenAI alike.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Sequence

from .base import l2_normalise


class SentenceTransformerEmbedder:
    """Local transformer embeddings via sentence-transformers."""

    name = "sentence_transformers"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", dim: int = 384) -> None:
        self.model_name = model
        self.dim = dim
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "provider 'sentence_transformers' needs: "
                    "pip install 'knowledge-foundry[sentence-transformers]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def fit(self, texts: Sequence[str]) -> None:
        self._ensure()  # pretrained: nothing to learn from the corpus

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in v] for v in vectors]

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps({"name": self.name, "model": self.model_name, "dim": self.dim}),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.model_name = data["model"]
        self.dim = int(data["dim"])


class OpenAICompatibleEmbedder:
    """Embeddings from any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Configured entirely by environment so the manifest never carries a secret:
    ``FOUNDRY_EMBED_BASE_URL`` (default: a local Ollama) and
    ``FOUNDRY_EMBED_API_KEY``.
    """

    name = "openai_compatible"

    def __init__(self, model: str = "nomic-embed-text", dim: int = 768) -> None:
        self.model_name = model
        self.dim = dim
        self.base_url = os.environ.get("FOUNDRY_EMBED_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.environ.get("FOUNDRY_EMBED_API_KEY", "not-needed")

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        # Batched to keep request bodies reasonable on hosted endpoints.
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            payload = json.dumps({"model": self.model_name, "input": batch}).encode()
            request = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/embeddings",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            for item in data["data"]:
                vector = [float(x) for x in item["embedding"]]
                self.dim = len(vector)
                out.append(l2_normalise(vector))
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps({"name": self.name, "model": self.model_name, "dim": self.dim}),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.model_name = data["model"]
        self.dim = int(data["dim"])
