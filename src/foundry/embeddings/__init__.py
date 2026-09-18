"""Embedding provider registry.

Adding a provider means adding one entry here. Nothing else in the system knows
which embedder is in use -- that is what keeps the semantic-retrieval decision
an experiment rather than an architecture commitment (§3 of the brief).
"""

from __future__ import annotations

import warnings
from pathlib import Path

from ..manifest import Manifest
from .base import Embedder, cosine, l2_normalise, pack, unpack
from .lsa import LSAEmbedder, NumpyRequired, TFIDFEmbedder

MODEL_FILENAME = "embedder.model"


def _make(provider: str, model: str | None, dim: int):
    if provider == "lsa":
        return LSAEmbedder(dim=dim)
    if provider == "tfidf":
        return TFIDFEmbedder(dim=dim)
    if provider == "sentence_transformers":
        from .remote import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model=model or "sentence-transformers/all-MiniLM-L6-v2"
        )
    if provider == "openai_compatible":
        from .remote import OpenAICompatibleEmbedder

        return OpenAICompatibleEmbedder(model=model or "nomic-embed-text")
    raise ValueError(
        f"unknown embedding provider {provider!r}; "
        "expected lsa, tfidf, sentence_transformers or openai_compatible"
    )


def get_embedder(provider: str = "lsa", model: str | None = None, dim: int = 192):
    """Build an embedder, degrading gracefully when numpy is unavailable.

    The fallback is loud and is recorded in the ``embeddings.model`` column, so
    a build that silently used the weaker embedder is still traceable after the
    fact -- which matters when comparing evaluation runs.
    """
    if provider == "lsa":
        try:
            import numpy  # noqa: F401
        except ImportError:
            warnings.warn(
                "numpy is not installed; falling back from the 'lsa' embedder to "
                "'tfidf'. Semantic retrieval will be weaker. "
                "Install with: pip install 'knowledge-foundry[fast]'",
                RuntimeWarning,
                stacklevel=2,
            )
            return TFIDFEmbedder(dim=dim)
    return _make(provider, model, dim)


def embedder_for(manifest: Manifest):
    cfg = manifest.retrieval
    return get_embedder(cfg.semantic_provider, cfg.semantic_model, cfg.semantic_dim)


def model_path(manifest: Manifest) -> Path:
    from .. import paths

    return paths.ka_var_dir(manifest.id) / MODEL_FILENAME


__all__ = [
    "Embedder",
    "LSAEmbedder",
    "TFIDFEmbedder",
    "NumpyRequired",
    "cosine",
    "embedder_for",
    "get_embedder",
    "l2_normalise",
    "model_path",
    "pack",
    "unpack",
]
