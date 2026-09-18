"""Embedder interface and vector serialisation.

Every embedder is interchangeable: fit on a corpus (a no-op for pretrained
models), encode texts, save and load. The knowledge area names its provider in
``manifest.yaml`` and nothing else in the system changes -- which is the whole
point of the no-lock-in constraint.
"""

from __future__ import annotations

import math
from array import array
from pathlib import Path
from typing import Protocol, Sequence


class Embedder(Protocol):
    """Minimal contract every embedding provider satisfies."""

    name: str
    dim: int

    def fit(self, texts: Sequence[str]) -> None: ...
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...


def pack(vector: Sequence[float]) -> bytes:
    """Serialise a vector as little-endian float32 for the embeddings table."""
    arr = array("f", vector)
    if arr.itemsize != 4:  # pragma: no cover - platform guard
        raise RuntimeError("float32 array expected")
    import sys

    if sys.byteorder == "big":
        arr.byteswap()
    return arr.tobytes()


def unpack(blob: bytes) -> list[float]:
    import sys

    arr = array("f")
    arr.frombytes(blob)
    if sys.byteorder == "big":
        arr.byteswap()
    return list(arr)


def l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Vectors are stored normalised, so this is a dot product."""
    if len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)
