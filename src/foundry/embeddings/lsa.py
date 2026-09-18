"""Local semantic embedder: TF-IDF + truncated SVD (latent semantic analysis).

This is the default provider, and the choice deserves justification.

A 400 MB transformer download is a poor *default* for a system that must run
identically in CI, in a container and on a laptop, and that is supposed to cost
nothing per knowledge area. LSA over the knowledge area's own corpus:

* costs nothing and needs no network;
* is **deterministic** given a fixed seed, which matters enormously when the
  regression gate compares today's retrieval metrics against last week's;
* captures term co-occurrence, so "cells getting hot and catching fire"
  retrieves thermal-runaway passages that pure BM25 would miss;
* is roughly 150 lines behind an interface.

It is genuinely weaker than a good sentence transformer on paraphrase. The
correct response to that is not to assume -- it is to run the evaluation suite
with ``retrieval.semantic.provider: sentence_transformers`` and see whether the
metrics move. That experiment is exactly what §3 of the brief asks for, and this
architecture makes it a one-line manifest change.
"""

from __future__ import annotations

import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Sequence

from ..text import tokenize
from .base import l2_normalise

MAGIC = b"KFLSA001"


class NumpyRequired(RuntimeError):
    """Raised when the LSA fit needs numpy and it is not installed."""


class LSAEmbedder:
    """TF-IDF with a truncated-SVD projection learned from the corpus."""

    name = "lsa"

    def __init__(
        self,
        dim: int = 192,
        max_vocab: int = 30000,
        min_df: int = 2,
        max_df_ratio: float = 0.6,
        seed: int = 20260918,
    ) -> None:
        self.dim = dim
        self.max_vocab = max_vocab
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.seed = seed
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        # projection[j] is the j-th concept vector over the vocabulary.
        self.projection: list[list[float]] = []
        self._fitted = False

    # -- fitting ---------------------------------------------------------

    def _build_vocab(self, docs: Sequence[list[str]]) -> None:
        df = Counter()
        for tokens in docs:
            df.update(set(tokens))
        n_docs = max(1, len(docs))
        max_df = max(self.min_df, int(self.max_df_ratio * n_docs))
        candidates = [
            (term, count)
            for term, count in df.items()
            if count >= self.min_df and count <= max_df
        ]
        # Very small corpora would otherwise produce an empty vocabulary.
        if not candidates:
            candidates = list(df.items())
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        candidates = candidates[: self.max_vocab]
        # Index by sorted term so the vocabulary -- and therefore the model
        # file -- is byte-identical across runs on the same corpus.
        self.vocab = {term: i for i, term in enumerate(sorted(t for t, _ in candidates))}
        self.idf = [0.0] * len(self.vocab)
        for term, index in self.vocab.items():
            self.idf[index] = math.log((1 + n_docs) / (1 + df[term])) + 1.0

    def _tfidf(self, tokens: Sequence[str]) -> dict[int, float]:
        """Sublinear-tf, idf-weighted, L2-normalised sparse vector."""
        counts = Counter(t for t in tokens if t in self.vocab)
        if not counts:
            return {}
        vector: dict[int, float] = {}
        for term, count in counts.items():
            index = self.vocab[term]
            vector[index] = (1.0 + math.log(count)) * self.idf[index]
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm:
            for index in vector:
                vector[index] /= norm
        return vector

    def fit(self, texts: Sequence[str]) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise NumpyRequired(
                "the 'lsa' embedder needs numpy for the SVD step: "
                "pip install 'knowledge-foundry[fast]', or set "
                "retrieval.semantic.provider to 'tfidf' in the manifest"
            ) from exc

        docs = [tokenize(t) for t in texts]
        docs = [d for d in docs if d]
        if not docs:
            self.vocab, self.idf, self.projection, self._fitted = {}, [], [], True
            return

        self._build_vocab(docs)
        vocab_size = len(self.vocab)
        sparse = [self._tfidf(d) for d in docs]
        n_docs = len(sparse)
        # The rank cannot exceed either dimension of the term-document matrix.
        k = max(1, min(self.dim, vocab_size, n_docs))

        rng = np.random.default_rng(self.seed)

        rows, cols, vals = [], [], []
        for i, vector in enumerate(sparse):
            for j, value in vector.items():
                rows.append(i)
                cols.append(j)
                vals.append(value)
        rows_a = np.asarray(rows, dtype=np.int64)
        cols_a = np.asarray(cols, dtype=np.int64)
        vals_a = np.asarray(vals, dtype=np.float64)

        def a_matmul(matrix):
            """A @ matrix, where A is the (n_docs x vocab) sparse tf-idf matrix."""
            out = np.zeros((n_docs, matrix.shape[1]), dtype=np.float64)
            np.add.at(out, rows_a, matrix[cols_a] * vals_a[:, None])
            return out

        def at_matmul(matrix):
            """A.T @ matrix."""
            out = np.zeros((vocab_size, matrix.shape[1]), dtype=np.float64)
            np.add.at(out, cols_a, matrix[rows_a] * vals_a[:, None])
            return out

        # Randomised range finder with two power iterations: enough to separate
        # the leading singular directions on text, and cheap.
        oversample = min(10, max(0, vocab_size - k))
        omega = rng.standard_normal((vocab_size, k + oversample))
        y = a_matmul(omega)
        for _ in range(2):
            y = a_matmul(at_matmul(y))
        q, _ = np.linalg.qr(y)
        b = at_matmul(q).T  # (k + oversample) x vocab
        _, _, vt = np.linalg.svd(b, full_matrices=False)
        projection = vt[:k]

        self.dim = int(projection.shape[0])
        # Quantise to float32 here rather than only on save, so a freshly fitted
        # embedder and one loaded from disk produce bit-identical vectors. The
        # version hash depends on that.
        self.projection = [
            [float(x) for x in np.asarray(row, dtype=np.float32)] for row in projection
        ]
        self._fitted = True

    # -- encoding --------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("LSAEmbedder.encode called before fit/load")
        out: list[list[float]] = []
        for text in texts:
            sparse = self._tfidf(tokenize(text))
            vector = [0.0] * self.dim
            if sparse:
                for concept in range(self.dim):
                    row = self.projection[concept]
                    vector[concept] = sum(value * row[index] for index, value in sparse.items())
            out.append(l2_normalise(vector))
        return out

    # -- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        """JSON header plus a float32 blob: small, portable, no pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "name": self.name,
            "dim": self.dim,
            "seed": self.seed,
            "vocab": list(self.vocab.keys()),  # ordered by index
            "idf": self.idf,
        }
        header_bytes = json.dumps(header).encode("utf-8")
        with path.open("wb") as fh:
            fh.write(MAGIC)
            fh.write(struct.pack("<I", len(header_bytes)))
            fh.write(header_bytes)
            for row in self.projection:
                fh.write(struct.pack(f"<{len(row)}f", *row))

    def load(self, path: Path) -> None:
        path = Path(path)
        with path.open("rb") as fh:
            if fh.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{path} is not an LSA model file")
            (header_len,) = struct.unpack("<I", fh.read(4))
            header = json.loads(fh.read(header_len).decode("utf-8"))
            self.dim = int(header["dim"])
            self.seed = int(header.get("seed", self.seed))
            self.vocab = {term: i for i, term in enumerate(header["vocab"])}
            self.idf = [float(x) for x in header["idf"]]
            vocab_size = len(self.vocab)
            self.projection = []
            for _ in range(self.dim):
                blob = fh.read(4 * vocab_size)
                self.projection.append(list(struct.unpack(f"<{vocab_size}f", blob)))
        self._fitted = True


class TFIDFEmbedder(LSAEmbedder):
    """Pure-Python fallback: no SVD, no numpy.

    Semantically weaker than LSA -- it is essentially cosine over sparse tf-idf,
    so it adds little beyond BM25. It exists so the system still runs end to end
    with zero optional dependencies, and so that a failure to install numpy
    degrades retrieval quality rather than breaking the build.
    """

    name = "tfidf"

    def fit(self, texts: Sequence[str]) -> None:
        docs = [tokenize(t) for t in texts]
        docs = [d for d in docs if d]
        if not docs:
            self.vocab, self.idf, self.projection, self._fitted = {}, [], [], True
            return
        self._build_vocab(docs)
        vocab_size = len(self.vocab)
        # Identity projection restricted to the vocabulary: encode() then simply
        # returns the (dense) tf-idf vector.
        self.dim = vocab_size
        self.projection = []
        self._fitted = True

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TFIDFEmbedder.encode called before fit/load")
        out = []
        for text in texts:
            sparse = self._tfidf(tokenize(text))
            vector = [0.0] * self.dim
            for index, value in sparse.items():
                vector[index] = value
            out.append(vector)
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"name": self.name, "vocab": list(self.vocab.keys()), "idf": self.idf}),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab = {term: i for i, term in enumerate(data["vocab"])}
        self.idf = [float(x) for x in data["idf"]]
        self.dim = len(self.vocab)
        self.projection = []
        self._fitted = True
