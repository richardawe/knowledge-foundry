"""Hybrid retrieval: keyword, semantic, structured and graph, fused (§8)."""

from .base import Candidate, Evidence, Retriever
from .graph import GraphRetriever
from .hybrid import HybridRetriever, RetrievalResult, reciprocal_rank_fusion
from .index import IndexReport, build_semantic_index, load_embedder
from .keyword import KeywordRetriever, build_match_query
from .semantic import SemanticRetriever
from .structured import Filters, StructuredRetriever, infer_filters

__all__ = [
    "Candidate",
    "Evidence",
    "Filters",
    "GraphRetriever",
    "HybridRetriever",
    "IndexReport",
    "KeywordRetriever",
    "RetrievalResult",
    "Retriever",
    "SemanticRetriever",
    "StructuredRetriever",
    "build_match_query",
    "build_semantic_index",
    "infer_filters",
    "load_embedder",
    "reciprocal_rank_fusion",
]
