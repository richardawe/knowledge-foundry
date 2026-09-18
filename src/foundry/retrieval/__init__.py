"""Hybrid retrieval: keyword, semantic, structured and graph, fused (§8)."""

from .base import Candidate, Evidence, Retriever
from .graph import GraphRetriever
from .hybrid import HybridRetriever, RetrievalResult, reciprocal_rank_fusion
from .index import IndexReport, build_semantic_index, load_embedder
from .keyword import KeywordRetriever, build_match_query
from .semantic import SemanticRetriever
from .structured import DEFAULT_TYPE_CUES, Filters, StructuredRetriever, infer_filters, merge_cues

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
    "DEFAULT_TYPE_CUES",
    "StructuredRetriever",
    "merge_cues",
    "build_match_query",
    "build_semantic_index",
    "infer_filters",
    "load_embedder",
    "reciprocal_rank_fusion",
]
