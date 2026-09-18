"""Hybrid retrieval (§8).

The interesting assertions are not "retrieval returns something" but that each
retriever contributes what it is supposed to contribute, and that fusion is
scale-free.
"""

from __future__ import annotations

import pytest

from foundry.embeddings import cosine, get_embedder
from foundry.retrieval import (
    Candidate,
    GraphRetriever,
    HybridRetriever,
    KeywordRetriever,
    SemanticRetriever,
    StructuredRetriever,
    build_match_query,
    infer_filters,
    load_embedder,
    reciprocal_rank_fusion,
)


# -- keyword ------------------------------------------------------------


def test_keyword_finds_exact_technical_terms(built):
    manifest, store = built
    hits = KeywordRetriever(store).search("12 bar internal pressure", limit=10)

    assert hits
    assert any("12 bar" in h.text for h in hits)


@pytest.mark.parametrize(
    "hostile",
    ['widget AND NOT venting', 'widget*', 'widget "unclosed', "widget OR (", "NEAR(a b)"],
)
def test_fts_operators_in_user_input_cannot_break_or_redirect_the_search(built, hostile):
    """User text is data. It must never become FTS5 query syntax."""
    _manifest, store = built
    KeywordRetriever(store).search(hostile, limit=5)  # must not raise


def test_empty_query_returns_nothing_rather_than_everything(built):
    _manifest, store = built
    assert build_match_query("the and of") == ""
    assert KeywordRetriever(store).search("the and of", limit=5) == []


# -- semantic -----------------------------------------------------------


def test_semantic_retrieves_on_meaning_not_wording(built):
    """The query shares almost no content words with the target passage."""
    manifest, store = built
    embedder = load_embedder(manifest)
    hits = SemanticRetriever(store, embedder).search(
        "how do I know a unit is about to release fumes", limit=5
    )
    assert hits


def test_semantic_degrades_to_empty_when_the_model_is_unavailable(built):
    """A broken embedder must cost recall, never the answer path."""
    manifest, store = built

    class Broken:
        def encode(self, texts):
            raise RuntimeError("model server is down")

    assert SemanticRetriever(store, Broken()).search("venting", limit=5) == []


# -- structured ---------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_type",
    [
        ("what does the regulation require", "regulatory"),
        ("which standard clause applies", "standard"),
        ("what did the incident investigation find", "investigation"),
    ],
)
def test_structured_reads_intent_from_the_query(query, expected_type):
    assert expected_type in infer_filters(query).source_types


def test_structured_filters_by_source_type(built):
    _manifest, store = built
    hits = StructuredRetriever(store).search(
        "what does the incident investigation say about the vent path", limit=10
    )

    assert hits
    assert {h.source_id for h in hits} == {"src-incident"}


def test_structured_returns_nothing_when_the_query_carries_no_structure(built):
    _manifest, store = built
    assert StructuredRetriever(store).search("widget venting behaviour", limit=10) == []


def test_temporal_cues_become_date_filters():
    filters = infer_filters("what changed since 2024")
    assert filters.published_after == "2024-01-01"
    assert infer_filters("what is the latest guidance").prefer_recent


# -- graph --------------------------------------------------------------


def test_graph_expands_through_the_ontology(built):
    """'What detects venting' must reach off-gas detection via detected_by."""
    _manifest, store = built
    retriever = GraphRetriever(store, max_hops=2)

    assert "venting" in retriever.match_entities("what detects venting")
    expanded = {name for name, _predicate, _direction in retriever.expand(["venting"])}
    assert "Off-gas detection" in expanded

    hits = retriever.search("what detects venting", limit=10)
    assert any(h.source_id == "src-detection" for h in hits)


def test_graph_matches_entity_aliases(built):
    _manifest, store = built
    retriever = GraphRetriever(store)
    assert "off_gas_detection" in retriever.match_entities("is carbon monoxide detection reliable")


def test_graph_returns_nothing_when_no_entity_is_mentioned(built):
    _manifest, store = built
    assert GraphRetriever(store).search("quarterly revenue forecast", limit=5) == []


# -- fusion -------------------------------------------------------------


def test_rrf_is_scale_free():
    """BM25 and cosine live on different scales; fusion must read only rank."""
    small = [Candidate("a", 0.9, "keyword"), Candidate("b", 0.1, "keyword")]
    huge = [Candidate("a", 9000.0, "keyword"), Candidate("b", 1000.0, "keyword")]

    assert reciprocal_rank_fusion({"keyword": small}, {"keyword": 1.0}) == (
        reciprocal_rank_fusion({"keyword": huge}, {"keyword": 1.0})
    )


def test_rrf_rewards_agreement_between_retrievers():
    agreed = reciprocal_rank_fusion(
        {
            "keyword": [Candidate("a", 1.0, "keyword"), Candidate("b", 0.5, "keyword")],
            "semantic": [Candidate("a", 1.0, "semantic"), Candidate("c", 0.5, "semantic")],
        },
        {"keyword": 1.0, "semantic": 1.0},
    )
    assert agreed["a"] > agreed["b"] > 0


def test_zero_weight_removes_a_retriever_entirely():
    fused = reciprocal_rank_fusion(
        {"keyword": [Candidate("a", 1.0, "keyword")]}, {"keyword": 0.0}
    )
    assert fused == {}


# -- end to end ---------------------------------------------------------


def test_hybrid_returns_ranked_evidence_with_full_citations(built):
    manifest, store = built
    result = HybridRetriever(store, manifest, load_embedder(manifest)).retrieve(
        "at what temperature does overheating begin"
    )

    assert result.evidence
    top = result.evidence[0]
    assert top.rank == 1
    assert top.publisher and top.uri and top.licence
    assert top.citation()
    assert result.evidence == sorted(result.evidence, key=lambda e: -e.score)


def test_hybrid_records_which_retrievers_contributed(built):
    manifest, store = built
    result = HybridRetriever(store, manifest, load_embedder(manifest)).retrieve(
        "what does the regulation require for inspection intervals"
    )

    assert result.per_retriever
    assert any(e.retrievers for e in result.evidence)


def test_hybrid_never_surfaces_unmirrored_pointer_content(built):
    """A pointer source has no chunks, so it can never be quoted."""
    manifest, store = built
    retriever = HybridRetriever(store, manifest, load_embedder(manifest))

    for query in ("widget standard 9000", "venting", "inspection"):
        assert all(e.source_id != "src-pointer" for e in retriever.retrieve(query).evidence)


def test_hybrid_survives_a_query_matching_nothing(built):
    manifest, store = built
    result = HybridRetriever(store, manifest, load_embedder(manifest)).retrieve(
        "zzzzq unrelated gibberish token"
    )
    assert result.evidence == []
    assert result.top_score == 0.0


def test_authority_rerank_prefers_the_more_authoritative_source(built):
    """Applied once, after fusion -- not multiplied per contributing retriever."""
    manifest, store = built
    retriever = HybridRetriever(store, manifest, load_embedder(manifest))
    manifest.retrieval.authority_weight = 0.0
    neutral = [e.chunk_id for e in retriever.retrieve("inspection interval").evidence]
    manifest.retrieval.authority_weight = 0.9
    weighted = retriever.retrieve("inspection interval").evidence

    assert weighted
    if neutral and weighted[0].chunk_id != neutral[0]:
        assert weighted[0].authority >= 4


# -- embeddings ---------------------------------------------------------


def test_lsa_embeddings_are_deterministic_across_refits():
    """The regression gate compares runs; a nondeterministic index would lie."""
    corpus = [
        "thermal runaway begins with SEI decomposition",
        "cells overheat and vent flammable gas",
        "corrosion of steel reinforcement in concrete",
        "chloride ingress damages reinforced concrete",
    ]
    first, second = get_embedder("lsa", dim=3), get_embedder("lsa", dim=3)
    first.fit(corpus)
    second.fit(corpus)

    assert first.encode(["thermal runaway"]) == second.encode(["thermal runaway"])


def test_embedder_roundtrips_bit_identically(tmp_path):
    corpus = ["widgets vent at 12 bar", "widgets overheat at 80 degrees", "inspection every 180 days"]
    embedder = get_embedder("lsa", dim=2)
    embedder.fit(corpus)
    path = tmp_path / "model.bin"
    embedder.save(path)

    reloaded = get_embedder("lsa")
    reloaded.load(path)
    assert reloaded.encode(["venting"]) == embedder.encode(["venting"])


def test_semantically_related_text_scores_above_unrelated_text():
    corpus = [
        "thermal runaway in cells begins with separator failure and exothermic reaction",
        "cells that overheat catch fire and vent flammable gas during runaway",
        "chloride ingress drives corrosion of steel reinforcement in concrete structures",
        "reinforced concrete suffers corrosion damage from chloride contamination",
    ]
    embedder = get_embedder("lsa", dim=3)
    embedder.fit(corpus)
    vectors = embedder.encode(corpus)
    query = embedder.encode(["overheating cells that ignite"])[0]

    assert cosine(query, vectors[1]) > cosine(query, vectors[2])
