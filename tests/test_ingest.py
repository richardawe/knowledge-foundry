"""Ingestion: provenance, incrementality and licence handling (§7).

The three properties under test are the ones the rest of the system depends on:
every chunk traces to a source, unchanged content is not reprocessed, and
non-redistributable sources are registered without being mirrored.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from foundry.ingest import ingest_knowledge_area, load_source_register
from foundry.ingest.claims import classify, extract_claims
from foundry.ingest.extractors import extract
from foundry.manifest import ManifestError
from foundry.storage import Store


def test_ingest_populates_sources_documents_and_chunks(manifest):
    store = Store(manifest.id)
    report = ingest_knowledge_area(manifest, store)

    assert report.indexed == 3, report.failures
    assert report.failures == []
    counts = store.counts()
    assert counts["sources"] == 4  # three mirrored plus one pointer
    assert counts["documents"] == 3
    assert counts["chunks"] > 0
    assert counts["claims"] > 0
    store.close()


def test_every_chunk_traces_to_a_source_with_full_provenance(built):
    """'Where did this answer come from?' must be a join, not a convention."""
    _manifest, store = built
    chunk_ids = [row["id"] for row in store.chunks()]
    resolved = store.chunks_with_provenance(chunk_ids)

    assert set(resolved) == set(chunk_ids)
    for row in resolved.values():
        assert row["uri"]
        assert row["publisher"]
        assert row["licence"]
        assert row["retrieved_at"]
        assert row["authority"] in range(1, 6)


def test_pointer_sources_are_registered_but_never_mirrored(built):
    """Licence compliance is structural: no content row can exist for a pointer."""
    _manifest, store = built
    pointer = store.get_source("src-pointer")

    assert pointer["status"] == "pointer"
    assert pointer["mirrored"] == 0
    assert pointer["content_sha256"] is None
    assert store.conn.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE source_id = 'src-pointer'"
    ).fetchone()["n"] == 0
    # It still appears in the register, because the evidence page must disclose it.
    assert pointer["title"] == "Widget Standard 9000 (full text)"


def test_reingesting_unchanged_content_is_a_no_op(manifest):
    """Success criterion #7: update without rebuilding the whole system."""
    store = Store(manifest.id)
    ingest_knowledge_area(manifest, store)
    before = {row["id"]: row["content_sha256"] for row in store.sources()}
    chunk_ids_before = {row["id"] for row in store.chunks()}

    second = ingest_knowledge_area(manifest, store, force=True)

    assert second.unchanged == 3
    assert second.documents == 0
    assert {row["id"]: row["content_sha256"] for row in store.sources()} == before
    assert {row["id"] for row in store.chunks()} == chunk_ids_before
    store.close()


def test_changed_source_is_reindexed_and_leaves_no_orphans(manifest, tmp_path):
    store = Store(manifest.id)
    ingest_knowledge_area(manifest, store)
    original_chunks = {row["id"] for row in store.chunks()}

    register = yaml.safe_load(manifest.sources_path.read_text())
    target = Path(register["sources"][0]["uri"])
    target.write_text(
        "# Widget Overheating\n\n"
        "Revised guidance: widget overheating begins at 75 °C, not 80 °C. "
        "Operators shall isolate the widget before the core reaches 90 °C. "
        "This revision supersedes all previous editions of the handbook.\n",
        encoding="utf-8",
    )

    report = ingest_knowledge_area(manifest, store, force=True)

    assert report.documents == 1
    current = {row["id"] for row in store.chunks()}
    assert current != original_chunks
    # No chunk may survive whose document is gone.
    orphans = store.conn.execute(
        "SELECT COUNT(*) AS n FROM chunks c "
        "LEFT JOIN documents d ON d.id = c.document_id WHERE d.id IS NULL"
    ).fetchone()["n"]
    assert orphans == 0
    store.close()


def test_fts_index_is_updated_when_a_document_is_replaced(manifest):
    """A stale FTS entry would let retrieval cite text that no longer exists."""
    store = Store(manifest.id)
    ingest_knowledge_area(manifest, store)

    register = yaml.safe_load(manifest.sources_path.read_text())
    Path(register["sources"][2]["uri"]).write_text(
        "# Northgate Depot\n\nThe depot was decommissioned in 2026 and the widget "
        "was removed from service without further incident. Records were archived "
        "with the regulator and no obstruction was found in the replacement unit.\n",
        encoding="utf-8",
    )
    ingest_knowledge_area(manifest, store, force=True)

    hits = store.conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH ?", ('"ignited"',)
    ).fetchone()["n"]
    assert hits == 0
    store.close()


def test_failed_fetch_is_recorded_not_swallowed(manifest, tmp_path):
    register = yaml.safe_load(manifest.sources_path.read_text())
    register["sources"].append(
        {
            "id": "src-missing",
            "uri": str(tmp_path / "does-not-exist.md"),
            "title": "Missing Document",
            "publisher": "Nobody",
            "source_type": "reference",
            "authority": 1,
            "licence": "test",
        }
    )
    manifest.sources_path.write_text(yaml.safe_dump(register), encoding="utf-8")

    store = Store(manifest.id)
    report = ingest_knowledge_area(manifest, store)

    assert any(f["source"] == "src-missing" for f in report.failures)
    assert store.get_source("src-missing")["status"] == "failed"
    assert store.get_source("src-missing")["error"]
    store.close()


def test_ontology_is_loaded_into_entities_and_triples(built):
    _manifest, store = built
    names = {row["name"] for row in store.entities()}
    assert {"Widget", "Overheating", "Venting"} <= names

    triples = {(t["subject_name"], t["predicate"], t["object_name"]) for t in store.triples()}
    assert ("Venting", "detected_by", "Off-gas detection") in triples


def test_ontology_rejects_dangling_relationships(manifest):
    ontology = yaml.safe_load(manifest.resolve("ontology.yaml").read_text())
    ontology["relationships"].append(
        {"subject": "widget", "predicate": "related_to", "object": "not_declared"}
    )
    manifest.resolve("ontology.yaml").write_text(yaml.safe_dump(ontology), encoding="utf-8")

    store = Store(manifest.id)
    with pytest.raises(ManifestError, match="undeclared entity"):
        ingest_knowledge_area(manifest, store)
    store.close()


@pytest.mark.parametrize(
    "field", ["title", "publisher", "source_type", "authority", "licence"]
)
def test_source_register_requires_provenance_fields(manifest, field):
    register = yaml.safe_load(manifest.sources_path.read_text())
    register["sources"][0].pop(field, None)
    register["defaults"].pop(field, None)
    manifest.sources_path.write_text(yaml.safe_dump(register), encoding="utf-8")

    with pytest.raises(ManifestError, match=field):
        load_source_register(manifest)


def test_source_register_rejects_duplicate_ids(manifest):
    register = yaml.safe_load(manifest.sources_path.read_text())
    register["sources"].append(dict(register["sources"][0]))
    manifest.sources_path.write_text(yaml.safe_dump(register), encoding="utf-8")

    with pytest.raises(ManifestError, match="duplicate source id"):
        load_source_register(manifest)


def test_html_extraction_drops_scripts_and_keeps_headings():
    html = (
        b"<html><head><title>Doc</title></head><body><nav>Home</nav>"
        b"<h2>4.1 Venting</h2><p>Widgets vent at 12 bar.</p>"
        b"<script>alert(1)</script><style>p{}</style></body></html>"
    )
    result = extract(html, "text/html")

    assert result.title == "Doc"
    assert "## 4.1 Venting" in result.text
    assert "alert(1)" not in result.text


def test_content_type_is_sniffed_when_the_header_lies():
    html = b"<!DOCTYPE html><html><body><p>Widgets vent at 12 bar.</p></body></html>"
    assert extract(html, "application/octet-stream").extractor == "html"


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Widgets vent when pressure exceeds 12 bar.", "fact"),
        ("Operators shall isolate the widget before 95 °C.", "requirement"),
        ("Onset occurs between 130 and 180 °C in these tests.", "range"),
        ("See also the appendix for more detail on this topic.", None),
    ],
)
def test_claim_classification(sentence, expected):
    assert classify(sentence) == expected


def test_claims_are_verbatim_not_paraphrased():
    """Grounding checks against claims would be circular if claims were rewritten."""
    text = "Widgets vent when internal pressure exceeds 12 bar in normal operation."
    claims = extract_claims("chunk-1", text)

    assert claims
    assert claims[0]["text"] in text
