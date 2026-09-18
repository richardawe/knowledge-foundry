"""Shared fixtures.

Tests build a complete throwaway knowledge area on disk. That is deliberate: the
factory's contract is "a knowledge area is a directory", so the tests exercise
that contract rather than poking at internals with mocks.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

MANIFEST = {
    "knowledge_area": {
        "id": "kb-test-widgets",
        "name": "Test Widget Failure Intelligence",
        "description": "A synthetic knowledge area used only by the test suite.",
        "domain": "testing",
        "version": "v0.1.0",
        "owner": "test-suite",
        "created": "2026-01-01",
        "last_updated": "2026-01-01",
        "visibility": "public",
    },
    "scope": {
        "in_scope": ["widget overheating", "widget venting", "widget inspection"],
        "out_of_scope": ["tax law", "medical diagnosis"],
    },
    "sources": {"register": "sources.yaml"},
    "knowledge_model": {"ontology": "ontology.yaml"},
    "retrieval": {
        "top_k": 4,
        "candidate_k": 20,
        "chunk": {"target_chars": 400, "overlap_chars": 60, "min_chars": 80},
        "semantic": {"provider": "lsa", "dim": 8},
    },
    "specialist": {
        "prompt": "prompts/system.md",
        "llm": {"provider": "local_extractive"},
        "sufficiency": {"min_evidence_score": 0.001, "min_chunks": 1},
    },
    "evaluation": {"suites": ["evaluation/core.yaml"]},
    "governance": {
        "review_status": "draft",
        "confidence": "low",
        "known_limitations": ["This knowledge area is synthetic and exists for tests."],
        "disclaimer": "Test fixture. Not advice.",
    },
}

DOC_OVERHEAT = """# Widget Overheating

Widget overheating begins when the core temperature exceeds 80 °C. At that point the
seal begins to soften and internal pressure rises. Operators shall isolate the widget
before the core reaches 95 °C.

## 3.1 Venting behaviour

A widget vents when internal pressure exceeds 12 bar. Vented gas contains carbon
monoxide and widget vapour. Venting is preceded by an audible click in most recorded
cases. The vent path must remain unobstructed at all times.
"""

DOC_DETECTION = """# Detecting Widget Failure

Off-gas detection identifies carbon monoxide before the core temperature rises.
Detection systems shall alarm within 30 seconds of gas ingress. Thermocouples respond
more slowly than gas sensors in every documented test.

## 2.4 Inspection intervals

Widgets must be inspected every 180 days under the Widget Safety Regulation 2024.
Inspection records shall be retained for 5 years.
"""

DOC_INCIDENT = """# Widget Incident at Northgate Depot

On 12 March 2025 a widget at Northgate Depot vented and ignited. The investigation
found that the vent path was obstructed by packaging material. Core temperature
reached 140 °C before ignition. No injuries were reported.

## Findings

The operator had not performed an inspection for 400 days. The obstruction was the
probable cause of the pressure build-up.
"""

ONTOLOGY = {
    "entities": [
        {"id": "widget", "name": "Widget", "kind": "artefact", "aliases": ["widgets"]},
        {"id": "overheating", "name": "Overheating", "kind": "failure_mode",
         "aliases": ["overheat", "thermal excursion"]},
        {"id": "venting", "name": "Venting", "kind": "failure_mode"},
        {"id": "off_gas_detection", "name": "Off-gas detection", "kind": "control",
         "aliases": ["gas detection", "carbon monoxide detection"]},
        {"id": "obstructed_vent", "name": "Obstructed vent path", "kind": "cause"},
    ],
    "relationships": [
        {"subject": "widget", "predicate": "susceptible_to", "object": "overheating"},
        {"subject": "overheating", "predicate": "leads_to", "object": "venting"},
        {"subject": "venting", "predicate": "detected_by", "object": "off_gas_detection"},
        {"subject": "venting", "predicate": "triggered_by", "object": "obstructed_vent"},
    ],
}

SYSTEM_PROMPT = """You are the Widget Failure specialist.

Answer only from the numbered evidence provided. Cite every claim with [n].
If the evidence is insufficient, say so plainly.
"""

CORE_SUITE = {
    "suite": "core",
    "questions": [
        {
            "id": "t-fact-001",
            "type": "factual",
            "question": "At what core temperature does widget overheating begin?",
            "graders": [{"kind": "must_include", "any_of": ["80"]},
                        {"kind": "must_cite_source", "source_ids": ["src-overheat"]}],
        },
        {
            "id": "t-unans-001",
            "type": "unanswerable",
            "question": "What is the resale value of a widget in Lagos in 2031?",
            "graders": [{"kind": "must_abstain"}],
        },
        {
            "id": "t-scope-001",
            "type": "out_of_scope",
            "question": "How should I treat a bacterial infection?",
            "graders": [{"kind": "must_abstain"}],
        },
    ],
}


def _write_yaml(path: Path, data) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# The environment chooses the model in production: a CI runner exports
# FOUNDRY_LLM_PROVIDER because it has no Ollama. That must not reach in here --
# a test asserting "an unreachable Ollama falls back" tests nothing if the
# runner has already swapped Ollama out. Tests state the provider they mean.
_AMBIENT_VARS = (
    "FOUNDRY_LLM_PROVIDER",
    "FOUNDRY_LLM_MODEL",
    "FOUNDRY_OLLAMA_HOST",
    "OLLAMA_HOST",
    "FOUNDRY_ROOT",
    "FOUNDRY_VAR",
)


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch):
    for name in _AMBIENT_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def ka_dir(tmp_path: Path, monkeypatch) -> Path:
    """A complete synthetic knowledge area, with FOUNDRY_ROOT/VAR pointed at tmp."""
    root = tmp_path / "repo"
    ka = root / "knowledge_areas" / "kb-test-widgets"
    (ka / "prompts").mkdir(parents=True)
    (ka / "evaluation").mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "overheat.md").write_text(DOC_OVERHEAT, encoding="utf-8")
    (docs / "detection.md").write_text(DOC_DETECTION, encoding="utf-8")
    (docs / "incident.md").write_text(DOC_INCIDENT, encoding="utf-8")

    _write_yaml(ka / "manifest.yaml", MANIFEST)
    _write_yaml(ka / "ontology.yaml", ONTOLOGY)
    _write_yaml(ka / "evaluation" / "core.yaml", CORE_SUITE)
    (ka / "prompts" / "system.md").write_text(SYSTEM_PROMPT, encoding="utf-8")

    _write_yaml(
        ka / "sources.yaml",
        {
            "defaults": {"licence": "test-fixture", "refresh_days": 365},
            "sources": [
                {
                    "id": "src-overheat",
                    "uri": str(docs / "overheat.md"),
                    "title": "Widget Overheating Handbook",
                    "publisher": "Widget Standards Body",
                    "source_type": "standard",
                    "authority": 5,
                    "published_at": "2024-06-01",
                    "tags": ["overheating"],
                },
                {
                    "id": "src-detection",
                    "uri": str(docs / "detection.md"),
                    "title": "Detecting Widget Failure",
                    "publisher": "Widget Regulator",
                    "source_type": "regulatory",
                    "authority": 5,
                    "published_at": "2024-01-15",
                },
                {
                    "id": "src-incident",
                    "uri": str(docs / "incident.md"),
                    "title": "Northgate Depot Widget Incident Report",
                    "publisher": "Widget Investigation Bureau",
                    "source_type": "investigation",
                    "authority": 4,
                    "published_at": "2025-05-20",
                },
                {
                    "id": "src-pointer",
                    "uri": "https://example.invalid/paywalled-standard",
                    "title": "Widget Standard 9000 (full text)",
                    "publisher": "Widget Standards Body",
                    "source_type": "standard",
                    "authority": 5,
                    "licence": "proprietary - not redistributable",
                    "mirrored": False,
                },
            ],
        },
    )

    monkeypatch.setenv("FOUNDRY_ROOT", str(root))
    monkeypatch.setenv("FOUNDRY_VAR", str(tmp_path / "var"))
    return ka


@pytest.fixture
def manifest(ka_dir):
    from foundry.manifest import Manifest

    return Manifest.load(ka_dir)


@pytest.fixture
def built(manifest):
    """An ingested and indexed knowledge area: (manifest, store)."""
    from foundry.ingest import ingest_knowledge_area
    from foundry.retrieval import build_semantic_index
    from foundry.storage import Store

    store = Store(manifest.id)
    ingest_knowledge_area(manifest, store)
    build_semantic_index(manifest, store)
    yield manifest, store
    store.close()
