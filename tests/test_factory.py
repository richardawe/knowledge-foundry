"""Success criterion #10: the factory must be domain-agnostic.

"We're not building one clever chatbot. We're building the factory that makes
specialist knowledge systems." The tests here are the ones that keep that true
as the code grows -- it is very easy to fix a KA-001 problem with a line of
KA-001-specific code and not notice that the factory has stopped being a
factory.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from foundry import paths
from foundry.evaluation import run_evaluation
from foundry.ingest import ingest_knowledge_area
from foundry.manifest import Manifest, list_knowledge_areas
from foundry.redteam import run_redteam
from foundry.retrieval import build_semantic_index
from foundry.specialist import Specialist
from foundry.storage import Store
from foundry.versioning import cut_version, publish_version

SRC = Path(__file__).resolve().parents[1] / "src" / "foundry"


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_no_knowledge_area_id_is_hard_coded_in_the_factory():
    """The clearest way the factory could rot: a domain id leaking into code."""
    pattern = re.compile(r"kb-\d{3}-[a-z-]+")
    offenders = []
    for path in _python_files():
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(SRC)}: {match.group(0)}")
    assert offenders == [], f"knowledge-area ids found in factory code: {offenders}"


def test_no_domain_vocabulary_is_hard_coded_in_the_factory():
    """Domain terms may appear in prose, never in code.

    Docstrings cite the battery domain freely -- explaining *why* a design
    choice was made needs a concrete example. What must never happen is a
    domain term reaching executable code: a string literal, a name, a branch.
    That is the moment the factory quietly becomes one chatbot.
    """
    import ast

    banned = (
        "thermal runaway", "lithium", "battery", "batteries", "e-bike", "nfpa",
        "9540", "vapour cloud", "dust explosion", "bleve", "fresnel",
    )
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docstrings.add(doc)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                lowered = node.value.lower()
                for term in banned:
                    if term in lowered:
                        offenders.append(
                            f"{path.relative_to(SRC)}:{node.lineno}: string contains {term!r}"
                        )
            elif isinstance(node, ast.Name):
                lowered = node.id.lower()
                for term in banned:
                    if term.replace(" ", "_") in lowered:
                        offenders.append(
                            f"{path.relative_to(SRC)}:{node.lineno}: name {node.id!r}"
                        )
    assert offenders == [], f"domain vocabulary in factory code: {offenders}"


def test_the_repository_ships_more_than_one_knowledge_area():
    """Criterion #10 is not provable with a sample size of one."""
    assert len(list_knowledge_areas()) >= 2


@pytest.mark.parametrize("ka_id", list_knowledge_areas())
def test_every_shipped_knowledge_area_is_valid(ka_id):
    manifest = Manifest.load(ka_id)
    assert manifest.id == ka_id
    assert manifest.system_prompt()
    assert manifest.sources_path.is_file()
    assert manifest.governance.known_limitations, "a knowledge area must declare limitations"


@pytest.mark.parametrize("ka_id", list_knowledge_areas())
def test_every_shipped_source_register_is_valid(ka_id):
    from foundry.ingest import load_source_register

    manifest = Manifest.load(ka_id)
    sources = load_source_register(manifest)
    assert sources
    for source in sources:
        assert source["licence"], f"{source['id']} has no licence"


@pytest.mark.parametrize("ka_id", list_knowledge_areas())
def test_every_shipped_evaluation_suite_is_loadable(ka_id):
    from foundry.evaluation.dataset import all_questions, load_suites

    manifest = Manifest.load(ka_id)
    questions = all_questions(load_suites(manifest))
    assert questions, f"{ka_id} has no evaluation questions"
    for question in questions:
        assert question.graders


def test_a_brand_new_knowledge_area_needs_no_code_changes(tmp_path, monkeypatch):
    """The end-to-end proof: a directory of text, and the whole loop runs.

    This builds a knowledge area in a domain nothing in the codebase has ever
    seen -- not batteries, not fire -- and drives it through ingest, index,
    retrieval, answering, evaluation, red team, versioning and publishing
    without importing anything domain-specific.
    """
    root = tmp_path / "repo"
    ka_id = "kb-999-lighthouse-optics"
    ka = root / "knowledge_areas" / ka_id
    (ka / "prompts").mkdir(parents=True)
    (ka / "evaluation").mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "lenses.md").write_text(
        "# Fresnel lens orders\n\n"
        "A first order Fresnel lens has a focal length of 920 mm and stands over "
        "2.5 metres tall. It was the largest order in routine service. Keepers "
        "rotated the optic on a mercury float bearing to reduce friction.\n\n"
        "## 2.1 Maintenance\n\n"
        "The mercury float must be cleaned annually to prevent oxidation. "
        "Oxidised mercury increases rotational drag and slows the characteristic "
        "flash interval, which makes the light harder to identify from sea.\n",
        encoding="utf-8",
    )

    (ka / "manifest.yaml").write_text(yaml.safe_dump({
        "knowledge_area": {
            "id": ka_id, "name": "Lighthouse Optics Intelligence",
            "description": "A domain the factory has never seen.",
            "domain": "historical optics", "version": "v0.1.0", "owner": "test",
        },
        "scope": {"in_scope": ["fresnel lens orders", "optic rotation"],
                  "out_of_scope": ["medical diagnosis"]},
        "retrieval": {"top_k": 4, "candidate_k": 20,
                      "chunk": {"target_chars": 400, "overlap_chars": 60, "min_chars": 80},
                      "semantic": {"provider": "lsa", "dim": 6}},
        "specialist": {"sufficiency": {"min_evidence_score": 0.001, "min_chunks": 1}},
        "evaluation": {"suites": ["evaluation/core.yaml"]},
        "governance": {"known_limitations": ["A synthetic knowledge area."],
                       "disclaimer": "Not advice."},
    }), encoding="utf-8")

    (ka / "sources.yaml").write_text(yaml.safe_dump({
        "defaults": {"licence": "test-fixture"},
        "sources": [{
            "id": "src-lenses", "uri": str(docs / "lenses.md"),
            "title": "Fresnel Lens Orders", "publisher": "Test Optical Society",
            "source_type": "reference", "authority": 4, "published_at": "2024-01-01",
        }],
    }), encoding="utf-8")

    (ka / "ontology.yaml").write_text(yaml.safe_dump({
        "entities": [
            {"id": "fresnel_lens", "name": "Fresnel lens", "kind": "artefact"},
            {"id": "mercury_float", "name": "Mercury float bearing",
             "kind": "component", "aliases": ["mercury float"]},
            {"id": "oxidation", "name": "Oxidation", "kind": "failure_mode"},
        ],
        "relationships": [
            {"subject": "fresnel_lens", "predicate": "rotates_on", "object": "mercury_float"},
            {"subject": "mercury_float", "predicate": "susceptible_to", "object": "oxidation"},
        ],
    }), encoding="utf-8")

    (ka / "prompts" / "system.md").write_text(
        "You are a lighthouse optics specialist. Answer only from the evidence, "
        "cite with [n], and say so when the evidence is insufficient.\n",
        encoding="utf-8",
    )

    (ka / "evaluation" / "core.yaml").write_text(yaml.safe_dump({
        "suite": "core",
        "questions": [
            {"id": "lo-001", "type": "factual",
             "question": "What is the focal length of a first order Fresnel lens?",
             "expected_sources": ["src-lenses"],
             "graders": [{"kind": "must_include", "any_of": ["920"]},
                         {"kind": "must_cite_source", "source_ids": ["src-lenses"]}]},
            {"id": "lo-002", "type": "unanswerable",
             "question": "How many lighthouses were built in Patagonia in 1873?",
             "graders": [{"kind": "must_abstain"}]},
        ],
    }), encoding="utf-8")

    monkeypatch.setenv("FOUNDRY_ROOT", str(root))
    monkeypatch.setenv("FOUNDRY_VAR", str(tmp_path / "var"))

    # From here on, nothing is domain-specific.
    assert ka_id in list_knowledge_areas()
    manifest = Manifest.load(ka_id)
    store = Store(manifest.id)

    ingest = ingest_knowledge_area(manifest, store)
    assert ingest.indexed == 1 and not ingest.failures
    assert build_semantic_index(manifest, store).chunks_encoded > 0

    version = cut_version(manifest, store)
    assert version.version == "v0.1.0"

    answer = Specialist(manifest, store).ask(
        "What is the focal length of a first order Fresnel lens?"
    )
    assert "920" in answer.answer
    assert answer.cited_source_ids == {"src-lenses"}

    report = run_evaluation(manifest, store, persist=False)
    assert report.total == 2
    assert report.metrics["citation_validity"] == 1.0

    redteam = run_redteam(manifest, store, count=8, persist=False)
    assert redteam.total == 8

    result = publish_version(manifest, store, report=report.as_dict(), force=True)
    assert result.published
    store.close()
