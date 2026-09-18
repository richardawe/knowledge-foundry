"""The manifest is the factory's only contract with a knowledge area (§6).

These tests exist because a manifest typo that silently degrades retrieval is
far more expensive than a loud failure at load time.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from foundry.manifest import Manifest, ManifestError
from tests.conftest import MANIFEST


def test_loads_and_exposes_required_fields(manifest):
    assert manifest.id == "kb-test-widgets"
    assert manifest.domain == "testing"
    assert manifest.visibility == "public"
    assert manifest.retrieval.top_k == 4
    assert manifest.specialist.llm_provider == "local_extractive"


def test_defaults_are_applied_for_omitted_blocks():
    data = {"knowledge_area": MANIFEST["knowledge_area"]}
    loaded = Manifest.from_dict(copy.deepcopy(data))
    assert loaded.retrieval.fusion_k == 60
    assert loaded.retrieval.fusion_weights["keyword"] == 1.0
    assert loaded.evaluation.thresholds["citation_validity"] == 0.95


@pytest.mark.parametrize("missing", ["id", "name", "description", "domain", "version", "owner"])
def test_missing_required_field_is_rejected(missing):
    data = copy.deepcopy(MANIFEST)
    del data["knowledge_area"][missing]
    with pytest.raises(ManifestError, match=missing):
        Manifest.from_dict(data)


def test_unknown_visibility_is_rejected():
    data = copy.deepcopy(MANIFEST)
    data["knowledge_area"]["visibility"] = "semi-public"
    with pytest.raises(ManifestError, match="visibility"):
        Manifest.from_dict(data)


def test_all_retrievers_disabled_is_rejected():
    data = copy.deepcopy(MANIFEST)
    data["retrieval"] = {
        "keyword": {"enabled": False},
        "semantic": {"enabled": False},
        "structured": {"enabled": False},
        "graph": {"enabled": False},
    }
    with pytest.raises(ManifestError, match="at least one retriever"):
        Manifest.from_dict(data)


def test_overlap_larger_than_chunk_is_rejected():
    data = copy.deepcopy(MANIFEST)
    data["retrieval"]["chunk"] = {"target_chars": 400, "overlap_chars": 500}
    with pytest.raises(ManifestError, match="overlap_chars"):
        Manifest.from_dict(data)


def test_id_must_match_directory_name(ka_dir):
    """The id is the primary key everywhere; a mismatch would corrupt var/ paths."""
    data = yaml.safe_load((ka_dir / "manifest.yaml").read_text())
    data["knowledge_area"]["id"] = "kb-something-else"
    (ka_dir / "manifest.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ManifestError, match="does not match its directory"):
        Manifest.load(ka_dir)


def test_content_hash_is_stable_and_sensitive(manifest):
    first = manifest.content_hash()
    assert first == manifest.content_hash()

    changed = copy.deepcopy(MANIFEST)
    changed["retrieval"]["top_k"] = 9
    assert Manifest.from_dict(changed).content_hash() != first


def test_content_hash_ignores_cosmetic_fields(manifest):
    """Renaming a knowledge area must not invalidate its published version."""
    changed = copy.deepcopy(MANIFEST)
    changed["knowledge_area"]["description"] = "A completely different description."
    assert Manifest.from_dict(changed).content_hash() == manifest.content_hash()


def test_system_prompt_loads_from_the_knowledge_area(manifest):
    assert "Widget Failure specialist" in manifest.system_prompt()
