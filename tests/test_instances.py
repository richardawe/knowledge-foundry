"""Templates, collectors and claim verification: knowledge areas per instance.

Two identity rules here were wrong on the first pilot run and are pinned so they
stay right: a bare trading name on a site corroborates nothing (it matched a
Belgian entity to a US company), and a register that could not be reached is
not a register with no match.
"""

from __future__ import annotations

import json

import pytest
import yaml

from foundry import collect, instantiate as inst, paths
from foundry.collect import CollectReport, Document, collect_register, declared_names, normalise_name
from foundry.evaluation.dataset import load_suite
from foundry.ingest.fetchers import FetchError
from foundry.manifest import Manifest
from foundry.storage import Store
from foundry.verify import verify_claims

TEMPLATE_COLLECTORS = yaml.safe_load(
    (paths.repo_root() / "templates" / "company" / "collectors.yaml").read_text())


def _register_cfg(**over):
    cfg = {
        "id": "reg", "kind": "register", "source_type": "official_record", "authority": 5,
        "publisher": "Test Register", "licence": "test", "format": "json",
        "search": "https://register.test/search?q={name_q}",
        "candidates": {"items": "data", "key": "id", "name": "name"},
        "legal_form_words": ["ltd", "limited", "inc", "corp", "corporation", "the"],
        "detail": [{"label": "Record", "url": "https://register.test/{key}",
                    "keep": [{"label": "Legal name", "path": "name"},
                             {"label": "Address", "path": "address"}]}],
    }
    cfg.update(over)
    return cfg


def _fake_fetch(responses):
    def fetch(uri, retries=1):
        if uri not in responses:
            raise FetchError(f"HTTP 404 fetching {uri}")
        value = responses[uri]
        if isinstance(value, Exception):
            raise value
        return json.dumps(value).encode(), "application/json", uri
    return fetch


ADDRESS = "1 Long Street, Somewhere, a postcode long enough to make the record clear the minimum. " * 3


def test_normalise_name_drops_legal_form_and_punctuation():
    drop = {"ltd", "inc", "the"}
    assert normalise_name("The Trade Desk, Inc.", drop) == "trade desk"
    assert normalise_name("ACME LTD", drop) == normalise_name("Acme", drop)


def test_declared_names_need_a_registration_cue():
    cfg = TEMPLATE_COLLECTORS["site_declared_names"]
    text = ("Our customers include Example Corp and Other Ltd.\n"
            "We are registered in England as Widget Works Ltd, company number 01234567.")
    names = [d["name"] for d in declared_names(text, cfg)]
    assert names == ["Widget Works Ltd"]


def test_a_bare_trading_name_on_the_site_corroborates_nothing(monkeypatch):
    """The 'STRIPE' LEI record was a Belgian entity; stripe.com says 'Stripe' everywhere."""
    monkeypatch.setattr(collect, "_fetch_text", _fake_fetch({
        "https://register.test/search?q=Acme": {"data": [{"id": "K1", "name": "ACME"}]},
        "https://register.test/K1": {"name": "ACME", "address": ADDRESS},
    }))
    report = CollectReport()
    collect_register(_register_cfg(), {"name": "Acme"}, report, site_text="Acme makes acme things.")
    record = report.identity["reg"]
    assert record["status"] == "pending_confirmation"
    assert record["entities"][0]["corroborated"] is False
    assert report.documents == []  # held out of the corpus


def test_a_legal_name_on_the_site_corroborates_and_is_indexed(monkeypatch):
    monkeypatch.setattr(collect, "_fetch_text", _fake_fetch({
        "https://register.test/search?q=Acme": {"data": [{"id": "K1", "name": "ACME LTD"}]},
        "https://register.test/K1": {"name": "ACME LTD", "address": ADDRESS},
    }))
    report = CollectReport()
    collect_register(_register_cfg(), {"name": "Acme"}, report,
                     site_text="Copyright 2026 Acme Ltd. All rights reserved.")
    assert report.identity["reg"]["status"] == "resolved"
    assert [d.authority for d in report.documents] == [5]


def test_a_name_the_site_declares_is_searched_and_trusted(monkeypatch):
    monkeypatch.setattr(collect, "_fetch_text", _fake_fetch({
        "https://register.test/search?q=Brand": {"data": [{"id": "X", "name": "BRAND HOLIDAYS LTD"}]},
        "https://register.test/search?q=Runtime%20Things%20Limited":
            {"data": [{"id": "K2", "name": "RUNTIME THINGS LIMITED"}]},
        "https://register.test/K2": {"name": "RUNTIME THINGS LIMITED", "address": ADDRESS},
    }))
    report = CollectReport()
    collect_register(_register_cfg(), {"name": "Brand"}, report, site_text="",
                     declared=[{"name": "Runtime Things Limited", "sentence": "..."}])
    entity = report.identity["reg"]["entities"][0]
    assert entity["key"] == "K2" and entity["corroborated"] and entity["via"] == "declared on the site"


def test_an_unreachable_register_is_not_reported_as_no_match(monkeypatch):
    monkeypatch.setattr(collect, "_fetch_text", _fake_fetch({
        "https://register.test/search?q=Acme": FetchError("HTTP 502 fetching it"),
    }))
    report = CollectReport()
    collect_register(_register_cfg(), {"name": "Acme"}, report, site_text="")
    assert report.identity["reg"]["status"] == "failed"


def test_a_register_needing_a_contact_does_not_run_without_one(monkeypatch):
    monkeypatch.delenv("FOUNDRY_CONTACT_EMAIL", raising=False)
    report = CollectReport()
    collect_register(_register_cfg(requires_env="FOUNDRY_CONTACT_EMAIL"), {"name": "Acme"},
                     report, site_text="")
    assert report.identity["reg"]["status"] == "not_run"


def test_the_company_template_renders_a_valid_knowledge_area(tmp_path, monkeypatch):
    monkeypatch.setattr(inst, "repo_root", lambda: tmp_path)
    (tmp_path / "templates").symlink_to(paths.repo_root() / "templates")
    target = inst.instantiate("company", {"domain": "Example.com", "name": "Example"})
    assert target.name == "co-example-com"
    manifest = Manifest.load(target)
    manifest.validate()
    assert "Example" in manifest.scope.in_scope[0]
    assert all("Example" in topic for topic in manifest.scope.out_of_scope[:3])
    suite = load_suite(target / "evaluation" / "core.yaml")
    assert any(q.should_abstain for q in suite.questions)
    with pytest.raises(Exception):
        inst.instantiate("company", {"domain": "example.com"})  # name missing


def _store_with(tmp_path, monkeypatch, docs):
    """A tiny built knowledge area from inline documents."""
    monkeypatch.setattr(paths, "var_dir", lambda: tmp_path / "var")
    ka = tmp_path / "co-test"
    (ka / "prompts").mkdir(parents=True)
    (ka / "prompts" / "system.md").write_text("test")
    (ka / "manifest.yaml").write_text(yaml.safe_dump({
        "knowledge_area": {"id": "co-test", "name": "Test", "description": "t",
                           "domain": "t", "version": "v0.1.0", "owner": "t"},
        "scope": {"in_scope": ["test"], "out_of_scope": ["medical"]},
        "retrieval": {"chunk": {"target_chars": 400, "overlap_chars": 40, "min_chars": 40},
                      "semantic": {"provider": "lsa", "dim": 4}},
        "evaluation": {"suites": []},
        "governance": {"known_limitations": ["t"], "disclaimer": "t"},
    }))
    (ka / "sources.yaml").write_text(yaml.safe_dump({"sources": [d.as_source() for d in docs]}))
    from foundry.ingest import ingest_knowledge_area
    manifest = Manifest.load(ka)
    store = Store(manifest.id)
    ingest_knowledge_area(manifest, store)
    return store


def _doc(id_, source_type, text):
    return Document(id=id_, title=id_, uri=f"https://x.test/{id_}", text=text, publisher="x",
                    source_type=source_type, authority=3, licence="test", tier=source_type,
                    collector="test")


def test_verdicts_say_who_supports_a_claim(tmp_path, monkeypatch):
    filler = " The company was founded to help teams understand their own workflows better." * 4
    store = _store_with(tmp_path, monkeypatch, [
        _doc("site", "self_published",
             "Menu\nBedtime reading\nPricing\n\nWe build session replay for product teams." + filler),
        _doc("news", "press", "Reporters say Acme offers session replay to product teams." + filler),
        _doc("site2", "self_published", "Our context warehouse stores every customer event for analysis." + filler),
        _doc("other", "community", "At my company the customer success team is understaffed." + filler),
        _doc("site3", "self_published", "Our customer success team onboards every account." + filler),
    ])
    with store:
        verdicts = {v.claim: v.verdict for v in verify_claims(store, [
            "Session replay", "Context warehouse", "Bedtime reading", "Quantum teleportation",
            "Customer success", "Reporters"], subject="Acme")}
    assert verdicts == {
        "Session replay": "corroborated",
        # A user comment about some other company's team is not about this one.
        "Customer success": "self_reported",
        "Context warehouse": "self_reported",
        "Bedtime reading": "label_only",
        "Quantum teleportation": "not_found",
        # One word found in a press item corroborates nothing.
        "Reporters": "not_found",
    }
