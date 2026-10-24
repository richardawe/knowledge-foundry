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


def test_a_source_removed_from_the_register_stops_answering(tmp_path, monkeypatch):
    """The first pilot run indexed a wrong-company record; later runs dropped it
    from the register, and it stayed in the store, citable, until this rule."""
    filler = " The company was founded to help teams understand their own workflows better." * 4
    keep = _doc("site", "self_published", "Our session replay helps product teams." + filler)
    wrong = _doc("wrong", "official_record", "Legal address: Somewhere else entirely." + filler)
    store = _store_with(tmp_path, monkeypatch, [keep, wrong])
    store.close()
    ka = tmp_path / "co-test"
    (ka / "sources.yaml").write_text(yaml.safe_dump({"sources": [keep.as_source()]}))
    from foundry.ingest import ingest_knowledge_area
    with Store("co-test") as store:
        report = ingest_knowledge_area(Manifest.load(ka), store)
        assert report.retired == ["wrong"]
        assert [r["id"] for r in store.sources()] == ["site"]
        assert not store.conn.execute("SELECT 1 FROM chunks WHERE source_id = 'wrong'").fetchall()


def test_changed_inline_content_is_reindexed_however_recent(tmp_path, monkeypatch):
    filler = " The company was founded to help teams understand their own workflows better." * 4
    store = _store_with(tmp_path, monkeypatch, [_doc("site", "self_published", "Version one." + filler)])
    store.close()
    ka = tmp_path / "co-test"
    changed = _doc("site", "self_published", "Version two of the page." + filler)
    (ka / "sources.yaml").write_text(yaml.safe_dump({"sources": [changed.as_source()]}))
    from foundry.ingest import ingest_knowledge_area
    with Store("co-test") as store:
        ingest_knowledge_area(Manifest.load(ka), store)
        text = store.conn.execute("SELECT text FROM documents WHERE source_id = 'site'").fetchone()["text"]
    assert "Version two" in text


def test_identity_questions_are_answered_from_register_fields(tmp_path, monkeypatch):
    from foundry.records import answer_from_fields
    filler = "\nNature of business (SIC): software." + " Padding to clear the minimum length." * 6
    store = _store_with(tmp_path, monkeypatch, [
        _doc("reg", "official_record",
             "Record for ACME LTD.\n\nLegal name: ACME LTD\nRegistered office address: 1 Long Street\n"
             "Current officer: SMITH, Jane, Director\nCurrent officer: JONES, Bob, Secretary" + filler),
        _doc("site", "self_published", "Our registered office address is a privacy matter." + filler),
    ])
    with store:
        office = answer_from_fields(store, "co-test", "Where is the registered office?",
                                    ["Registered office address"])
        officers = answer_from_fields(store, "co-test", "Who are the officers?", ["Current officer"])
        nothing = answer_from_fields(store, "co-test", "Parent?", ["Ultimate parent legal name"])
    assert "1 Long Street" in office.answer
    assert [s["source_type"] for s in office.sources] == ["official_record"]  # never the site
    assert "SMITH, Jane, Director; JONES, Bob, Secretary" in officers.answer
    assert nothing is None  # falls back to retrieval


def test_corroboration_needs_one_sentence_naming_the_subject(tmp_path, monkeypatch):
    """A job board listing Acme among many employers mentions both, states neither."""
    filler = " The company was founded to help teams understand their own workflows better." * 4
    store = _store_with(tmp_path, monkeypatch, [
        _doc("site", "self_published", "Acme builds data pipelines for retailers." + filler),
        _doc("board", "community",
             "Hiring now: Acme, Initech and others. Initech builds data pipelines for banks." + filler),
    ])
    with store:
        [verdict] = verify_claims(store, ["Data pipelines"], subject="Acme")
    assert verdict.verdict == "self_reported"


class _Model:
    name, model = "fake", "judge-1"

    def __init__(self, answer):
        self.answer = answer

    def complete(self, system, user, **kwargs):
        from foundry.llm.base import LLMResponse
        return LLMResponse(text=self.answer, provider=self.name, model=self.model)


def test_a_model_can_veto_a_sentence_and_is_named_when_it_agrees(tmp_path, monkeypatch):
    filler = " The company was founded to help teams understand their own workflows better." * 4
    store = _store_with(tmp_path, monkeypatch, [
        _doc("site", "self_published", "Acme offers session replay." + filler),
        _doc("news", "press", "Analysts say Acme offers session replay to product teams." + filler),
    ])
    with store:
        [vetoed] = verify_claims(store, ["Session replay"], subject="Acme", entailment=_Model("NO"))
        [agreed] = verify_claims(store, ["Session replay"], subject="Acme", entailment=_Model("YES"))
    assert vetoed.verdict == "self_reported"
    assert agreed.verdict == "corroborated" and "fake:judge-1" in agreed.note


def test_a_keyless_provider_is_refused_as_a_verifier(monkeypatch):
    from foundry.verify import entailment_from_env
    monkeypatch.setenv("FOUNDRY_VERIFY_PROVIDER", "local_extractive")
    with pytest.raises(ValueError):
        entailment_from_env()
    monkeypatch.delenv("FOUNDRY_VERIFY_PROVIDER")
    assert entailment_from_env() is None


def test_the_subject_confirms_or_rejects_a_held_match(monkeypatch):
    responses = {
        "https://register.test/search?q=Acme": {"data": [{"id": "K1", "name": "ACME"},
                                                         {"id": "K2", "name": "ACME"}]},
        "https://register.test/K1": {"name": "ACME", "address": ADDRESS},
        "https://register.test/K2": {"name": "ACME", "address": ADDRESS},
    }
    monkeypatch.setattr(collect, "_fetch_text", _fake_fetch(responses))
    report = CollectReport()
    collect_register(_register_cfg(max_entities=2), {"name": "Acme", "confirmed": ["reg:K1"],
                                                     "rejected": ["reg:K2"]}, report, site_text="")
    entities = report.identity["reg"]["entities"]
    assert [(e["key"], e["corroborated"], e["via"]) for e in entities] == [
        ("K1", True, "confirmed by the subject")]
    assert len(report.documents) == 1


def test_a_wire_press_release_is_the_company_speaking(monkeypatch):
    rss = ("<rss><channel>"
           "<item><title>Acme launches thing</title><link>https://www.businesswire.com/news/1</link>"
           "<description>BOSTON--(BUSINESS WIRE)-- Acme, a global leader in things, today announced a thing "
           "that does more things for more people in more places than any thing before, and will ship "
           "it to customers in every region by the end of the year.</description></item>"
           "<item><title>Acme criticised by analysts</title><link>https://news.example.com/2</link>"
           "<description>Analysts said Acme's new thing does fewer things than promised, according to a "
           "review published this week by an independent research firm that tested it for a month "
           "against three competing products.</description></item>"
           "</channel></rss>")
    monkeypatch.setattr(collect, "_fetch_text", lambda uri, retries=1: (rss.encode(), "application/rss+xml", uri))
    cfg = next(c for c in TEMPLATE_COLLECTORS["collectors"] if c["id"] == "news")
    report = CollectReport()
    collect.collect_feed(cfg, {"name": "Acme"}, report)
    assert [d.source_type for d in report.documents] == ["self_published", "press"]


def test_pages_the_caller_rendered_are_indexed_labelled_and_not_refetched(monkeypatch):
    home = ("<html><head><title>Acme</title></head><body><p>" + "Acme makes things for teams. " * 12
            + "</p><a href='/pricing'>Pricing</a></body></html>")
    fetched = []

    def fetch(uri, retries=1):
        fetched.append(uri)
        if uri == "https://acme.test/":
            return home.encode(), "text/html", uri
        raise FetchError(f"HTTP 404 fetching {uri}")

    monkeypatch.setattr(collect, "_fetch_text", fetch)
    rendered = ("<html><body><nav>Plans</nav><p>" + "Our pricing menu is built in the browser. " * 8
                + "Call us on <a href='tel:+442071234567'>020 7123 4567</a>.</p></body></html>")
    provided = [
        {"url": "https://acme.test/pricing", "html": rendered, "rendered_at": "2026-10-24T10:00:00Z"},
        {"url": "https://elsewhere.test/page", "text": "Not Acme's page. " * 20},
    ]
    cfg = next(c for c in TEMPLATE_COLLECTORS["collectors"] if c["id"] == "site")
    report = CollectReport()
    collect.collect_site({**cfg, "probes": []}, {"domain": "acme.test", "name": "Acme"}, report, provided)

    supplied = [d for d in report.documents if "supplied-by-caller" in d.tags]
    assert [d.uri for d in supplied] == ["https://acme.test/pricing"]
    assert "rendered by the caller on 2026-10-24" in supplied[0].publisher
    assert supplied[0].source_type == "self_published"
    assert "https://acme.test/pricing" not in fetched           # the caller's render wins
    assert any(s["uri"] == "https://elsewhere.test/page" and "not on acme.test" in s["reason"]
               for s in report.skipped)
    facts = report.facts["site"]
    assert facts["pages_crawled"] == 1 and facts["pages_supplied"] == 1
    assert "+442071234567" in [p["value"] for p in facts["phones"]]   # from the rendered tel: link
