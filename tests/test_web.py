"""Public evidence pages, the ask UI and the API (§15, §16, §18, §19).

The load-bearing assertions are about disclosure and isolation: the evidence
page must publish what the system *fails*, a pointer source must never leak its
content, and a private knowledge area must not be served to a request that has
no right to it.
"""

from __future__ import annotations

import json

import pytest

from foundry.evaluation import run_evaluation
from foundry.evidence import knowledge_area_report, source_detail
from foundry.feedback import add_challenge, list_challenges, promote_challenge
from foundry.redteam import run_redteam
from foundry.storage import Store
from foundry.versioning import cut_version, publish_version
from foundry.web import Registry, Router


@pytest.fixture
def router(built):
    manifest, _store = built
    return Router(registry=Registry(manifests={manifest.id: manifest}))


def _get(router, path, method="GET", body=None, tenant=None):
    status, content_type, payload = router.handle(method, path, {}, body or {}, tenant)
    return status, content_type, payload.decode("utf-8")


# -- evidence report ----------------------------------------------------


def test_report_counts_pointer_sources_separately(built):
    manifest, store = built
    report = knowledge_area_report(manifest, store)

    assert report["corpus"]["pointer_sources"] == 1
    assert report["corpus"]["indexed_sources"] == 3
    assert any(p["title"].startswith("Widget Standard 9000") for p in report["limitations"]["pointer_sources"])


def test_report_publishes_failures_not_just_passes(built):
    """§25: here is where it failed. Hiding failures would defeat the product."""
    manifest, store = built
    run_evaluation(manifest, store, persist=True)
    report = knowledge_area_report(manifest, store)

    assert "failures" in report["evaluation"]
    assert report["evaluation"]["total"] > 0


def test_report_includes_declared_limitations(built):
    manifest, store = built
    report = knowledge_area_report(manifest, store)
    assert report["limitations"]["declared"]
    assert report["about"]["disclaimer"]


def test_source_detail_exposes_exactly_what_can_be_cited(built):
    manifest, store = built
    detail = source_detail(store, "src-overheat")

    assert detail["source"]["licence"]
    assert detail["chunks"]
    assert detail["source"]["content_sha256"]


def test_pointer_source_detail_has_no_stored_content(built):
    """The licence guarantee has to hold at the viewer, not only at ingestion."""
    manifest, store = built
    detail = source_detail(store, "src-pointer")

    assert detail["source"]["mirrored"] is False
    assert detail["chunks"] == []
    assert detail["claims"] == []


def test_unknown_source_returns_none(built):
    _manifest, store = built
    assert source_detail(store, "src-does-not-exist") is None


# -- pages --------------------------------------------------------------


def test_index_lists_knowledge_areas(router, built):
    manifest, _store = built
    status, _ct, html = _get(router, "/")
    assert status == 200
    assert manifest.name in html


def test_evidence_page_renders_every_required_section(router):
    """§15 names the sections. All of them must actually be there."""
    status, _ct, html = _get(router, "/k/kb-test-widgets")
    assert status == 200
    for section in ("About", "Sources", "Coverage", "Evaluation", "Updates",
                    "Known limitations", "Challenges"):
        assert section in html, section


def test_evidence_page_invites_scrutiny(router):
    status, _ct, html = _get(router, "/k/kb-test-widgets")
    assert "Try to break it" in html


def test_source_register_marks_pointer_sources(router):
    status, _ct, html = _get(router, "/k/kb-test-widgets/sources")
    assert status == 200
    assert "pointer" in html
    assert "Widget Standard 9000" in html


def test_pointer_source_page_states_that_it_is_not_held(router):
    status, _ct, html = _get(router, "/k/kb-test-widgets/source/src-pointer")
    assert status == 200
    assert "Pointer source" in html
    assert "not stored" in html


def test_evaluation_page_lists_failing_questions(router, built):
    manifest, store = built
    run_evaluation(manifest, store, persist=True)
    status, _ct, html = _get(router, "/k/kb-test-widgets/evaluation")

    assert status == 200
    assert "Questions it currently fails" in html


def test_ask_page_renders_evidence_and_limitations(router):
    status, _ct, html = _get(
        router, "/k/kb-test-widgets/ask", method="POST",
        body={"question": "At what core temperature does overheating begin?"},
    )
    assert status == 200
    assert "Evidence" in html
    assert "Limitations" in html


def test_unknown_page_is_a_404(router):
    status, _ct, _html = _get(router, "/k/kb-test-widgets/nonsense")
    assert status == 404


# -- API ----------------------------------------------------------------


def test_api_ask_returns_the_documented_contract(router):
    """§19 names the fields. Callers depend on them."""
    status, content_type, payload = _get(
        router, "/knowledge/kb-test-widgets/ask", method="POST",
        body={"question": "At what core temperature does overheating begin?"},
    )
    assert status == 200
    assert content_type == "application/json"
    data = json.loads(payload)
    for key in ("answer", "sources", "confidence", "knowledge_version", "evaluation_status"):
        assert key in data


def test_api_ask_requires_a_question(router):
    status, _ct, payload = _get(
        router, "/knowledge/kb-test-widgets/ask", method="POST", body={"question": "  "}
    )
    assert status == 400
    assert "error" in json.loads(payload)


def test_api_ask_rejects_get(router):
    status, _ct, _payload = _get(router, "/knowledge/kb-test-widgets/ask", method="GET")
    assert status == 405


def test_api_ask_on_unknown_area_is_404_json(router):
    status, content_type, _payload = _get(
        router, "/knowledge/kb-nope/ask", method="POST", body={"question": "hi"}
    )
    assert status == 404
    assert content_type == "application/json"


def test_healthz(router):
    status, _ct, payload = _get(router, "/healthz")
    assert status == 200 and json.loads(payload)["ok"] is True


# -- multi-tenancy (§18) ------------------------------------------------


def test_private_area_is_hidden_from_anonymous_requests(built, monkeypatch):
    manifest, _store = built
    manifest.visibility = "private"
    manifest.tenant = "acme"
    router = Router(registry=Registry(manifests={manifest.id: manifest}))

    assert _get(router, f"/k/{manifest.id}")[0] == 404
    assert _get(router, f"/knowledge/{manifest.id}/ask", method="POST",
                body={"question": "hi"})[0] == 404
    assert json.loads(_get(router, "/api/knowledge")[2]) == []


def test_private_area_is_served_to_its_own_tenant(built):
    manifest, _store = built
    manifest.visibility = "private"
    manifest.tenant = "acme"
    router = Router(registry=Registry(manifests={manifest.id: manifest}))

    assert _get(router, f"/k/{manifest.id}", tenant="acme")[0] == 200
    assert _get(router, f"/k/{manifest.id}", tenant="globex")[0] == 404


def test_public_area_needs_no_tenant(router):
    assert _get(router, "/k/kb-test-widgets", tenant=None)[0] == 200


# -- challenges (§14) ---------------------------------------------------


def test_challenge_submitted_through_the_form_is_recorded(router, built):
    manifest, store = built
    status, _ct, html = _get(
        router, "/k/kb-test-widgets/challenges", method="POST",
        body={"question": "Is 80 °C right for every chemistry?",
              "problem": "no conditions given", "by": "a fire engineer"},
    )
    assert status == 200
    with Store(manifest.id) as fresh:
        rows = list_challenges(fresh)
    assert any(r["question"].startswith("Is 80") for r in rows)
    assert "a fire engineer" in html


def test_empty_challenge_is_ignored_rather_than_stored(router, built):
    manifest, store = built
    _get(router, "/k/kb-test-widgets/challenges", method="POST", body={"question": "   "})
    with Store(manifest.id) as fresh:
        assert list_challenges(fresh) == []


def test_accepted_challenge_becomes_a_permanent_test(built):
    """§14 + §24.9: expert feedback is a dataset, not an inbox."""
    import yaml

    manifest, store = built
    challenge_id = add_challenge(
        manifest, store,
        question="Does the 80 °C figure hold for every widget format?",
        claimed_problem="states a single number with no conditions",
        expected="depends on the widget format",
        submitted_by="a reviewer",
    )
    path = promote_challenge(manifest, store, challenge_id)

    data = yaml.safe_load(open(path).read())
    assert data["questions"]
    entry = data["questions"][0]
    assert entry["origin"] == "challenge"
    assert entry["graders"]
    assert list_challenges(store, status="promoted")


def test_promoting_a_challenge_twice_does_not_duplicate_it(built):
    import yaml

    manifest, store = built
    challenge_id = add_challenge(manifest, store, question="Why 80 °C?", expected="conditions")
    path = promote_challenge(manifest, store, challenge_id)
    promote_challenge(manifest, store, challenge_id)

    data = yaml.safe_load(open(path).read())
    assert len(data["questions"]) == 1


def test_a_challenge_needs_a_question(built):
    manifest, store = built
    with pytest.raises(ValueError):
        add_challenge(manifest, store, question="   ")


# -- end to end ---------------------------------------------------------


def test_the_whole_loop_is_visible_on_one_page(built):
    """ingest -> retrieve -> answer -> evaluate -> red team -> version -> publish."""
    manifest, store = built
    version = cut_version(manifest, store)
    report = run_evaluation(manifest, store, persist=True)
    run_redteam(manifest, store, count=8, persist=True)
    publish_version(manifest, store, report=report.as_dict(), force=True)

    page = knowledge_area_report(manifest, store)
    assert page["corpus"]["chunks"] > 0
    assert page["evaluation"]["total"] > 0
    assert page["redteam"]["total"] > 0
    assert page["updates"] and page["updates"][0]["version"] == version.version
    assert page["version"]["published"] == version.version


# -- cross-origin ask (the published page calling a local instance) -----


def test_healthz_reports_which_knowledge_areas_are_served(router):
    """The ask client uses this to tell "not running" from "running the wrong build"."""
    status, _ct, payload = _get(router, "/healthz")
    assert status == 200
    assert json.loads(payload)["knowledge_areas"] == ["kb-test-widgets"]


def test_healthz_advertises_the_build_and_its_cors_support(router):
    """A browser cannot see why a cross-origin call failed; this is how it finds out.

    An instance predating cross-origin support answers /healthz but the reply is
    unreadable from another origin -- indistinguishable, in the browser, from
    nothing listening at all. These fields let the page say which it is.
    """
    _status, _ct, payload = _get(router, "/healthz")
    data = json.loads(payload)
    assert data["cors"] is True
    assert data["version"]


def test_api_routes_allow_cross_origin_calls():
    """A published static page is a different origin from the instance it calls."""
    from foundry.web.app import cors_headers

    headers = dict(cors_headers("https://example.github.io", "/knowledge/kb-x/ask"))
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert "POST" in headers["Access-Control-Allow-Methods"]


def test_write_routes_stay_same_origin():
    """A page you did not open must not be able to record a challenge for you."""
    from foundry.web.app import cors_headers

    assert cors_headers("https://evil.example", "/k/kb-x/challenges") == []
    assert cors_headers("https://evil.example", "/k/kb-x/ask") == []


def test_cors_origins_can_be_narrowed(monkeypatch):
    from foundry.web.app import cors_headers

    monkeypatch.setenv("FOUNDRY_CORS_ORIGINS", "https://richardawe.github.io")
    allowed = dict(cors_headers("https://richardawe.github.io", "/api/knowledge"))
    assert allowed["Access-Control-Allow-Origin"] == "https://richardawe.github.io"
    assert cors_headers("https://somewhere.else", "/api/knowledge") == []


def test_the_ask_client_and_engine_are_both_served(router):
    """The page is inert without either, and the engine is the larger half."""
    for asset, marker in (("ask.js", "FoundryEngine"), ("engine.js", "function ask")):
        status, content_type, payload = _get(router, f"/assets/{asset}")
        assert status == 200, asset
        assert "javascript" in content_type
        assert marker in payload, f"{marker} missing from {asset}"


def test_asset_route_rejects_path_traversal(router):
    """An asset path is a filename, never a way out of the asset directory."""
    for path in ("/assets/../app.py", "/assets/nope.js"):
        assert _get(router, path)[0] == 404
