"""Static export and gh-pages deploy.

The static site must never be able to disagree with the live one, and it must
never offer a control that cannot work. Both are testable properties.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from foundry.deploy import DeployError, authenticated_url, deploy_pages, pages_url
from foundry.evaluation import run_evaluation
from foundry.evidence import transcript
from foundry.export import export_site
from foundry.web import Registry, Router


# -- export -------------------------------------------------------------


@pytest.fixture
def site(built, tmp_path):
    manifest, store = built
    run_evaluation(manifest, store, persist=True)
    out = tmp_path / "site"
    report = export_site(out, knowledge_areas=[manifest.id])
    return out, report


def test_export_writes_every_public_page(site):
    out, report = site
    for path in (
        "index.html", "404.html", ".nojekyll", "build.json",
        "k/kb-test-widgets/index.html",
        "k/kb-test-widgets/ask/index.html",
        "k/kb-test-widgets/sources/index.html",
        "k/kb-test-widgets/evaluation/index.html",
        "k/kb-test-widgets/transcript/index.html",
        "k/kb-test-widgets/challenges/index.html",
        "api/knowledge/kb-test-widgets.json",
    ):
        assert (out / path).is_file(), path
    assert report.warnings == []


def test_export_writes_a_page_per_source(site):
    """Including the pointer source, whose page says it is not held."""
    out, _report = site
    pointer = out / "k/kb-test-widgets/source/src-pointer/index.html"
    assert pointer.is_file()
    assert "not stored" in pointer.read_text()
    assert (out / "k/kb-test-widgets/source/src-overheat/index.html").is_file()


def test_nojekyll_is_written(site):
    """Without it, GitHub Pages silently drops files beginning with an underscore."""
    out, _report = site
    assert (out / ".nojekyll").exists()


def test_static_pages_offer_no_dead_forms(site):
    """A control that cannot work is worse than no control.

    The ask form is the exception and is not dead: it posts to a running
    instance over the API, and the client hides it until one answers. The
    challenge form has no such route, so it stays absent.
    """
    out, _report = site
    challenges = (out / "k/kb-test-widgets/challenges/index.html").read_text()
    assert "<form" not in challenges

    ask = (out / "k/kb-test-widgets/ask/index.html").read_text()
    assert 'method="post"' not in ask, "a static page cannot submit a form to itself"
    assert 'id="ask-form" hidden' in ask, "the ask form stays hidden until an instance answers"


def test_static_ask_page_explains_how_to_ask_for_real(site):
    """When nothing is answering, say how to start it -- and where the answers are."""
    out, _report = site
    html = (out / "k/kb-test-widgets/ask/index.html").read_text()
    assert "ollama" in html.lower()
    assert "make serve" in html
    assert "transcript" in html.lower()


def test_base_url_prefixes_every_internal_link(built, tmp_path):
    """A GitHub Pages project site is served from a subpath, not the root."""
    manifest, _store = built
    out = tmp_path / "site"
    export_site(out, base_url="/knowledge-foundry", knowledge_areas=[manifest.id])

    html = (out / "index.html").read_text()
    assert 'href="/knowledge-foundry/' in html
    assert 'href="/k/' not in html


def test_export_and_live_server_render_the_same_content(built, tmp_path):
    """One renderer, two outputs -- so the snapshot cannot drift from the system."""
    manifest, _store = built
    out = tmp_path / "site"
    export_site(out, knowledge_areas=[manifest.id])

    live = Router(registry=Registry(manifests={manifest.id: manifest}))
    _status, _ct, payload = live.handle("GET", f"/k/{manifest.id}/sources", {}, {}, None)

    exported = (out / f"k/{manifest.id}/sources/index.html").read_text()
    # Both list the same sources; only the snapshot chrome differs.
    for source_id in ("src-overheat", "src-detection", "src-pointer"):
        assert source_id in exported
        assert source_id in payload.decode()


def test_api_snapshot_is_valid_json(site):
    out, _report = site
    data = json.loads((out / "api/knowledge/kb-test-widgets.json").read_text())
    assert data["corpus"]["sources"] == 4
    assert "evaluation" in data


def test_export_is_idempotent(built, tmp_path):
    manifest, _store = built
    out = tmp_path / "site"
    first = export_site(out, knowledge_areas=[manifest.id])
    second = export_site(out, knowledge_areas=[manifest.id])
    assert first.pages == second.pages


def test_unknown_knowledge_area_is_a_warning_not_a_crash(tmp_path):
    report = export_site(tmp_path / "site", knowledge_areas=["kb-does-not-exist"])
    assert report.warnings
    assert report.knowledge_areas == []


# -- transcript ---------------------------------------------------------


def test_transcript_records_what_the_system_actually_said(built):
    manifest, store = built
    report = run_evaluation(manifest, store, persist=True)
    data = transcript(store)

    assert data["run_id"] == report.run_id
    assert len(data["entries"]) == report.total
    assert data["passed"] + data["failed"] == report.total
    assert data["provider"], "the transcript must name the model that produced it"


def test_transcript_includes_failures(built):
    """A transcript of only the good answers would be marketing, not evidence."""
    manifest, store = built
    run_evaluation(manifest, store, persist=True)
    data = transcript(store)

    failed = [e for e in data["entries"] if not e["passed"]]
    if failed:
        assert failed[0]["failed_graders"]


def test_transcript_is_empty_before_any_evaluation(built):
    _manifest, store = built
    assert transcript(store)["entries"] == []


# -- deploy -------------------------------------------------------------


def test_pages_url_is_derived_from_the_remote():
    assert pages_url("https://github.com/richardawe/knowledge-foundry.git", "gh-pages") == (
        "https://richardawe.github.io/knowledge-foundry/"
    )
    assert pages_url("git@example.com:thing.git", "gh-pages") is None


def test_credential_helper_is_preferred_over_an_embedded_token(tmp_path, monkeypatch):
    """Embedding a PAT in a URL is the fallback, not the default."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(["git", "config", "credential.helper", "store"], cwd=repo, check=True)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")

    url = "https://github.com/owner/repo.git"
    assert authenticated_url(repo, url) == url  # unchanged: the helper handles it


def test_token_is_embedded_only_when_no_helper_exists(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")

    result = authenticated_url(repo, "https://github.com/owner/repo.git")
    assert result.startswith("https://x-access-token:ghp_secret@github.com/")


def test_deploy_refuses_an_empty_site(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(DeployError, match="nothing to deploy"):
        deploy_pages(empty, repo_dir=repo, push=False)


def test_deploy_refuses_a_non_repository(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("hi")

    with pytest.raises(DeployError, match="not a git repository"):
        deploy_pages(site, repo_dir=tmp_path / "not-a-repo", push=False)


def test_deploy_commits_the_site_to_the_branch(tmp_path):
    """Full local round trip against a bare remote -- no network involved."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)

    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (
        ["git", "init", "--quiet", "-b", "main"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "remote", "add", "origin", str(remote)],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    (repo / "README.md").write_text("hi")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=repo, check=True)
    subprocess.run(["git", "push", "--quiet", "origin", "main"], cwd=repo, check=True)

    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>evidence</h1>")
    (site / ".nojekyll").write_text("")

    result = deploy_pages(site, repo_dir=repo, push=True)
    assert result.pushed and result.commit

    # The branch on the remote really holds the site.
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "gh-pages"],
        cwd=remote, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert "index.html" in listing
    assert ".nojekyll" in listing

    # The working branch is untouched -- that is the point of using a worktree.
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert branch == "main"
    assert not (repo / ".gh-pages-worktree").exists()


def test_redeploying_identical_content_is_a_no_op(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (
        ["git", "init", "--quiet", "-b", "main"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "remote", "add", "origin", str(remote)],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    (repo / "README.md").write_text("hi")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=repo, check=True)

    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>evidence</h1>")

    first = deploy_pages(site, repo_dir=repo, push=True)
    second = deploy_pages(site, repo_dir=repo, push=True)
    assert first.pushed
    assert not second.pushed
    assert "up to date" in second.message


def test_deploy_replaces_stale_pages(tmp_path):
    """A page left over from a previous build would be indistinguishable from current."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (
        ["git", "init", "--quiet", "-b", "main"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "remote", "add", "origin", str(remote)],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    (repo / "README.md").write_text("hi")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=repo, check=True)

    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>v1</h1>")
    (site / "old-area.html").write_text("stale")
    deploy_pages(site, repo_dir=repo, push=True)

    (site / "old-area.html").unlink()
    (site / "index.html").write_text("<h1>v2</h1>")
    deploy_pages(site, repo_dir=repo, push=True)

    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "gh-pages"],
        cwd=remote, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert "old-area.html" not in listing


# -- the published ask box ----------------------------------------------


def test_export_ships_the_ask_client(site):
    out, _report = site
    asset = out / "assets/ask.js"
    assert asset.is_file()
    # It calls the API; it does not reimplement retrieval.
    source = asset.read_text()
    assert "/knowledge/" in source and "/healthz" in source
    for forbidden in ("bm25", "cosine", "tokenize"):
        assert forbidden not in source.lower(), "the client must not reimplement retrieval"


def test_static_ask_page_mounts_the_client(site):
    out, _report = site
    html = (out / "k/kb-test-widgets/ask/index.html").read_text()
    assert 'id="ask-app"' in html
    assert 'data-knowledge-area="kb-test-widgets"' in html
    assert "assets/ask.js" in html


def test_ask_endpoint_is_configurable_at_export(built, tmp_path):
    """So the box can point at a tunnel, not only at localhost."""
    manifest, _store = built
    out = tmp_path / "site"
    export_site(out, knowledge_areas=[manifest.id], api_base="https://kf.example.dev")

    html = (out / f"k/{manifest.id}/ask/index.html").read_text()
    assert 'data-api-base="https://kf.example.dev"' in html


def test_static_ask_page_still_explains_the_offline_case(site):
    out, _report = site
    html = (out / "k/kb-test-widgets/ask/index.html").read_text()
    assert "Nothing is answering yet" in html
    assert "transcript" in html.lower()


def test_the_client_diagnoses_why_a_connection_failed(site):
    """"Nothing is listening" and "listening but blocked" need different fixes."""
    out, _report = site
    source = (out / "assets/ask.js").read_text()

    # The no-cors second probe is what separates the two cases.
    assert 'mode: "no-cors"' in source
    assert "nothing is listening" in source
    assert "git pull" in source
    # And it knows about the one browser that blocks http://localhost from HTTPS.
    assert "safari" in source.lower()


def test_the_client_tries_the_ports_the_project_actually_uses(site):
    out, _report = site
    source = (out / "assets/ask.js").read_text()
    assert "8000" in source and "8080" in source


def test_offline_banner_offers_the_same_origin_route(site):
    """The local instance serves this page itself -- no CORS, nothing to block."""
    out, _report = site
    html = (out / "k/kb-test-widgets/ask/index.html").read_text()
    assert "http://localhost:8000/k/kb-test-widgets/ask" in html
    assert "Safari" in html
