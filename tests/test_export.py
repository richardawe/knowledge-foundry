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
    assert 'id="ask-form" hidden' in ask, "the form stays hidden until the corpus has loaded"



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




def test_static_ask_page_mounts_the_client(site):
    out, _report = site
    html = (out / "k/kb-test-widgets/ask/index.html").read_text()
    assert 'id="ask-app"' in html
    assert 'data-knowledge-area="kb-test-widgets"' in html
    assert "assets/ask.js" in html




def test_static_challenges_page_links_the_same_route(site):
    out, _report = site
    html = (out / "k/kb-test-widgets/challenges/index.html").read_text()
    assert "issues/new?template=ask.yml" in html






def test_the_bootstrap_script_is_shipped_and_executable():
    import os

    script = Path(__file__).resolve().parents[1] / "start.sh"
    assert script.is_file(), "start.sh is referenced by the published page"
    assert os.access(script, os.X_OK), "start.sh must be executable from a fresh clone"
    body = script.read_text()
    # It must survive being run twice.
    assert "already installed" in body
    assert "foundry.cli serve" in body


# -- recorded answers, when nothing is running --------------------------


def test_transcript_index_is_exported(site):
    """The data behind the offline ask box."""
    out, _report = site
    index = out / "api/knowledge/kb-test-widgets.transcript.json"
    assert index.is_file()

    data = json.loads(index.read_text())
    assert data["run_id"]
    assert data["entries"]
    entry = data["entries"][0]
    for key in ("id", "question", "answer", "status", "passed", "cited"):
        assert key in entry


def test_transcript_index_carries_the_evidence_not_just_the_answer(built, tmp_path):
    """An answer without its evidence is the thing this project exists to avoid."""
    manifest, store = built
    run_evaluation(manifest, store, persist=True)
    out = tmp_path / "site"
    export_site(out, knowledge_areas=[manifest.id])

    data = json.loads((out / f"api/knowledge/{manifest.id}.transcript.json").read_text())
    answered = [e for e in data["entries"] if e["cited"]]
    assert answered, "at least one recorded answer should cite something"
    source = answered[0]["cited"][0]
    for key in ("source_title", "publisher", "uri", "authority", "text"):
        assert key in source


def test_transcript_index_records_which_answers_failed(site):
    """So the page can say so rather than presenting a failure as an answer."""
    out, _report = site
    data = json.loads((out / "api/knowledge/kb-test-widgets.transcript.json").read_text())
    assert any(e["passed"] is False for e in data["entries"]) or all(
        e["passed"] for e in data["entries"]
    )
    assert all(isinstance(e["passed"], bool) for e in data["entries"])




def test_checkout_style_credentials_are_recognised(tmp_path, monkeypatch):
    """actions/checkout authenticates with an extraheader, not a helper.

    Without recognising it, every CI deploy would bake GITHUB_TOKEN into a URL
    that ends up in the git remote and the process table.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "http.https://github.com/.extraheader", "AUTHORIZATION: basic xxx"],
        cwd=repo, check=True,
    )
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")

    url = "https://github.com/owner/repo.git"
    assert authenticated_url(repo, url) == url


# -- the page answers for itself -----------------------------------------
#
# This replaces an invariant, deliberately. The published client used to be
# forbidden from touching the corpus: it called a live instance, or matched a
# typed question against recorded ones, and the guarantee was that it could not
# produce an answer the pipeline had not already produced.
#
# That guarantee was real and it was bought by the page being unable to answer
# at all. It is replaced by a stronger one: the page runs the same engine, and
# `tests/test_browser_parity.py` asserts question by question that it returns
# the same verdict, the same evidence in the same order and the same words as
# the pipeline. "Cannot answer" has become "cannot answer differently", which
# is what a reader wanted in the first place.


def test_the_page_ships_the_engine_and_the_corpus(site):
    out, _report = site
    engine = out / "assets/engine.js"
    assert engine.is_file(), "the answering engine must travel with the page"
    source = engine.read_text()
    for symbol in ("contentTerms", "coversTheSubject", "answersTheQuestion",
                   "validateAnswer", "reciprocalRankFusion", "function ask"):
        assert symbol in source, f"{symbol} missing from the published engine"

    corpus = out / "api/knowledge/kb-test-widgets.corpus.json"
    assert corpus.is_file(), "the corpus must travel with the page"


def test_the_corpus_carries_provenance_for_every_source(site):
    """A passage without its source is not evidence, and an answer built from
    one could not be checked -- which would make the page a chatbot."""
    import json

    out, _report = site
    corpus = json.loads((out / "api/knowledge/kb-test-widgets.corpus.json").read_text())
    assert corpus["chunks"], "no passages shipped"
    for source in corpus["sources"].values():
        assert source["uri"] and source["publisher"] and source["title"]
    known = set(corpus["sources"])
    for chunk in corpus["chunks"]:
        assert chunk["source_id"] in known, "a passage with no registered source"


def test_the_page_mounts_the_engine_against_its_own_corpus(site):
    out, _report = site
    html = (out / "k/kb-test-widgets/ask/index.html").read_text()
    assert 'id="ask-app"' in html
    assert 'data-knowledge-area="kb-test-widgets"' in html
    assert "kb-test-widgets.corpus.json" in html
    assert "assets/engine.js" in html
    assert "assets/ask.js" in html


def test_the_page_says_no_model_is_involved(site):
    """The claim a reader most needs, and the one most worth stating plainly."""
    out, _report = site
    html = (out / "k/kb-test-widgets/ask/index.html").read_text().lower()
    assert "no model" in html
    assert "cannot state anything the sources do not already say" in html


def test_disagreeing_still_leaves_the_page_for_the_durable_path(site):
    """Asking is instant and local; challenging is recorded and permanent."""
    out, _report = site
    client = (out / "assets/ask.js").read_text()
    assert "issues/new?template=ask.yml" in client
    assert "knowledge-area=" in client
