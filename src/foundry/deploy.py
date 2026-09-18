"""Publish the exported site to a ``gh-pages`` branch.

The mechanism is the one already proven in the sibling ``localtest`` project:
a **git worktree** checkout of the deploy branch, so the working tree of the
branch you are on is never disturbed, and a force-add of the built site.

Authentication follows the same rule as that project, and for the same reason:
if a credential helper is configured (macOS keychain, say) let git use it, and
only fall back to embedding a token when there is none — which is the case in a
bare CI runner.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BRANCH = "gh-pages"


class DeployError(RuntimeError):
    pass


@dataclass
class DeployResult:
    branch: str
    commit: str | None
    pushed: bool
    message: str
    url_hint: str | None = None

    def render(self) -> str:
        lines = [self.message]
        if self.url_hint:
            lines.append(f"  once Pages is enabled: {self.url_hint}")
        return "\n".join(lines)


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise DeployError(
            f"git {' '.join(cmd[1:])} failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return result


def remote_url(repo_dir: Path, remote: str = "origin") -> str:
    result = _run(["git", "remote", "get-url", remote], cwd=repo_dir)
    return result.stdout.strip()


def authenticated_url(repo_dir: Path, url: str) -> str:
    """Prefer credentials git already has; embed a token only as a last resort.

    Two ways git may already be able to push, and both must be recognised or we
    would needlessly bake a token into a URL:

    * a credential helper (the macOS keychain, say);
    * an ``http.<url>.extraheader`` carrying an authorization header, which is
      how ``actions/checkout`` authenticates a runner.
    """
    helper = _run(["git", "config", "credential.helper"], cwd=repo_dir, check=False)
    if helper.stdout.strip():
        return url
    extraheader = _run(
        ["git", "config", "--get-regexp", r"http\..*\.extraheader"],
        cwd=repo_dir, check=False,
    )
    if extraheader.stdout.strip():
        return url
    token = os.environ.get("GITHUB_TOKEN", "")
    if token and url.startswith("https://github.com/"):
        return url.replace("https://", f"https://x-access-token:{token}@")
    return url


def pages_url(url: str, branch: str) -> str | None:
    """Guess the published URL so the operator has something to click."""
    if "github.com" not in url:
        return None
    path = url.split("github.com", 1)[1].lstrip(":/").removesuffix(".git")
    if "/" not in path:
        return None
    owner, repo = path.split("/", 1)
    return f"https://{owner}.github.io/{repo}/"


def _ensure_branch(repo_dir: Path, push_url: str, branch: str) -> bool:
    """Create the deploy branch on the remote as an orphan if it is missing."""
    listing = _run(["git", "ls-remote", "--heads", push_url, branch], cwd=repo_dir, check=False)
    if branch in listing.stdout:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run(["git", "init", "--quiet"], cwd=tmp_path)
        _run(["git", "checkout", "--quiet", "--orphan", branch], cwd=tmp_path)
        (tmp_path / "index.html").write_text(
            "<!DOCTYPE html><meta charset=utf-8><title>Deploying</title>"
            "<p>Deploying…</p>",
            encoding="utf-8",
        )
        (tmp_path / ".nojekyll").write_text("", encoding="utf-8")
        _run(["git", "add", "-A"], cwd=tmp_path)
        _run(["git", "-c", "user.email=noreply@anthropic.com",
              "-c", "user.name=knowledge-foundry",
              "commit", "--quiet", "-m", f"chore: initialise {branch}"], cwd=tmp_path)
        _run(["git", "push", "--quiet", push_url, f"HEAD:refs/heads/{branch}"], cwd=tmp_path)
    return True


def deploy_pages(
    site_dir: Path | str,
    repo_dir: Path | str = ".",
    branch: str = DEFAULT_BRANCH,
    remote: str = "origin",
    message: str | None = None,
    push: bool = True,
) -> DeployResult:
    """Copy ``site_dir`` onto ``branch`` via a worktree and push it."""
    site = Path(site_dir).resolve()
    repo = Path(repo_dir).resolve()

    if not site.is_dir() or not any(site.iterdir()):
        raise DeployError(f"nothing to deploy: {site} is missing or empty")
    if not (repo / ".git").exists():
        raise DeployError(f"{repo} is not a git repository")

    url = remote_url(repo, remote)
    push_url = authenticated_url(repo, url)
    created = _ensure_branch(repo, push_url, branch)

    _run(["git", "fetch", "--quiet", push_url, f"{branch}:refs/remotes/{remote}/{branch}"],
         cwd=repo, check=False)

    worktree = repo / ".gh-pages-worktree"
    # A worktree left behind by an interrupted run would otherwise block this.
    _run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, check=False)
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)
    _run(["git", "worktree", "prune"], cwd=repo, check=False)

    added = _run(
        ["git", "worktree", "add", "-B", branch, str(worktree), f"{remote}/{branch}"],
        cwd=repo, check=False,
    )
    if added.returncode != 0:
        _run(["git", "worktree", "add", "-B", branch, str(worktree)], cwd=repo)

    try:
        # Replace the branch contents wholesale: a stale page left behind would
        # be indistinguishable from a current one, which is the one thing a
        # published evidence page must never be.
        for entry in worktree.iterdir():
            if entry.name == ".git":
                continue
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        shutil.copytree(site, worktree, dirs_exist_ok=True)

        _run(["git", "add", "-A"], cwd=worktree)
        status = _run(["git", "status", "--porcelain"], cwd=worktree)
        if not status.stdout.strip():
            return DeployResult(
                branch=branch, commit=None, pushed=False,
                message=f"{branch} is already up to date; nothing to push",
                url_hint=pages_url(url, branch),
            )

        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        _run(
            ["git", "-c", "user.email=noreply@anthropic.com",
             "-c", "user.name=knowledge-foundry",
             "commit", "--quiet", "-m", message or f"site: publish evidence pages ({stamp})"],
            cwd=worktree,
        )
        commit = _run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()

        if not push:
            return DeployResult(
                branch=branch, commit=commit, pushed=False,
                message=f"committed {commit[:10]} to {branch} (not pushed)",
                url_hint=pages_url(url, branch),
            )

        _run(["git", "push", "--quiet", push_url, f"HEAD:refs/heads/{branch}"], cwd=worktree)
        note = " (branch created)" if created else ""
        return DeployResult(
            branch=branch, commit=commit, pushed=True,
            message=f"published {commit[:10]} to {branch}{note}",
            url_hint=pages_url(url, branch),
        )
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, check=False)
        _run(["git", "worktree", "prune"], cwd=repo, check=False)
