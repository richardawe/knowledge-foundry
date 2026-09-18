"""Filesystem layout.

Two roots, and the distinction between them is the whole architecture:

``knowledge_areas/``  the asset  -- plain text, version controlled, human edited
``var/``              derived    -- databases, caches, reports; always rebuildable
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Locate the repository root, overridable for tests and deployments."""
    env = os.environ.get("FOUNDRY_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def knowledge_areas_dir() -> Path:
    return repo_root() / "knowledge_areas"


def knowledge_area_dir(ka_id: str) -> Path:
    return knowledge_areas_dir() / ka_id


def var_dir() -> Path:
    env = os.environ.get("FOUNDRY_VAR")
    return Path(env).resolve() if env else repo_root() / "var"


def ka_var_dir(ka_id: str) -> Path:
    d = var_dir() / ka_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path(ka_id: str) -> Path:
    return ka_var_dir(ka_id) / "kb.sqlite3"


def raw_dir(ka_id: str) -> Path:
    d = ka_var_dir(ka_id) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir(ka_id: str) -> Path:
    d = ka_var_dir(ka_id) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d
