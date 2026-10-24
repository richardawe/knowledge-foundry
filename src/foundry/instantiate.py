"""Templates: render a directory that writes knowledge areas.

A template is a knowledge-area directory with Jinja placeholders and a
``template.yaml`` that names its parameters and how an instance id is formed.
Rendering one is the whole of "create a knowledge area for this instance":
nothing under ``src/foundry/`` learns what the instance is.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import date
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from .manifest import ManifestError
from .paths import repo_root

TEXT_SUFFIXES = {".yaml", ".yml", ".md", ".txt", ".json"}


def templates_dir() -> Path:
    return repo_root() / "templates"


def load_template(name: str) -> tuple[Path, dict]:
    directory = templates_dir() / name
    spec_path = directory / "template.yaml"
    if not spec_path.is_file():
        raise ManifestError(f"no template {name!r} (looked for {spec_path})")
    return directory, yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}


def instantiate(name: str, params: dict, force: bool = False) -> Path:
    """Render template ``name`` with ``params``; return the new knowledge-area directory."""
    directory, spec = load_template(name)
    missing = [p for p in spec.get("parameters", []) if not params.get(p)]
    if missing:
        raise ManifestError(f"template {name!r} needs parameters: {', '.join(missing)}")

    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    context = {**params, "today": date.today().isoformat()}
    ka_id = env.from_string(spec["id"]).render(**context).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", ka_id):
        raise ManifestError(f"template {name!r} produced an invalid id {ka_id!r}")
    context["id"] = ka_id

    # FOUNDRY_INSTANCES moves instances out of the code tree (a container's
    # data volume, say): they hold collected material, not configuration.
    base = os.environ.get("FOUNDRY_INSTANCES")
    target = (Path(base) if base else repo_root() / spec.get("into", "instances")) / ka_id
    if target.exists():
        if not force:
            raise ManifestError(f"{target} already exists; pass force to re-render it")
        shutil.rmtree(target)

    skip = {"template.yaml", "README.md"}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_dir() or str(relative) in skip:
            continue
        out = target / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in TEXT_SUFFIXES:
            out.write_text(env.from_string(path.read_text(encoding="utf-8")).render(**context),
                           encoding="utf-8")
        else:
            shutil.copy2(path, out)
    (target / "instance.yaml").write_text(
        yaml.safe_dump({"template": name, "id": ka_id, "params": params,
                        "rendered": context["today"]}, sort_keys=False),
        encoding="utf-8")
    return target
