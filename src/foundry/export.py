"""Static site export.

Renders the same templates the live server uses into a directory of HTML, so
the public evidence surface (§15) can be hosted anywhere static — GitHub Pages,
object storage, a CDN — for nothing.

The split is honest rather than arbitrary. Everything the evidence page shows
is a *published artefact of a build*: the source register, every stored
passage, the evaluation results, the red-team findings, the version history.
None of that needs a server. What does need one is asking a new question, which
runs hybrid retrieval and a local model — so the exported ask page says so and
points at the transcript, which contains every answer the system has actually
given, generated locally by Ollama.

One renderer, two outputs. The static site cannot drift from the live one,
because there is no second implementation to drift.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import Manifest, list_knowledge_areas
from .storage import Store


@dataclass
class ExportReport:
    out_dir: str
    base_url: str
    generated_at: str
    api_base: str = ""
    pages: int = 0
    knowledge_areas: list[str] = field(default_factory=list)
    bytes_written: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "out_dir": self.out_dir,
            "base_url": self.base_url,
            "generated_at": self.generated_at,
            "pages": self.pages,
            "knowledge_areas": self.knowledge_areas,
            "bytes_written": self.bytes_written,
            "warnings": self.warnings,
        }

    def render(self) -> str:
        lines = [
            f"exported {self.pages} pages ({self.bytes_written / 1024:.0f} KiB) "
            f"to {self.out_dir}",
            f"  base URL: {self.base_url or '/'}",
            f"  ask endpoint: {self.api_base}",
            f"  knowledge areas: {', '.join(self.knowledge_areas) or 'none'}",
        ]
        for warning in self.warnings:
            lines.append(f"  WARNING {warning}")
        return "\n".join(lines)


def export_site(
    out_dir: Path | str,
    base_url: str = "",
    knowledge_areas: list[str] | None = None,
    clean: bool = True,
    api_base: str = "http://localhost:8000",
) -> ExportReport:
    """Render the whole site to ``out_dir``.

    ``api_base`` is the instance the published ask box calls. It defaults to the
    address ``make serve`` uses, so a reader running the project locally gets a
    working ask box with no configuration. Point it at a tunnelled HTTPS
    endpoint to make the box work for readers who are not running anything.
    """
    from .web.app import STATIC_DIR, Registry, Router

    out = Path(out_dir)
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    wanted = knowledge_areas or list_knowledge_areas()
    manifests: dict[str, Manifest] = {}
    report = ExportReport(
        out_dir=str(out.resolve()),
        base_url=base_url,
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        api_base=api_base,
    )

    for ka_id in wanted:
        try:
            manifests[ka_id] = Manifest.load(ka_id)
        except Exception as exc:
            report.warnings.append(f"skipped {ka_id}: {exc}")

    router = Router(registry=Registry(manifests=manifests), base_url=base_url, static=True)
    # Every rendered page carries the snapshot timestamp in its header.
    router.env.globals["generated_at"] = report.generated_at
    router.env.globals["api_base"] = api_base

    def write(rel_path: str, payload: bytes) -> None:
        target = out / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        report.pages += 1
        report.bytes_written += len(payload)

    def render(route: str, rel_path: str) -> None:
        status, _content_type, payload = router.handle("GET", route, {}, {}, None)
        if status != 200:
            report.warnings.append(f"{route} returned {status}; not written")
            return
        write(rel_path, payload)

    # GitHub Pages runs Jekyll by default, which drops files and directories
    # beginning with an underscore. This opts out.
    write(".nojekyll", b"")
    render("/", "index.html")

    # The ask client. It calls the live API rather than reimplementing anything,
    # so the published page cannot drift from the system it describes.
    for asset in sorted(STATIC_DIR.glob("*")):
        if asset.is_file():
            write(f"assets/{asset.name}", asset.read_bytes())

    for ka_id in manifests:
        base = f"k/{ka_id}"
        render(f"/k/{ka_id}", f"{base}/index.html")
        for page in ("ask", "sources", "evaluation", "transcript", "challenges"):
            render(f"/k/{ka_id}/{page}", f"{base}/{page}/index.html")

        with Store(ka_id) as store:
            source_ids = [row["id"] for row in store.sources()]
        for source_id in source_ids:
            render(f"/k/{ka_id}/source/{source_id}", f"{base}/source/{source_id}/index.html")

        # The machine-readable snapshot, so the data is usable without scraping.
        status, _ct, payload = router.handle("GET", f"/api/knowledge/{ka_id}", {}, {}, None)
        if status == 200:
            write(f"api/knowledge/{ka_id}.json", payload)

        report.knowledge_areas.append(ka_id)

    status, _ct, payload = router.handle("GET", "/api/knowledge", {}, {}, None)
    if status == 200:
        write("api/knowledge/index.json", payload)

    # GitHub Pages serves 404.html for unknown paths.
    _status, _ct, payload = router.handle("GET", "/does-not-exist", {}, {}, None)
    write("404.html", payload)

    write(
        "build.json",
        json.dumps(report.as_dict(), indent=2, default=str).encode("utf-8"),
    )
    return report
