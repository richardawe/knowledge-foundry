"""Run the Requstory pilot: one company knowledge area per company, compared line by line.

    python integrations/requstory/research.py integrations/requstory/pilot.yaml \
        companies/_requstory companies/_pilot.json [--only slug,slug] [--skip-collect]

For each company this does what a request from Requstory would trigger:
instantiate the company template, collect sources, build, run the behaviour
suite, ask the research questions, and verify every line of Requstory's
published map against the evidence. Nothing runs on a schedule.

The output is one JSON file; ``report.py`` renders it. Company knowledge areas
and the output stay under ``companies/``, which is never committed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from foundry.manifest import Manifest
from foundry.specialist import Specialist
from foundry.storage import Store
from foundry.verify import verify_claims

ROOT = Path(__file__).resolve().parents[2]


def foundry(*args: str) -> str:
    result = subprocess.run([sys.executable, "-m", "foundry.cli", *args], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"foundry {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


# -- Requstory map -> claims ---------------------------------------------------

def map_claims(rq: dict) -> list[dict]:
    """Every checkable line of a published map, labelled with where it came from."""
    claims = []
    if rq.get("summary"):
        claims.append({"section": "Summary", "text": rq["summary"], "requstory": "from the website"})
    for band in rq.get("bands", []):
        for unit in band["units"]:
            claims.append({
                "section": band["band"], "text": unit["name"],
                "requstory": "found on the website" if unit["found_on_website"] else
                             "typical for the industry; not confirmed",
            })
    for stage in rq.get("journey", []):
        if stage.get("quote"):
            claims.append({"section": "Customer journey",
                           "text": stage["quote"].strip("“”\"' "),
                           "requstory": f"quoted at '{stage['stage']}'"})
    for finding in rq.get("findings", []):
        pages = re.search(r"Named on (\d+) pages?", finding["description"])
        claims.append({"section": "What research found", "text": finding["title"],
                       "detail": finding["description"],
                       "requstory": f"named on {pages.group(1) if pages else '?'} pages of the website"})
    return claims


# -- Requstory checks -> rule comparisons --------------------------------------

def compare_checks(rq: dict, facts: dict, answers: dict) -> list[dict]:
    """Set each published check beside what the foundry found, by rule."""
    site = facts.get("site", {})
    probes = site.get("probes", {})
    pages = site.get("page_uris", [])
    n = site.get("pages_read", 0)
    contact_pages = [u for u in pages if re.search(r"contact|talk-to|support|help", u, re.I)]
    rows = []
    for check in rq.get("checks", []):
        text = check["text"]
        low = text.lower()
        foundry_says, agrees = None, None
        if "contact page or email" in low:
            found = site.get("emails") or contact_pages
            foundry_says = (f"{len(site.get('emails', []))} published email address(es) and "
                            f"{len(contact_pages)} contact/support page(s) in {n} pages read"
                            if found else f"none found in {n} pages read")
            agrees = (check["status"] == "ok") == bool(found)
        elif "phone number" in low:
            found = site.get("phones")
            foundry_says = (f"{len(found)} telephone number(s) found, e.g. {found[0]['value']}"
                            if found else f"none found in {n} pages read")
            agrees = (check["status"] == "ok") == bool(found)
        elif "sitemap" in low:
            probe = probes.get("/sitemap.xml", {})
            foundry_says = ("sitemap present at " + probe.get("uri", "/sitemap.xml")
                            if probe.get("present") else f"not present ({probe.get('detail')})")
            agrees = (check["status"] == "ok") == bool(probe.get("present"))
        elif "llms.txt" in low:
            probe = probes.get("/llms.txt", {})
            foundry_says = (f"llms.txt present ({probe.get('detail')})"
                            if probe.get("present") else f"not present ({probe.get('detail')})")
            agrees = (check["status"] == "ok") == bool(probe.get("present"))
        elif "who it is for" in low:
            # Shown, never scored: an extractive answer can be grounded and still
            # not be about the audience, so it cannot overturn a check on its own.
            answer = answers.get("audience", {})
            foundry_says = (f"an answer was produced; read it before relying on it: "
                            f"{answer.get('answer', '')[:160]}"
                            if answer.get("status") == "answered"
                            else f"also not established ({answer.get('status')})")
        elif "complaints" in low:
            answer = answers.get("complaints", {})
            answered = answer.get("status") == "answered"
            foundry_says = "a complaints route is described" if answered else \
                f"not established ({answer.get('status')})"
            agrees = None  # an answer about complaints is not proof of a process
        rows.append({**check, "foundry": foundry_says, "agrees": agrees})
    return rows


def research_company(company: dict, rq_dir: Path, skip_collect: bool) -> dict:
    slug, domain, name = company["slug"], company["domain"], company["name"]
    started = time.monotonic()
    ka_dir = ROOT / "companies" / f"co-{domain.replace('.', '-')}"
    timing = ka_dir / "timing.json"
    if not skip_collect or not ka_dir.exists():
        foundry("instantiate", "company", "--param", f"domain={domain}",
                "--param", f"name={name}", "--force")
        foundry("collect", str(ka_dir))
        timing.write_text(json.dumps({"collect_seconds": round(time.monotonic() - started, 1)}))
    collect_seconds = json.loads(timing.read_text())["collect_seconds"] if timing.exists() else None
    built = time.monotonic()
    foundry("build", str(ka_dir), "--rebuild")
    evaluation = json.loads(foundry("eval", str(ka_dir), "--json"))

    collection = json.loads((ka_dir / "collection.json").read_text())
    identity = yaml.safe_load((ka_dir / "identity.yaml").read_text())
    bank = yaml.safe_load((ka_dir / "research.yaml").read_text())["questions"]
    rq = json.loads((rq_dir / f"{slug}.json").read_text())
    claims = map_claims(rq)

    manifest = Manifest.load(ka_dir)
    with Store(manifest.id) as store:
        specialist = Specialist(manifest, store)
        answers = {}
        for item in bank:
            t0 = time.monotonic()
            answer = specialist.ask(item["question"]).as_dict()
            answer["topic"], answer["check"], answer["tiers"] = item["id"], item["check"], item["tiers"]
            answer["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
            answers[item["id"]] = answer
        verdicts = verify_claims(store, [c["text"] for c in claims], subject=name)
        sources = [dict(row) for row in store.sources()]

    for claim, verdict in zip(claims, verdicts):
        claim.update(verdict.as_dict())

    return {
        "slug": slug, "domain": domain, "name": name,
        "requstory": {k: rq.get(k) for k in ("title", "summary", "score", "researched", "read", "url")},
        "counts": collection["counts"],
        "facts": collection["facts"],
        "skipped": collection["skipped"],
        "identity": identity["registers"],
        "sources": [{k: s[k] for k in ("id", "title", "uri", "publisher", "source_type",
                                       "authority", "licence", "published_at", "tags", "status")}
                    for s in sources],
        "evaluation": {"passed": evaluation["passed"], "pass_rate": evaluation["pass_rate"],
                       "total": evaluation["total"], "metrics": evaluation["metrics"],
                       "results": [{"id": r.get("question_id") or r.get("id"),
                                    "passed": r.get("passed"),
                                    "status": r.get("status")} for r in evaluation["results"]]},
        "answers": answers,
        "claims": claims,
        "checks": compare_checks(rq, collection["facts"], answers),
        "collect_seconds": collect_seconds,
        "seconds": round(time.monotonic() - built, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot")
    parser.add_argument("requstory_dir")
    parser.add_argument("out")
    parser.add_argument("--only", default="")
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()

    pilot = yaml.safe_load(Path(args.pilot).read_text())
    only = {s for s in args.only.split(",") if s}
    out = Path(args.out)
    results = json.loads(out.read_text())["companies"] if out.exists() else {}
    for company in pilot["companies"]:
        if only and company["slug"] not in only:
            continue
        print(f"== {company['name']} ({company['domain']})", flush=True)
        results[company["slug"]] = research_company(company, Path(args.requstory_dir),
                                                    args.skip_collect)
        r = results[company["slug"]]
        print(f"   {sum(r['counts'].values())} sources {r['counts']}, "
              f"{len(r['claims'])} map lines checked, {r['seconds']}s", flush=True)
        out.write_text(json.dumps({"industry": pilot["industry"], "companies": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
