"""Render the pilot JSON as one self-contained HTML page.

    python integrations/requstory/report.py companies/_pilot.json companies/pilot-report.html
"""

from __future__ import annotations

import json
from datetime import date
from html import escape
import re
import sys
from pathlib import Path

ROLE_WORDS = re.compile(r"^(sales|support|help|contact|info|hello|hi|privacy|security|press|media|"
                        r"careers|jobs|legal|billing|enquiries|inquiries|dpo|gdpr|abuse|team|"
                        r"customer|service|partners|investors|ir|accounts)", re.I)


PUBLIC_TYPES = {"official_record", "self_published"}
WITHHELD = "Shown to the company once it has claimed its page (rests on press or user sources)."


def public_view(claim: dict) -> dict:
    """What anyone may see of a checked line: the same rule as the TypeScript viewOf."""
    support = [s for s in claim.get("support", []) if s["source_type"] in PUBLIC_TYPES]
    verdict = claim["verdict"]
    if verdict == "corroborated" and not any(s["source_type"] != "self_published" for s in support):
        verdict = "self_reported" if support else "not_found"
    return {**claim, "verdict": verdict, "support": support,
            "withheld": verdict != claim["verdict"],
            "note": claim.get("note", "") if verdict == claim["verdict"] else
            "independent support exists; shown to the company once it has claimed its page"}


def claim_row(c: dict) -> dict:
    return {"section": c["section"], "text": c["text"], "requstory": c["requstory"],
            "verdict": c["verdict"], "note": c.get("note", ""), "withheld": c.get("withheld", False),
            "support": [{"type": s["source_type"], "title": s["title"], "uri": s["uri"],
                         "quote": s["quote"]} for s in c.get("support", [])[:2]]}


def slim(company: dict, public: bool = False) -> dict:
    site = company["facts"].get("site", {})
    def answer(a: dict) -> dict:
        cited = [s for s in a.get("sources", []) if s.get("cited")]
        if public and any(s["source_type"] not in PUBLIC_TYPES for s in cited):
            return {"topic": a["topic"], "question": a["question"], "status": "withheld",
                    "answer": WITHHELD, "types": [], "source": None, "ms": 0}
        return {
            "topic": a["topic"], "question": a["question"], "status": a["status"],
            "answer": re.sub(r"\s+", " ", a["answer"])[:420],
            "types": sorted({s["source_type"] for s in cited}),
            "source": ({"title": cited[0].get("title") or cited[0].get("citation", ""),
                        "uri": cited[0].get("uri", ""), "type": cited[0]["source_type"]}
                       if cited else None),
            "ms": a.get("elapsed_ms", 0),
        }
    return {
        "slug": company["slug"], "name": company["name"], "domain": company["domain"],
        "requstory": company["requstory"],
        "counts": company["counts"],
        # End to end -- collection dominates; a rebuild alone takes a few seconds.
        "seconds": company.get("collect_seconds") or company["seconds"],
        "site": {
            "pages": site.get("pages_read", 0),
            "domain": site.get("domain"),
            "redirected_from": site.get("redirected_from"),
            "emails": [e for e in site.get("emails", []) if ROLE_WORDS.match(e["value"])][:4],
            "phones": site.get("phones", [])[:3],
            "probes": site.get("probes", {}),
        },
        "declared": company["facts"].get("declared_names", []),
        "identity": {
            k: {"register": v.get("register", k), "status": v["status"],
                "reason": v.get("reason", ""),
                "entities": [{"name": e["name"], "key": e["key"], "corroborated": e["corroborated"],
                              "via": e.get("via", "")} for e in v.get("entities", [])]}
            for k, v in company["identity"].items()
        },
        "evaluation": {"pass_rate": company["evaluation"]["pass_rate"],
                       "total": company["evaluation"]["total"],
                       "failed": [r["id"] for r in company["evaluation"]["results"] if not r["passed"]]},
        "checks": [{"status": c["status"], "text": c["text"], "foundry": c["foundry"],
                    "agrees": c["agrees"]} for c in company["checks"]],
        "claims": [claim_row(public_view(c) if public else c) for c in company["claims"]],
        "answers": [answer(a) for a in company["answers"].values()],
        "licences": sorted({s["licence"] for s in company["sources"]}),
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Company Foundry Pilot</title>
<style>
:root {
  --bg: #f6f5f1; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #75746f;
  --line: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --chip: #efeee9;
  --v-corroborated: #0ca30c; --v-self: #2a78d6; --v-label: #eda100; --v-none: #6250d6;
  --t-official: #2a78d6; --t-self: #eb6834; --t-press: #1baf7a; --t-community: #eda100;
  --good-ink: #006300; --bad-ink: #a32020; --warn-ink: #7a5200;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --bg: #121211; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #9a998f;
    --line: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --chip: #242422;
    --v-corroborated: #0ca30c; --v-self: #3987e5; --v-label: #c98500; --v-none: #9085e9;
    --t-official: #3987e5; --t-self: #d95926; --t-press: #199e70; --t-community: #c98500;
    --good-ink: #3fcf3f; --bad-ink: #ff8a8a; --warn-ink: #f2c25c;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #121211; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #9a998f;
  --line: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --chip: #242422;
  --v-corroborated: #0ca30c; --v-self: #3987e5; --v-label: #c98500; --v-none: #9085e9;
  --t-official: #3987e5; --t-self: #d95926; --t-press: #199e70; --t-community: #c98500;
  --good-ink: #3fcf3f; --bad-ink: #ff8a8a; --warn-ink: #f2c25c;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; font-variant-numeric: tabular-nums; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 80px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 44px 0 6px; }
h3 { font-size: 16px; margin: 22px 0 8px; }
p.lede { color: var(--ink-2); max-width: 760px; margin: 0 0 4px; }
.muted { color: var(--ink-3); }
.small { font-size: 13px; }
a { color: inherit; text-decoration-color: var(--axis); text-underline-offset: 2px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 18px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 24px; }
.tile .k { font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--ink-3); }
.tile .v { font-size: 30px; font-weight: 650; line-height: 1.15; margin: 4px 0 2px; }
.tile .d { font-size: 13px; color: var(--ink-2); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--ink-2); margin: 6px 0 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.bars { display: grid; gap: 8px; }
.bar-row { display: grid; grid-template-columns: 150px 1fr 44px; gap: 10px; align-items: center; }
.bar-row .lbl { font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-row .tot { font-size: 13px; color: var(--ink-3); text-align: right; }
.track { display: flex; gap: 2px; height: 18px; }
.seg { height: 100%; border-radius: 4px; min-width: 3px; position: relative; cursor: default; }
.seg:hover { outline: 2px solid var(--ink); outline-offset: 1px; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th { text-align: left; font-weight: 600; color: var(--ink-2); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
th, td { padding: 8px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
.scroll { overflow-x: auto; }
.pill { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; font-size: 12.5px;
  padding: 1px 8px 1px 6px; border-radius: 999px; background: var(--chip); }
.pill .sw { width: 8px; height: 8px; border-radius: 50%; }
.ok { color: var(--good-ink); } .bad { color: var(--bad-ink); } .warn { color: var(--warn-ink); }
details.co { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; margin: 10px 0; }
details.co > summary { cursor: pointer; padding: 14px 18px; list-style: none; display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline; }
details.co > summary::-webkit-details-marker { display: none; }
details.co > summary b { font-size: 16px; }
details.co[open] > summary { border-bottom: 1px solid var(--line); }
.co-body { padding: 4px 18px 18px; }
.quote { color: var(--ink-2); font-size: 12.5px; margin-top: 3px; }
.tabs { display: flex; gap: 6px; margin: 4px 0 10px; flex-wrap: wrap; }
.tabs button { font: inherit; font-size: 13px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--ring);
  background: var(--surface); color: var(--ink-2); cursor: pointer; }
.tabs button[aria-pressed="true"] { background: var(--ink); color: var(--surface); border-color: var(--ink); }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
ul.tight { margin: 6px 0; padding-left: 18px; } ul.tight li { margin: 3px 0; }
.flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; counter-reset: s; }
.flow div { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 10px 12px; font-size: 13px; }
.flow div::before { counter-increment: s; content: counter(s); font-weight: 700; margin-right: 6px; color: var(--ink-3); }
@media (max-width: 720px) {
  .grid2 { grid-template-columns: 1fr; }
  .bar-row { grid-template-columns: 96px 1fr 34px; }
  h1 { font-size: 23px; }
}
</style>
</head>
<body>
<main>
__BODY__
</main>
</body>
</html>
"""


# Rendered in Python, not in the browser: file viewers that do not run scripts
# would otherwise show an empty page.

VERDICTS = [
    ("corroborated", "Corroborated", "--v-corroborated", "✓"),
    ("self_reported", "Company's own site only", "--v-self", "◐"),
    ("label_only", "Only as a label or heading", "--v-label", "▭"),
    ("not_found", "Not found", "--v-none", "○"),
]
TIERS = [
    ("official_record", "Official records", "--t-official", ""),
    ("self_published", "Company website", "--t-self", ""),
    ("press", "Press", "--t-press", ""),
    ("community", "Users (Hacker News)", "--t-community", ""),
]
TYPE_NAME = {t[0]: t[1] for t in TIERS}
VERDICT = {v[0]: v for v in VERDICTS}
STATUS = {
    "answered": ("ok", "Answered"),
    "insufficient_evidence": ("muted", "Declined: not enough evidence"),
    "ambiguous": ("muted", "Declined: under-specified"),
    "out_of_scope": ("muted", "Declined: out of scope"),
    "unsupported": ("warn", "Withheld: failed grounding"),
    "irrelevant": ("warn", "Withheld: off-topic"),
    "withheld": ("muted", "Not shown publicly"),
}
REGISTERS = [("uk-register", "Companies House (UK)"), ("lei-register", "GLEIF (global LEI)"),
             ("us-filings", "SEC EDGAR (US)")]
VIEW_NOTE = ("<p class=\"card small\"><b>Public view.</b> Register records and each company's own site are shown in "
             "full. Evidence from the press and from user discussion is withheld, as Requstory will withhold it "
             "from everyone except the company, once it has claimed its page; verdicts that rested on it are shown "
             "at the level the public evidence supports.</p>")
FLOW = [
    "A company maps its site on Requstory and asks for a knowledge foundry",
    "Requstory calls the foundry with domain + confirmed trading name",
    "Instantiate the company template, collect (~40 s), build",
    "Verify every map line; answer the research questions",
    "Company sees held identity matches and confirms or rejects them",
    "Map lines turn green only when evidence supports them; third-party findings shown to the claimant",
]


def e(value) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def link(uri: str, text: str) -> str:
    return f'<a href="{e(uri)}" target="_blank" rel="noopener noreferrer">{e(text)}</a>' if uri else e(text)


def pct(a: int, b: int) -> int:
    return round(100 * a / b) if b else 0


def pill(verdict: str) -> str:
    _, label, var, icon = VERDICT[verdict]
    return f'<span class="pill"><span class="sw" style="background:var({var})"></span>{icon} {e(label)}</span>'


def legend(keys) -> str:
    return '<div class="legend">' + "".join(
        f'<span><span class="sw" style="background:var({k[2]})"></span>{(k[3] + " ") if k[3] else ""}{e(k[1])}</span>'
        for k in keys) + "</div>"


def bars(companies: list[dict], keys, values_of) -> str:
    rows = [(c, values_of(c)) for c in companies]
    top = max(sum(v) for _, v in rows) or 1
    out = []
    for c, values in rows:
        total = sum(values)
        segs = "".join(
            f'<div class="seg" title="{e(c["name"])} · {e(keys[i][1])}: {v} of {total} ({pct(v, total)}%)" '
            f'style="width:{100 * v / top:.2f}%;background:var({keys[i][2]})"></div>'
            for i, v in enumerate(values) if v)
        out.append(f'<div class="bar-row"><div class="lbl">{e(c["name"])}</div>'
                   f'<div class="track">{segs}</div><div class="tot">{total}</div></div>')
    return legend(keys) + '<div class="bars">' + "".join(out) + "</div>"


def identity_cell(record: dict | None) -> str:
    if not record:
        return "–"
    if record["entities"]:
        parts = []
        for ent in record["entities"]:
            mark = '<span class="ok">✓</span>' if ent["corroborated"] else '<span class="warn">⏸</span>'
            note = " · declared on site" if ent["via"] == "declared on the site" else ""
            note += "" if ent["corroborated"] else " · held: name match only"
            parts.append(f'{mark} {e(ent["name"])} <span class="muted small">{e(ent["key"])}{note}</span>')
        return "<br>".join(parts)
    return {
        "unresolved": '<span class="muted">no match</span>',
        "failed": '<span class="bad">register unreachable</span>',
        "not_run": '<span class="muted">not run: needs a contact email</span>',
    }.get(record["status"], e(record["status"]))


def company_section(c: dict) -> str:
    counts = [(v, sum(1 for x in c["claims"] if x["verdict"] == v[0])) for v in VERDICTS]
    probes = c["site"]["probes"]

    def probe(key: str) -> str:
        if key not in probes:
            return ""
        name = e(key.lstrip("/"))
        return f'<span class="ok">✓</span> {name}' if probes[key]["present"] else f'<span class="muted">✗ {name}</span>'

    emails = c["site"]["emails"]
    phones = c["site"]["phones"]
    facts = [
        ("Contact addresses: " + ", ".join(link(x["found_on"], x["value"]) for x in emails))
        if emails else '<span class="muted">No role email address on the pages read</span>',
        ("Telephone: " + ", ".join(link(x["found_on"], x["value"]) for x in phones))
        if phones else '<span class="muted">No telephone number on the pages read</span>',
        " &nbsp; ".join(probe(k) for k in ("/robots.txt", "/sitemap.xml", "/llms.txt")),
    ]
    if c["declared"]:
        facts.append("Site declares: " + ", ".join(e(d["name"]) for d in c["declared"]))

    claim_rows = []
    for x in c["claims"]:
        support = "".join(
            f'<div class="quote">{e(TYPE_NAME.get(s["type"], s["type"]))}: {link(s["uri"], s["title"])}'
            f'<br>“{e(s["quote"])}”</div>' for s in x["support"])
        note = f' <span class="muted small">{e(x["note"])}</span>' if x["note"] else ""
        claim_rows.append(
            f'<tr><td><span class="muted small">{e(x["section"])}</span><br>{e(x["text"])}</td>'
            f'<td class="small">{e(x["requstory"])}</td><td>{pill(x["verdict"])}{note}{support}</td></tr>')

    answer_rows = []
    for a in c["answers"]:
        cls, label = STATUS.get(a["status"], ("", a["status"]))
        cites = (f' <span class="muted small">· cites '
                 f'{", ".join(e(TYPE_NAME.get(t, t)) for t in a["types"])}</span>') if a["types"] else ""
        source = (f'<div class="quote">{link(a["source"]["uri"], a["source"]["title"])}</div>'
                  if a["source"] else "")
        answer_rows.append(
            f'<tr><td class="small">{e(a["question"])}</td><td><span class="{cls} small">{label}</span>'
            f'{cites}<div class="small">{e(a["answer"])}</div>{source}</td></tr>')

    ev = c["evaluation"]
    failed = f' · failed: {", ".join(e(f) for f in ev["failed"])}' if ev["failed"] else ""
    score = f' · Requstory score {e(c["requstory"]["score"])}/100' if c["requstory"].get("score") else ""
    return f"""<details class="co"><summary><b>{e(c["name"])}</b>
<span class="muted small">{e(c["domain"])} · {sum(c["counts"].values())} sources · {c["site"]["pages"]} pages read · {c["seconds"]}s{score}</span>
<span class="small">{" &nbsp; ".join(f"{v[3]} {n}" for v, n in counts if n)}</span></summary>
<div class="co-body">
<div class="grid2">
<div><h3>Site facts, by rule</h3><ul class="tight small">{"".join(f"<li>{f}</li>" for f in facts)}</ul></div>
<div><h3>Behaviour suite</h3><p class="small">{round(ev["pass_rate"] * ev["total"])}/{ev["total"]} passed{failed}</p>
<p class="muted small">Same six questions for every company: citations resolve, claims grounded, and three questions that must be declined.</p></div>
</div>
<h3>Requstory's map, checked</h3>
<div class="scroll"><table><thead><tr><th>Map line</th><th>Requstory</th><th>Foundry verdict and evidence</th></tr></thead>
<tbody>{"".join(claim_rows)}</tbody></table></div>
<h3>Research questions</h3>
<div class="scroll"><table><thead><tr><th style="width:28%">Question</th><th>Foundry answer</th></tr></thead>
<tbody>{"".join(answer_rows)}</tbody></table></div>
<p class="muted small">Licences in this corpus: {" · ".join(e(x) for x in c["licences"])}</p>
</div></details>"""


def disagreement_row(name: str, x: dict) -> str:
    mark = {"ok": "✓", "missing": "✗"}.get(x["status"], "?")
    cls = "ok" if x["status"] == "ok" else "bad"
    aside = ' <span class="warn small">(Requstory looked more widely here)</span>' if x["status"] == "ok" else ""
    return (f'<tr><td>{e(name)}</td><td><span class="{cls}">{mark}</span> {e(x["text"])}</td>'
            f'<td>{e(x["foundry"])}{aside}</td></tr>')


def identity_row(c: dict) -> str:
    moved = c["site"]["redirected_from"]
    where = e(c["site"]["domain"]) + (f" (from {e(moved)})" if moved else "")
    cells = "".join(f"<td>{identity_cell(c['identity'].get(key))}</td>" for key, _ in REGISTERS)
    return f'<tr><td><b>{e(c["name"])}</b><br><span class="muted small">{where}</span></td>{cells}</tr>'


def render(companies: list[dict], industry: str, generated: str, limits: str,
           public: bool = False) -> str:
    claims = [x for c in companies for x in c["claims"]]
    count = lambda verdict: sum(1 for x in claims if x["verdict"] == verdict)  # noqa: E731
    comparable = [x for c in companies for x in c["checks"] if x["agrees"] is not None]
    gaps = [x for x in comparable if x["status"] != "ok"]
    overturned = [x for x in gaps if x["agrees"] is False]
    confirmed = sum(1 for c in companies
                    if any(ent["corroborated"] for r in c["identity"].values() for ent in r["entities"]))
    sources = sum(sum(c["counts"].values()) for c in companies)
    seconds = round(sum(c["seconds"] for c in companies) / len(companies))
    tiles = [
        ("Companies", len(companies), f'{sum(1 for c in companies if c["checks"])} with scored Requstory checks'),
        ("Sources collected", sources, f"{round(sources / len(companies))} per company, 4 kinds"),
        ("Legal entity confirmed", f"{confirmed}/{len(companies)}",
         "by a register the company's own site corroborates"),
        ("Map lines checked", len(claims),
         (f'{sum(1 for x in claims if x["withheld"])} corroborated by press or users; that evidence '
          'is shown to the company once claimed' if public else
          f'{count("corroborated")} corroborated by a sentence naming the company; all hold up when read')),
        ("Requstory gaps overturned", f"{len(overturned)}/{len(gaps)}",
         'rule-checkable "not found" checks where the evidence exists'),
        ("Time and cost", f"{seconds} s",
         "per company end to end, almost all of it polite crawling; no model calls"),
    ]
    disagreements = [(c["name"], x) for c in companies for x in c["checks"] if x["agrees"] is False]
    dis_rows = "".join(disagreement_row(name, x) for name, x in disagreements)
    id_rows = "".join(identity_row(c) for c in companies)
    return f"""
<h1>A knowledge foundry per company: 10-company pilot</h1>
<p class="lede">Ten Software &amp; SaaS companies from Requstory's directory. For each one, a company knowledge area was
created on request, filled from four kinds of source, and every line of Requstory's published map was checked against it.
No model was called: everything here is register fields, rules, retrieval and extraction.</p>
<p class="muted small">Run {e(generated)} · {e(industry)} · map snapshots from requstory.com</p>
{VIEW_NOTE if public else ""}
<section class="tiles">{"".join(f'<div class="card tile"><div class="k">{e(k)}</div><div class="v">{e(v)}</div><div class="d">{e(d)}</div></div>' for k, v, d in tiles)}</section>

<h2>Requstory's map, line by line</h2>
<p class="muted small">Every team, product, journey quote and research finding on each company's Requstory page,
checked against everything collected. "Corroborated" means a source other than the company's own site supports it.</p>
<div class="card">{bars(companies, VERDICTS, lambda c: [sum(1 for x in c["claims"] if x["verdict"] == v[0]) for v in VERDICTS])}</div>

<h2>Where the evidence comes from</h2>
<p class="muted small">Sources collected per company, by who is speaking. Register records appear only when the
company's own site confirms the match; name-only matches are held back (see identity, below).</p>
<div class="card">{bars(companies, TIERS, lambda c: [c["counts"].get(t[0], 0) for t in TIERS])}</div>

<h2>Requstory checks the foundry overturns</h2>
<p class="muted small">Requstory scores from 3–6 pages; the foundry reads up to 25 plus robots.txt, sitemaps and
llms.txt. These are checks where both made a call and the calls differ.</p>
<div class="card scroll"><table><thead><tr><th>Company</th><th>Requstory says</th><th>The foundry found</th></tr></thead>
<tbody>{dis_rows}</tbody></table></div>

<h2>Who is the company, legally?</h2>
<p class="muted small">Registers are searched by trading name and by any legal name the company's own site declares
("registered as …"). A name match is only used when the site confirms it.</p>
<div class="card scroll"><table><thead><tr><th>Company</th>{"".join(f"<th>{e(n)}</th>" for _, n in REGISTERS)}</tr></thead>
<tbody>{id_rows}</tbody></table></div>

<h2>Each company</h2>
<p class="muted small">Open a company for every map line with its evidence, and the 16 research questions with the
foundry's answer, or its refusal.</p>
{"".join(company_section(c) for c in companies)}

<h2>What did not work, plainly</h2>
<div class="card">{limits}</div>

<h2>How it would plug into Requstory</h2>
<div class="flow">{"".join(f"<div>{e(s)}</div>" for s in FLOW)}</div>
<p class="muted small" style="margin-top:10px">Third-party findings go only to the company, once it has claimed its page.
Company knowledge areas are never committed to the repository.</p>
"""


def main() -> int:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    # --public: the view anyone may see. Press and user evidence is withheld,
    # as Requstory will withhold it from everyone but the claimed company.
    public = "--public" in sys.argv[3:]
    data = json.loads(source.read_text())
    companies = [slim(v, public) for v in data["companies"].values()]
    limits = Path(__file__).with_name("limits.html").read_text(encoding="utf-8")
    body = render(companies, data["industry"], date.today().isoformat(), limits, public)
    target.write_text(PAGE.replace("__BODY__", body), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
