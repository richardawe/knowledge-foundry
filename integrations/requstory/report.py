"""Render the pilot JSON as one self-contained HTML page.

    python integrations/requstory/report.py companies/_pilot.json companies/pilot-report.html
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROLE_WORDS = re.compile(r"^(sales|support|help|contact|info|hello|hi|privacy|security|press|media|"
                        r"careers|jobs|legal|billing|enquiries|inquiries|dpo|gdpr|abuse|team|"
                        r"customer|service|partners|investors|ir|accounts)", re.I)


def slim(company: dict) -> dict:
    site = company["facts"].get("site", {})
    def answer(a: dict) -> dict:
        cited = [s for s in a.get("sources", []) if s.get("cited")]
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
        "claims": [{"section": c["section"], "text": c["text"], "requstory": c["requstory"],
                    "verdict": c["verdict"], "note": c.get("note", ""),
                    "support": [{"type": s["source_type"], "title": s["title"], "uri": s["uri"],
                                 "quote": s["quote"]} for s in c.get("support", [])[:2]]}
                   for c in company["claims"]],
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
#tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--surface);
  font-size: 12px; padding: 6px 8px; border-radius: 6px; opacity: 0; transition: opacity .08s; z-index: 9; max-width: 260px; }
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
  <h1>A knowledge foundry per company: 10-company pilot</h1>
  <p class="lede">Ten Software &amp; SaaS companies from Requstory's directory. For each one, a company knowledge area was
  created on request, filled from four kinds of source, and every line of Requstory's published map was checked against it.
  No model was called: everything here is retrieval, rules and extraction.</p>
  <p class="muted small" id="when"></p>

  <section class="tiles" id="tiles"></section>

  <h2>Requstory's map, line by line</h2>
  <p class="muted small">Every team, product, journey quote and research finding on each company's Requstory page,
  checked against everything collected. "Corroborated" means a source other than the company's own site supports it.</p>
  <div class="card">
    <div class="legend" id="verdict-legend"></div>
    <div class="bars" id="verdict-bars"></div>
  </div>

  <h2>Where the evidence comes from</h2>
  <p class="muted small">Sources collected per company, by who is speaking. Register records appear only when the
  company's own site confirms the match; name-only matches are held back (see identity, below).</p>
  <div class="card">
    <div class="legend" id="tier-legend"></div>
    <div class="bars" id="tier-bars"></div>
  </div>

  <h2>Requstory checks the foundry overturns</h2>
  <p class="muted small">Requstory scores from 3–6 pages; the foundry reads up to 25 plus robots.txt, sitemaps and
  llms.txt. These are checks where both made a call and the calls differ.</p>
  <div class="card scroll"><table id="disagree"></table></div>

  <h2>Who is the company, legally?</h2>
  <p class="muted small">Registers are searched by trading name and by any legal name the company's own site declares
  ("registered as …"). A name match is only used when the site confirms it.</p>
  <div class="card scroll"><table id="identity"></table></div>

  <h2>Each company</h2>
  <p class="muted small">Open a company for every map line with its evidence, and the 16 research questions with the
  foundry's answer, or its refusal.</p>
  <div id="companies"></div>

  <h2>What did not work, plainly</h2>
  <div class="card" id="limits"></div>

  <h2>How it would plug into Requstory</h2>
  <div class="flow" id="flow"></div>
  <p class="muted small" style="margin-top:10px">Third-party findings go only to the company, once it has claimed its page.
  Nothing here is published, and company knowledge areas are never committed to the repository.</p>
</main>
<div id="tip" role="tooltip"></div>
<script>
const DATA = __DATA__;
const C = Object.values(DATA.companies);
const VERDICTS = [
  ["corroborated", "Corroborated", "--v-corroborated", "✓"],
  ["self_reported", "Company's own site only", "--v-self", "◐"],
  ["label_only", "Only as a label or heading", "--v-label", "▭"],
  ["not_found", "Not found", "--v-none", "○"],
];
const TIERS = [
  ["official_record", "Official records", "--t-official"],
  ["self_published", "Company website", "--t-self"],
  ["press", "Press", "--t-press"],
  ["community", "Users (Hacker News)", "--t-community"],
];
const VNAME = Object.fromEntries(VERDICTS.map(v => [v[0], v]));
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pill = v => { const x = VNAME[v]; return `<span class="pill"><span class="sw" style="background:var(${x[2]})"></span>${x[3]} ${x[1]}</span>`; };
const sum = a => a.reduce((x, y) => x + y, 0);
const pct = (a, b) => b ? Math.round(100 * a / b) : 0;
const link = (uri, text) => uri ? `<a href="${esc(uri)}" target="_blank" rel="noopener noreferrer">${esc(text)}</a>` : esc(text);

document.getElementById("when").textContent = `Run ${DATA.generated} · ${DATA.industry} · map snapshots from requstory.com`;

// ---- tiles
const claims = C.flatMap(c => c.claims);
const vc = k => claims.filter(x => x.verdict === k).length;
const comparable = C.flatMap(c => c.checks).filter(x => x.agrees !== null);
const overturned = comparable.filter(x => x.agrees === false && x.status !== "ok");
const confirmedCos = C.filter(c => Object.values(c.identity).some(r => r.entities.some(e => e.corroborated))).length;
const sources = sum(C.map(c => sum(Object.values(c.counts))));
const secs = C.map(c => c.seconds);
const tiles = [
  ["Companies", C.length, `${C.filter(c => c.checks.length).length} with scored Requstory checks`],
  ["Sources collected", sources, `${Math.round(sources / C.length)} per company, 4 kinds`],
  ["Legal entity confirmed", `${confirmedCos}/${C.length}`, "by a register the company's own site corroborates"],
  ["Map lines checked", claims.length, `${pct(vc("corroborated"), claims.length)}% marked corroborated, under half of which hold up when read (see below)`],
  ["Requstory gaps overturned", `${overturned.length}/${comparable.filter(x => x.status !== "ok").length}`, "rule-checkable \"not found\" checks where the evidence exists"],
  ["Time and cost", `${Math.round(sum(secs) / secs.length)} s`, "per company end to end, almost all of it polite crawling; no model calls"],
];
document.getElementById("tiles").innerHTML = tiles.map(t =>
  `<div class="card tile"><div class="k">${t[0]}</div><div class="v">${t[1]}</div><div class="d">${t[2]}</div></div>`).join("");

// ---- stacked bars
const tip = document.getElementById("tip");
function bars(el, legendEl, keys, rowsOf) {
  legendEl.innerHTML = keys.map(k => `<span><span class="sw" style="background:var(${k[2]})"></span>${k[3] ? k[3] + " " : ""}${k[1]}</span>`).join("");
  const rows = C.map(c => ({ c, vals: rowsOf(c) }));
  const max = Math.max(...rows.map(r => sum(r.vals)));
  el.innerHTML = rows.map(r => {
    const total = sum(r.vals);
    const segs = r.vals.map((v, i) => v ? `<div class="seg" tabindex="0" style="width:${100 * v / max}%;background:var(${keys[i][2]})" data-tip="${esc(r.c.name)} · ${esc(keys[i][1])}: ${v} of ${total} (${pct(v, total)}%)"></div>` : "").join("");
    return `<div class="bar-row"><div class="lbl" title="${esc(r.c.name)}">${esc(r.c.name)}</div><div class="track">${segs}</div><div class="tot">${total}</div></div>`;
  }).join("");
  el.querySelectorAll(".seg").forEach(s => {
    const show = e => { tip.textContent = s.dataset.tip; tip.style.opacity = 1;
      const r = s.getBoundingClientRect(); const x = e.clientX ?? r.left + r.width / 2; const y = e.clientY ?? r.top;
      tip.style.left = Math.min(x + 12, innerWidth - 270) + "px"; tip.style.top = (y + 14) + "px"; };
    s.addEventListener("mousemove", show); s.addEventListener("focus", show);
    s.addEventListener("mouseleave", () => tip.style.opacity = 0); s.addEventListener("blur", () => tip.style.opacity = 0);
  });
}
bars(document.getElementById("verdict-bars"), document.getElementById("verdict-legend"), VERDICTS,
     c => VERDICTS.map(v => c.claims.filter(x => x.verdict === v[0]).length));
bars(document.getElementById("tier-bars"), document.getElementById("tier-legend"), TIERS,
     c => TIERS.map(t => c.counts[t[0]] || 0));

// ---- disagreements
const dis = C.flatMap(c => c.checks.filter(x => x.agrees === false).map(x => ({ ...x, co: c.name })));
document.getElementById("disagree").innerHTML = `<thead><tr><th>Company</th><th>Requstory says</th><th>The foundry found</th></tr></thead><tbody>` +
  dis.map(d => `<tr><td>${esc(d.co)}</td><td><span class="${d.status === "ok" ? "ok" : "bad"}">${d.status === "ok" ? "✓" : d.status === "missing" ? "✗" : "?"}</span> ${esc(d.text)}</td><td>${esc(d.foundry)}${d.status === "ok" ? ' <span class="warn small">(Requstory looked more widely here)</span>' : ""}</td></tr>`).join("") + "</tbody>";

// ---- identity
const REG = { "uk-register": "Companies House", "lei-register": "GLEIF (LEI)", "us-filings": "SEC EDGAR" };
const idCell = r => {
  if (!r) return "–";
  const ents = r.entities.map(e => `${e.corroborated ? '<span class="ok">✓</span>' : '<span class="warn">⏸</span>'} ${esc(e.name)} <span class="muted small">${esc(e.key)}${e.via === "declared on the site" ? " · declared on site" : ""}${e.corroborated ? "" : " · held: name match only"}</span>`).join("<br>");
  const label = { resolved: "", pending_confirmation: "", unresolved: '<span class="muted">no match</span>', failed: '<span class="bad">register unreachable</span>', not_run: '<span class="muted">not run: needs a contact email</span>' }[r.status] ?? r.status;
  return ents || label;
};
document.getElementById("identity").innerHTML = `<thead><tr><th>Company</th><th>Companies House (UK)</th><th>GLEIF (global LEI)</th><th>SEC EDGAR (US)</th></tr></thead><tbody>` +
  C.map(c => `<tr><td><b>${esc(c.name)}</b><br><span class="muted small">${esc(c.site.domain)}${c.site.redirected_from ? ` (from ${esc(c.site.redirected_from)})` : ""}</span></td>${["uk-register", "lei-register", "us-filings"].map(k => `<td>${idCell(c.identity[k])}</td>`).join("")}</tr>`).join("") + "</tbody>";

// ---- companies
const STATUS = { answered: ["ok", "Answered"], insufficient_evidence: ["muted", "Declined: not enough evidence"],
  ambiguous: ["muted", "Declined: under-specified"], out_of_scope: ["muted", "Declined: out of scope"],
  unsupported: ["warn", "Withheld: failed grounding"], irrelevant: ["warn", "Withheld: off-topic"] };
const TYPE = Object.fromEntries(TIERS.map(t => [t[0], t[1]]));
document.getElementById("companies").innerHTML = C.map(c => {
  const counts = VERDICTS.map(v => [v, c.claims.filter(x => x.verdict === v[0]).length]).filter(x => x[1]);
  const p = c.site.probes;
  const probe = k => p[k] ? (p[k].present ? `<span class="ok">✓</span> ${esc(k.slice(1))}` : `<span class="muted">✗ ${esc(k.slice(1))}</span>`) : "";
  return `<details class="co"><summary><b>${esc(c.name)}</b><span class="muted small">${esc(c.domain)} · ${sum(Object.values(c.counts))} sources · ${c.site.pages} pages read · ${c.seconds}s${c.requstory.score ? ` · Requstory score ${c.requstory.score}/100` : ""}</span>
    <span class="small">${counts.map(([v, n]) => `${v[3]} ${n}`).join(" &nbsp; ")}</span></summary>
    <div class="co-body">
      <div class="grid2">
        <div><h3>Site facts, by rule</h3><ul class="tight small">
          <li>${c.site.emails.length ? "Contact addresses: " + c.site.emails.map(e => link(e.found_on, e.value)).join(", ") : '<span class="muted">No role email address on the pages read</span>'}</li>
          <li>${c.site.phones.length ? "Telephone: " + c.site.phones.map(e => link(e.found_on, e.value)).join(", ") : '<span class="muted">No telephone number on the pages read</span>'}</li>
          <li>${["/robots.txt", "/sitemap.xml", "/llms.txt"].map(probe).join(" &nbsp; ")}</li>
          ${c.declared.length ? `<li>Site declares: ${c.declared.map(d => esc(d.name)).join(", ")}</li>` : ""}
        </ul></div>
        <div><h3>Behaviour suite</h3><p class="small">${Math.round(c.evaluation.pass_rate * c.evaluation.total)}/${c.evaluation.total} passed${c.evaluation.failed.length ? ` · failed: ${c.evaluation.failed.map(esc).join(", ")}` : ""}</p>
          <p class="muted small">Same six questions for every company: citations resolve, claims grounded, and three questions that must be declined.</p></div>
      </div>
      <h3>Requstory's map, checked</h3>
      <div class="scroll"><table><thead><tr><th>Map line</th><th>Requstory</th><th>Foundry verdict and evidence</th></tr></thead><tbody>
      ${c.claims.map(x => `<tr><td><span class="muted small">${esc(x.section)}</span><br>${esc(x.text)}</td><td class="small">${esc(x.requstory)}</td>
        <td>${pill(x.verdict)}${x.note ? ` <span class="muted small">${esc(x.note)}</span>` : ""}${x.support.map(s => `<div class="quote">${esc(TYPE[s.type] || s.type)}: ${link(s.uri, s.title)}<br>“${esc(s.quote)}”</div>`).join("")}</td></tr>`).join("")}
      </tbody></table></div>
      <h3>Research questions</h3>
      <div class="scroll"><table><thead><tr><th style="width:28%">Question</th><th>Foundry answer</th></tr></thead><tbody>
      ${c.answers.map(a => { const s = STATUS[a.status] || ["", a.status]; return `<tr><td class="small">${esc(a.question)}</td>
        <td><span class="${s[0]} small">${s[1]}</span>${a.types.length ? ` <span class="muted small">· cites ${a.types.map(t => esc(TYPE[t] || t)).join(", ")}</span>` : ""}<div class="small">${esc(a.answer)}</div>
        ${a.source ? `<div class="quote">${link(a.source.uri, a.source.title)}</div>` : ""}</td></tr>`; }).join("")}
      </tbody></table></div>
      <p class="muted small">Licences in this corpus: ${c.licences.map(esc).join(" · ")}</p>
    </div></details>`;
}).join("");

document.getElementById("limits").innerHTML = DATA.limits;
document.getElementById("flow").innerHTML = [
  "A company maps its site on Requstory and asks for a knowledge foundry",
  "Requstory calls the foundry with domain + confirmed trading name",
  "Instantiate the company template, collect (~40 s), build",
  "Verify every map line; answer the research questions",
  "Company sees held identity matches and confirms or rejects them",
  "Map lines turn green only when evidence supports them; third-party findings shown to the claimant",
].map(s => `<div>${esc(s)}</div>`).join("");
</script>
</body>
</html>
"""


def main() -> int:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    data = json.loads(source.read_text())
    limits = Path(__file__).with_name("limits.html").read_text(encoding="utf-8")
    payload = {
        "industry": data["industry"],
        "generated": __import__("datetime").date.today().isoformat(),
        "companies": {k: slim(v) for k, v in data["companies"].items()},
        "limits": limits,
    }
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    target.write_text(PAGE.replace("__DATA__", blob), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
