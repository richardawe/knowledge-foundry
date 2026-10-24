# Knowledge Foundry in Requstory's research phase

Copy this folder into Requstory's repository (suggested: `research/foundry/`).
It is the Requstory side of one contract, `foundry.research/v1`; the other side
is the [knowledge-foundry](https://github.com/richardawe/knowledge-foundry)
service. Nothing here depends on how Requstory is built beyond Node 18+ for the
TypeScript client.

```
foundry/types.ts         the contract as TypeScript types
foundry/client.ts        call the foundry over HTTP (a job) or as a CLI
foundry/adapter.ts       Requstory map -> claims; result -> what each viewer may see
foundry/adapter.test.ts  the visibility rule, tested
schema/*.v1.json         JSON Schemas for request and result (source of truth)
examples/                a request and a public-view result
docker-compose.foundry.yml  run the service next to Requstory's workers
```

## Where it fits

Requstory already reads a company's site and drafts the map. The foundry
does not replace that; it checks it, and adds what a website alone cannot say.

```
company maps its site on Requstory           (unchanged)
  -> Requstory drafts the map                (unchanged)
  -> company claims the page and asks for a knowledge foundry
       Requstory sends: domain, confirmed trading name, every map line as a claim
       foundry: collect (~40 s) -> build -> answer -> verify -> result
  -> Requstory shows, per map line, a verdict and the evidence the viewer may see
  -> company confirms or rejects held register matches; the next run uses them
```

Run it **on request, never on a schedule**: when a company asks, and again when
it confirms identity matches.

## What comes back

| Field | What it is | Who may see it |
|---|---|---|
| `identity` | Per register (Companies House, GLEIF, SEC EDGAR): matched entities, and whether the company's own site confirms each | public |
| `registers` | Labelled facts read from confirmed records: legal name, number, office, incorporation, status, current officers (names and roles only) | public |
| `site` | Rule-checked facts: role contact emails, phone numbers, robots.txt, sitemap, llms.txt, pages read | public |
| `claims[]` | Each map line with a verdict and its evidence | per evidence item |
| `answers[]` | The research questions, with what produced each answer | per answer |

Verdicts, strongest first:

* `corroborated`: a source other than the company states it, in a sentence that names the company.
  Press releases (wire services, or items carrying a wire dateline) count as the company's own
  voice, not as press.
* `self_reported`: only the company's own site says it.
* `label_only`: the words appear on the site, but only as a menu item, button or heading.
* `not_found`: nothing collected supports it. It can be the foundry's blind spot (a page
  it did not read, or content built in the browser that was not supplied in `pages`), so
  don't show `not_found` to a company as "Requstory was wrong".

## The one rule that must hold

Evidence from the press, reviews and user discussion is about a named company
from somebody else. It is shown **only to that company, after it has claimed its
page**. Every evidence item carries `visibility`, and `viewOf(claim, viewer)`
applies the rule: a public viewer never receives `claimant_only` evidence, and a
verdict that rested on it is downgraded to what the public evidence supports.
Render map lines through `viewOf`, never from `claim.support` directly.

## Wiring it in

```ts
import { mapToClaims, viewOf, edgeFor, claimFlowPrompts } from "./foundry/adapter";
import { researchOverHttp } from "./foundry/client";

const result = await researchOverHttp(
  { contract: "foundry.research/v1",
    params: { domain: company.domain, name: company.confirmedName,
              confirmed: company.confirmedRegisterKeys, rejected: company.rejectedRegisterKeys },
    claims: mapToClaims(map) },
  { baseUrl: process.env.FOUNDRY_URL!, token: process.env.FOUNDRY_API_TOKEN });

for (const claim of result.claims) {
  const { verdict, support } = viewOf(claim, viewerIsClaimant ? "claimant" : "public");
  // colour the line by edgeFor(verdict); list `support` as its evidence
}
const { confirmIdentity } = claimFlowPrompts(result); // "Is this you?" on the claim screen
```

`mapToClaims` is written against the structure of Requstory's public company
pages (bands of units, journey, findings). If the internal map type differs,
change that one function; the rest keys on claim ids.

## Handing over pages Requstory has already rendered

Requstory renders sites in a browser; the foundry fetches plain HTML. For a site
whose content is built in the browser, pass the rendered pages in the request so
both are checking the same text:

```ts
pages: rendered.map((p) => ({
  url: p.url,                         // must be on the company's site
  html: p.outerHTML,                  // document.documentElement.outerHTML
  rendered_at: p.renderedAt,          // shown with the evidence
})),
```

* **Send `html`, not `innerText`.** innerText leaves out collapsed menus, which is
  exactly where product names live; in the pilot, Cvent's rendered innerText had
  none of its five product lines and its outerHTML had all five.
* Up to 150 pages, 2 MB each, 64 MB per request. Supplied URLs are not fetched
  again and do not count against the foundry's own 25-page crawl.
* Evidence from a supplied page names it: "cvent.com (rendered by the caller on
  …)". Off-site pages are skipped and listed in `sources.skipped`.
* It only changes results for sites built in the browser. Cvent's menus are in
  its plain HTML, so its verdicts were the same with or without supplied pages;
  what fixed them was keeping navigation and footers (below).

Menus and footers are read on company sites. Product names found only in a menu
come back `label_only` ("appears only as a menu item"), which is the honest
verdict: the company lists it, and says nothing more about it.

## Running the service

```
docker compose -f docker-compose.foundry.yml up -d
```

or, without Docker, from a knowledge-foundry checkout:

```
pip install ".[research]"
FOUNDRY_API_TOKEN=... FOUNDRY_CONTACT_EMAIL=... foundry serve-research --host 0.0.0.0
# or one-off: foundry research examples/request.json --out result.json
```

| Variable | Needed | Why |
|---|---|---|
| `FOUNDRY_API_TOKEN` | to listen beyond localhost | the service refuses to start without one |
| `FOUNDRY_CONTACT_EMAIL` | for SEC EDGAR | SEC refuses automated clients without a contact address; without it US filings are skipped and reported as `not_run` |
| `FOUNDRY_VERIFY_PROVIDER`, `FOUNDRY_VERIFY_MODEL` | optional | adds a model yes/no check on each candidate corroboration (`openai_compatible`, `anthropic` or `ollama`, plus that provider's key). Keyless providers are refused. |

## Cost and limits, honestly

* ~40 s per company, almost all of it polite crawling (robots.txt respected, one
  request per second per host). No model is called unless you configure one.
* **Reviews are not covered.** Trustpilot, G2, Product Hunt and Reddit refuse
  automated access; the App Store disallows it. "What users say" is Hacker News only.
  Real coverage needs a licensed review-data feed.
* **Press is Bing News RSS**, whose terms permit personal, non-commercial use only.
  Fine for a pilot. Production needs a licensed news API; the collector is one
  entry in `templates/company/collectors.yaml`.
* Corroboration is strict by design: in the pilot, 3 of 3 corroborated lines
  held up on reading, and one genuine case was missed. A configured model can
  recover some of those misses.
* Answers to open questions ("what do users say") are extracted sentences, not
  prose. Identity questions are answered from register fields and are reliable.
