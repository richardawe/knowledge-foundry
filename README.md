# Knowledge Foundry

A factory for building specialist AI knowledge systems that can survive being
challenged by people who know the subject.

Not a chatbot. Each **knowledge area** is a curated corpus with full provenance,
a hybrid retriever, an evidence-first agent that abstains rather than guesses,
and a published evaluation suite it is scored against on every build — including
the questions it currently fails.

> Don't trust it. Test it.

See [`PROJECT_INITIATION.md`](PROJECT_INITIATION.md) for the brief,
[`DOMAIN_SELECTION.md`](DOMAIN_SELECTION.md) for why the first domain was
chosen, and [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the
architecture and the reasoning behind each choice.

## What ships today

| Knowledge area | Domain | Sources | Passages | Eval questions |
|---|---|---|---|---|
| `kb-001-battery-failure` | Lithium-ion battery failure and thermal runaway | 34 (29 indexed, 5 pointer-only) | ~1,270 | 103 |
| `kb-002-industrial-fire-explosion` | Industrial fire and explosion hazards | 13 (11 indexed, 2 pointer-only) | ~190 | 30 |

KA-002 exists to prove the point of the whole project: it was built by adding a
directory of YAML and Markdown, with **no changes under `src/foundry/`**. A test
enforces that (`tests/test_factory.py`).

## Public evidence pages

The read-only surface is published as static HTML to GitHub Pages:

**https://richardawe.github.io/knowledge-foundry/**

That includes every stored passage, the full source register with licences, the
evaluation results, the red-team findings, the version history, and a
**transcript** of every question the system has been asked with the answer it
actually gave — failures included. It is rendered from the same templates the
live server uses, so the snapshot cannot drift from the system.

**The ask box on that page works, with or without anything running.**

With **nothing running anywhere**, there is still a way to ask the real thing:
[open an issue with the *Ask the specialist* form](https://github.com/richardawe/knowledge-foundry/issues/new?template=ask.yml).
A workflow builds the corpus, answers the question through the same retrieval,
gates and grounding validation every evaluation question goes through, and posts
the reply with its sources and its scores. It takes a few minutes rather than a
few seconds — there is no server to be instant.

What that buys is worth the wait. The exchange is recorded as a **challenge**,
committed to this repository, and shown on the knowledge area's challenges page.
If the answer is wrong and you say so, it can be promoted into the permanent
evaluation suite, which means the same mistake cannot be made twice without a
test failing. A chat box produces a conversation nobody can audit afterwards;
this produces a record sitting next to the tests it can become.

With **nothing running**, it searches the 103 answers the system has already
given — published as static JSON by the machine that produced them — and shows
the closest one with its real evidence, badged as recorded rather than live. If
that recorded answer failed its own evaluation, the page says so. If nothing is
close enough, it says that too rather than forcing a match. This is the same
shape as the sibling `localtest` project: the model runs on the Mac, the
results are published as files.

It never retrieves over the corpus in the browser — the 1,267 passages are
never downloaded. It matches your question against recorded *questions*, so it
cannot produce an answer the real pipeline has not already produced.

With an **instance running**, it switches to live answers automatically. It is
the interface; the engine runs where your knowledge and your model are. Start the project on the machine with Ollama
and the published page connects to it and answers live — retrieval, the gates,
generation and grounding validation all run in that instance, over its API. The
page reimplements nothing, so it cannot drift from the system it describes.

```bash
make serve          # on the machine with Ollama
```

Then reload the published page: it detects the instance, shows
*"Connected to http://localhost:8000"*, and the box goes live. With nothing
running it says so and points at the transcript instead of offering a dead form.

To let other people use it, expose your instance over HTTPS (a Cloudflare
Tunnel, say) and publish with that address baked in:

```bash
make deploy API_BASE=https://kf.example.dev
```

Readers can also point the page anywhere themselves — the endpoint is editable
on the page and shareable as `?api=<url>`.

> **Browser note.** Chrome and Firefox allow an HTTPS page to call
> `http://localhost`. Safari is stricter and may block it — use Chrome or
> Firefox, run a tunnelled HTTPS endpoint, or just open the local instance
> directly at `http://localhost:8000`, which serves the same pages.

## Quick start

```bash
pip install -e ".[dev,web]"

make build          # fetch sources, extract, chunk, index, cut a draft version
make eval           # run the evaluation suites and apply the threshold gate
make redteam        # generate adversarial questions; promote failures to tests
make publish        # publish only if nothing regressed against the last release
make serve          # ask box + API on http://127.0.0.1:8000

make site           # render the public site into ./site
make deploy         # publish ./site to the gh-pages branch
```

## Inference runs on your machine

Both knowledge areas call a local **Ollama** (`llama3.1:8b`) at
`http://localhost:11434/v1` — its OpenAI-compatible endpoint. Nothing is sent
to a hosted API and inference costs nothing.

```bash
ollama serve &
ollama pull llama3.1:8b
make doctor              # checks every link in the chain and says what to fix
```

It uses Ollama's **native** API (`/api/chat`, falling back to `/api/generate`),
not the OpenAI-compatible shim at `/v1` — the shim only exists from Ollama
0.1.24, and the native API is what the `ollama` Python package calls, so a
machine already running Ollama workloads is proven against exactly this
surface. The daemon address comes from `OLLAMA_HOST`, the same variable every
other Ollama workload honours.

`make doctor` output looks like this:

```
  [PASS] Ollama daemon          reachable at http://localhost:11434
  [PASS] Model llama3.1:8b      available
  [PASS] Native API /api/chat   available
  [PASS] kb-001 corpus          29 sources, 1267 passages, version v0.1.0
  [WARN] Local server           nothing answering on http://localhost:8000
         fix: foundry serve --port 8000
```

Where Ollama is not running — CI, a fresh clone, a container — the system
answers with the keyless local provider instead and **records that it fell
back** in the stored answer, so an evaluation run that used the weaker model
stays identifiable afterwards. Set `specialist.llm.fallback: false` in a
manifest to make an unreachable model a hard failure instead.

Ask it something:

```bash
foundry ask kb-001-battery-failure "What watt-hour limit applies to a spare lithium-ion battery in cabin baggage?"
```

Or over HTTP:

```bash
curl -s localhost:8000/knowledge/kb-001-battery-failure/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What gases are released when a cell vents?"}'
```

```json
{
  "answer": "...",
  "sources": [{"rank": 1, "citation": "...", "uri": "...", "cited": true}],
  "confidence": "medium",
  "knowledge_version": "v0.1.1",
  "evaluation_status": "passed"
}
```

## Where each part runs

Answering splits cleanly in two, and the two halves run in different places.

```
GitHub Actions  (all the Python)            your Mac  (only the model)
──────────────────────────────              ────────────────────────────
ingest -> chunk -> index
retrieve -> gates -> build prompts
  write inference/<ka>/requests/*  ──────▶
                                            worker.sh: read prompts,
                                            call Ollama, write responses/*
                                 ◀──────
validate: citations resolve, claims
grounded, numbers present
score, version, publish or hold
export and deploy the evidence pages
```

**Actions does everything deterministic**: ingestion, retrieval, the gates,
validation, grounding, scoring, versioning, the red team, the site. It runs on
a schedule, needs no model, and produces a complete publishable site on its own
using the keyless extractive provider.

**Your Mac does only the model call.** `worker.sh` reads the queued prompts,
calls Ollama, and pushes the replies back. It performs no retrieval, applies no
gates and scores nothing — a machine that generates an answer must not be the
one that marks it.

The transport is the repository itself: a queue of files, pushed and pulled like
any other change. No server, no tunnel, no open port. If the Mac is asleep the
work simply waits.

A reply is only accepted if the prompt it answered still hashes to the prompt
the corpus produces now. If the corpus moved in between, the model answered
against different evidence than we would validate it against, and the stale
reply is discarded rather than scored.

```bash
# on the Mac, once Ollama is running
./worker.sh          # or: make worker
```

### And a third path, with no machine of yours at all

The split above moves the *model* off the runner. The public ask path moves the
*whole question* onto it: someone opens an issue, `.github/workflows/ask.yml`
answers it, and the reply is posted back.

```
a person opens an issue          GitHub Actions
─────────────────────────        ──────────────────────────────────
"Ask the specialist" form  ────▶ build the corpus
                                 retrieve -> gates -> answer
                                 validate, score, record a challenge
                           ◀──── post the reply; republish the site
```

This is the same trick as the queue — the repository is the transport — applied
to questions arriving from outside rather than prompts going out. It needs no
host, puts no API key in anybody's browser, and every exchange lands in
`knowledge_areas/<id>/challenges/` as durable repository state rather than in a
cache that a cold runner throws away.

The model for that path is configured by repository secret rather than by the
manifest, so it can be a hosted one. See
[`DEPLOYMENT.md`](DEPLOYMENT.md#answering-public-questions-without-a-host).

## How it works

```
sources.yaml ─▶ fetch ─▶ cache(sha256) ─▶ extract ─▶ chunk ─▶ index ─▶ claims
                                                                  │
      ┌───────────────────────────────────────────────────────────┘
      ▼
  hybrid retrieval          BM25 · LSA vectors · structured metadata · ontology graph
      │                     fused by reciprocal rank fusion, reranked once by authority
      ▼
  specialist agent          scope gate ▸ specificity gate ▸ sufficiency gate
      │                     ▸ generate ▸ grounding validation (can withhold the answer)
      ▼
  evidence-first answer     answer · reasoning · sources · confidence · limitations
      │
      ▼
  evaluation + red team ─▶ regression gate ─▶ publish or hold ─▶ public evidence page
```

Six design commitments, each with its reasoning in the implementation plan:

**Nothing is locked in.** Every external dependency sits behind an interface
with at least two implementations, one of which runs locally with no API key.
The default LLM and embedder are keyless, so the full loop — including the CI
regression gate — runs at zero cost.

**Provenance is structural, not a convention.** A chunk cannot exist without a
document, which cannot exist without a source carrying a URI, publisher, dates,
licence and content hash. *"Where did this come from?"* is a SQL join.

**The knowledge is the asset.** Sources, ontology, prompts, evaluation suites
and version history are plain text in `knowledge_areas/`. The database in `var/`
is derived and always rebuildable.

**Abstention is a success state.** Three gates can end a turn without an answer,
and the evaluation suite rewards declining when declining is right.

**The model is not trusted.** Every answer is machine-checked after generation
against the evidence it cites — citations must resolve, sentences must be
grounded, stated numbers and quotations must appear in the cited passages. A
fluent answer that fails grounding is withheld, not shipped.

**Failures become tests.** Every red-team break and accepted expert challenge is
promoted into the knowledge area's regression suite automatically. The same
failure cannot recur silently.

## Licensed material

Copyrighted standards (NFPA 855, UL 9540A, IEC 62619, UN 38.3, UNECE R100,
NFPA 652, COMAH) are registered as **pointer sources**: named, linked and
attributed, but never stored, retrieved or quoted. The system will tell you a
standard governs a topic and decline to quote its clauses. This is enforced
structurally — a pointer source has no rows in the content tables — and asserted
by tests.

## Repository layout

```
src/foundry/            the factory — domain-agnostic machinery
  manifest.py           the knowledge-area contract (§6)
  storage.py            SQLite schema; provenance built into the shape
  ingest/               fetch, cache, extract, chunk, claims (§7)
  retrieval/            keyword, semantic, structured, graph, fusion (§8)
  llm/                  provider registry; keyless local default (§4)
  specialist/           gates, prompts, grounding validation (§9, §10)
  evaluation/           suites, graders, metrics, gate (§11, §12)
  redteam/              adversarial generation, judging, promotion (§13)
  versioning.py         content-hash versions, publish gate, rollback
  feedback.py           expert challenges → permanent tests (§14)
  evidence.py, web/     public evidence pages and the API (§15, §16, §19)

knowledge_areas/<id>/   the asset — plain text, no code
  manifest.yaml  sources.yaml  ontology.yaml  prompts/  evaluation/  versions/

var/                    derived: databases, caches, reports (git-ignored)
```

## Cost

The prototype target is under ~$100/month for the whole lab. In practice:
a $6–12 VPS, ~$1 of object storage for the raw cache, a domain, and inference —
which is the only usage-scaled cost and is optional, because the default
provider runs locally. Adding the tenth knowledge area adds a SQLite file, not a
server. See [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Honest status

This is a working MVP, not a finished product.

* The default extractive answer provider does not synthesise across passages; it
  selects and cites. It cannot hallucinate, which makes it a useful control, but
  its prose is stitched rather than written. Point the manifest at a hosted model
  and the evaluation suite will tell you whether that is worth paying for.
* KA-001 currently passes **76 of its 103** evaluation questions, KA-002 **20 of its
  30**. The failures are published on the evidence page rather than hidden, and most
  are generation failures rather than retrieval failures — retrieval recall is 0.96.
* **No expert has challenged it yet.** The path is now open — anyone can ask via
  an issue, and every exchange is recorded — but a challenge count of zero is a
  fact about this project, not a feature of it. §14 of the brief is only half
  built until people are actually using it.
* The scope gate is lexical and misses paraphrased out-of-domain questions; the
  sufficiency gate catches most of what it misses. This is measured, not assumed.
* The grounding checker is lexical. It catches fabricated numbers and quotations
  reliably and will miss a paraphrase that changes meaning while keeping the
  vocabulary.
* **Nothing measures completeness.** Every metric asks whether what was said is
  supported. None asks whether what was left out changes the meaning. A worked
  example: asked for the aircraft-cabin watt-hour limit, the specialist cites the
  FAA correctly, scores 100% citation-valid and 100% grounded, and drops the words
  "with airline approval" from a passage it had open — so it reports that two
  101–160 Wh spares are permitted without saying the carrier must consent. That
  answer is published. Closing this is the next design decision, not a bug fix.

That list is the product working as intended. The claim is not that the system is
smart — it is that you can see what it knows, where that came from, how it is
tested, where it failed, and what changed.
