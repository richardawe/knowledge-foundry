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
| `kb-001-battery-failure` | Lithium-ion battery failure and thermal runaway | 34 (29 indexed, 5 pointer-only) | ~1,270 | 100 |
| `kb-002-industrial-fire-explosion` | Industrial fire and explosion hazards | 13 (11 indexed, 2 pointer-only) | ~190 | 27 |

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

Asking a *new* question runs hybrid retrieval and a local model, which a static
page cannot do. For that, run it locally (below).

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
make ollama-check        # confirms the endpoint and lists pulled models
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
* KA-001 currently passes **73 of its 100** evaluation questions. The failures
  are published on the evidence page rather than hidden, and most are generation
  failures rather than retrieval failures — retrieval recall is 0.95.
* The scope gate is lexical and misses paraphrased out-of-domain questions; the
  sufficiency gate catches most of what it misses. This is measured, not assumed.
* The grounding checker is lexical. It catches fabricated numbers and quotations
  reliably and will miss a paraphrase that changes meaning while keeping the
  vocabulary.

That list is the product working as intended. The claim is not that the system is
smart — it is that you can see what it knows, where that came from, how it is
tested, where it failed, and what changed.
