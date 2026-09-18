# Knowledge Foundry — Implementation Plan

Response to `PROJECT_INITIATION.md` §26. Read that document and
`DOMAIN_SELECTION.md` first.

The objective of this plan is **not** a platform. It is to prove one complete loop
— *source ingestion → knowledge representation → retrieval → specialist answer →
citations → automated evaluation → adversarial testing → versioning → public
evidence* — for one knowledge area, cheaply enough that the marginal cost of the
second knowledge area is a config directory rather than an engineering project.

---

## 1. Governing constraints

These four constraints determine almost every choice below.

1. **No lock-in.** Every external dependency (LLM, embedding model, host) sits
   behind a small interface with at least two working implementations, one of
   which runs locally with no API key.
2. **Cost floor near zero.** The full loop must run on a laptop and on a $6/month
   VPS. Infrastructure cost scales with usage and corpus size, never with the
   number of knowledge areas (§17).
3. **The knowledge is the asset.** Everything that makes a knowledge area valuable
   — sources, provenance, ontology, prompts, evaluation sets, version history — is
   plain text in git. The database is a derived artefact and is always rebuildable
   from the repository plus the source cache.
4. **Evaluation is a feature, not a test suite.** Evaluation results are published
   product surface (§11, §15), so they are first-class data, versioned alongside
   the knowledge.

### Deliberate non-goals for the MVP

Per §23: no mobile app, no accounts, no payments, no branding, no fine-tuning, no
GPU infrastructure, no graph database, no agent framework. The knowledge graph is
a triple table, not Neo4j. Multi-tenancy is a field and a scope filter, not an
isolation architecture.

---

## 2. Technology choices

| Concern | Choice | Why, and what replaces it |
|---|---|---|
| Language | Python 3.11, stdlib-first | Ubiquitous; the entire core loop has **zero** required third-party packages except PyYAML |
| Store | One SQLite file per knowledge area | Single-file, atomic, backs up with `cp`, needs no server. Replaced by Postgres behind `storage.py` if concurrency demands it |
| Keyword retrieval | SQLite **FTS5** with BM25 | In-process, no Elasticsearch, excellent on exact technical terminology (§8) |
| Semantic retrieval | Pluggable embedder; default **local LSA** (TF‑IDF + truncated SVD) | Zero cost, offline, deterministic, trained on the KA's own corpus. Swappable for `sentence-transformers` or any OpenAI-compatible endpoint by changing one manifest line |
| Vector index | Flat cosine scan in the KA process | At MVP corpus sizes (10⁴–10⁵ chunks) a flat scan is milliseconds. A real ANN index is a later optimisation, not an architecture change |
| Graph | `triples` table + recursive CTE | §23 says no complicated knowledge graphs. Entity→relationship→entity in SQL covers §8's example |
| LLM | Provider registry; default **local extractive** provider | CI and the default install need no API key. `anthropic` and `openai_compatible` (Ollama, vLLM, Together, OpenRouter) implement the same interface |
| Web | stdlib `http.server` + Jinja2 templates | No framework. Serves the ask endpoint, the evidence page and the source viewer |
| Scheduling | GitHub Actions cron (or `cron` on the VPS) | Nightly evaluation is a workflow file, not a service |
| Packaging | `pyproject.toml`, optional extras | `pip install -e .` gets a working system; extras add remote models |

**Why a local LSA embedder is the default and not a transformer.** A 400 MB model
download is a poor default for a system that must run in CI, in a container and on
a founder's laptop identically. TF‑IDF+SVD over the KA corpus captures
co-occurrence structure well enough to make hybrid fusion meaningfully better than
BM25 alone, costs nothing, is deterministic across runs (which matters enormously
for regression testing), and is 30 lines behind an interface. When evaluation shows
a transformer embedder earns its keep on a given knowledge area, that knowledge
area sets `retrieval.semantic.provider: sentence_transformers` and rebuilds. The
evaluation suite is how that decision gets made — which is the whole point of §3.

---

## 3. Directory structure

```
knowledge-foundry/
├── PROJECT_INITIATION.md          # the brief
├── DOMAIN_SELECTION.md            # shortlist scoring → KA-001
├── IMPLEMENTATION_PLAN.md         # this document
├── pyproject.toml
├── Makefile                       # build / eval / redteam / serve
├── src/foundry/
│   ├── manifest.py                # KA schema, load + validate (§6)
│   ├── storage.py                 # SQLite schema, provenance, FTS5 (§7)
│   ├── chunking.py
│   ├── ingest/                    # fetchers, extractors, pipeline (§7)
│   ├── embeddings/                # provider registry: local LSA, remote
│   ├── retrieval/                 # keyword, semantic, structured, graph, hybrid (§8)
│   ├── llm/                       # provider registry: local, anthropic, openai-compatible (§4)
│   ├── specialist/                # agent, prompts, grounding + citation validation (§9, §10)
│   ├── evaluation/                # dataset, graders, runner, metrics, regression (§11, §12)
│   ├── redteam/                   # adversarial generators + judge (§13)
│   ├── versioning.py              # snapshot, publish gate, version log (§12)
│   ├── feedback.py                # expert challenges → new tests (§14)
│   ├── web/                       # ask API + public evidence page (§15, §16, §19)
│   └── cli.py                     # `foundry ...`
├── knowledge_areas/
│   └── kb-001-battery-failure/    # the knowledge asset — all plain text
│       ├── manifest.yaml
│       ├── sources.yaml           # the source register
│       ├── ontology.yaml          # entities + relationships
│       ├── prompts/system.md
│       ├── evaluation/*.yaml      # curated question sets
│       ├── versions/              # published version log + eval snapshots
│       └── challenges/            # expert submissions
├── var/                           # derived, git-ignored: kb.sqlite3, raw/, reports/
└── tests/
```

The split is the architecture: `src/foundry/` is the **factory** and is
domain-agnostic; `knowledge_areas/<id>/` is the **asset** and contains no code.
If a knowledge area ever needs Python to work, the factory has a gap.

---

## 4. Data model

One SQLite database per knowledge area, under `var/<ka-id>/kb.sqlite3`.

```
sources          id, uri, title, publisher, authority(1-5), source_type,
                 published_at, retrieved_at, licence, doc_version,
                 content_sha256, raw_path, status
documents        id, source_id, title, text, char_count, extracted_at, extractor
chunks           id, document_id, source_id, ordinal, text, token_estimate,
                 section, char_start, char_end
chunks_fts       FTS5(text, section) external-content over chunks   → BM25
embeddings       chunk_id, model, dim, vector (BLOB float32)
entities         id, name, kind, aliases, description
triples          subject_id, predicate, object_id, source_id, confidence
claims           id, chunk_id, source_id, text, kind(fact|range|requirement),
                 subject, extracted_at
kb_versions      version, created_at, manifest_sha, corpus_sha, index_sha,
                 eval_report_id, status(draft|published|rolled_back), notes
eval_runs        id, version, started_at, suite, metrics_json, passed
eval_results     run_id, question_id, score_json, answer_json, passed
challenges       id, submitted_by, question, claimed_problem, status, created_at,
                 promoted_to_test
```

**Provenance is structural, not a comment.** A chunk cannot exist without a
document, which cannot exist without a source carrying URI, publisher, dates,
licence and content hash. `Where did this answer come from?` (§7) is a join, so it
cannot silently stop being true.

The `claims` table implements §7's *"claims derived from source"* and is what the
grounding checker verifies answer sentences against.

---

## 5. Ingestion pipeline (§7)

```
sources.yaml → fetch → cache(sha256) → extract → clean → chunk → index → triples
```

* **Fetchers** by URI scheme: `http(s)`, `file`, and `inline` (for expert-supplied
  text). Every fetch writes the raw bytes to `var/<ka>/raw/<sha256>` and records
  the hash. Re-ingesting unchanged content is a no-op — this is what makes §24.7
  (*update without rebuilding everything*) work.
* **Extractors** by content type: HTML (stdlib `HTMLParser`, no BeautifulSoup
  requirement), PDF (`pypdf`, optional extra), plain text, Markdown.
* **Politeness and legality.** Per-host rate limiting, `robots.txt` respected, a
  declared user agent, and a `licence` field required on every source. Sources
  whose licence forbids storage are registered as *pointers* — title, publisher,
  URI and our own summary — and are never mirrored. The manifest's
  `known_limitations` must say so.
* **Chunking** is section-aware: split on headings first, then to a target ~1200
  characters with ~150 character overlap, never mid-sentence. Section headings ride
  along with the chunk because they carry most of the retrievable context in
  standards and reports.
* **Incremental.** `foundry ingest <ka>` re-fetches only sources whose
  `refresh_after` has elapsed, and reindexes only documents whose hash changed.

---

## 6. Retrieval architecture (§8)

Four retrievers behind one `Retriever` protocol, fused by **Reciprocal Rank
Fusion** (`score = Σ w_i / (k + rank_i)`, k=60):

| Retriever | Mechanism | Earns its place on |
|---|---|---|
| Keyword | FTS5 BM25 | exact terms: `UL 9540A`, `UN 38.3`, `LiPF6`, clause numbers |
| Semantic | cosine over LSA vectors | paraphrase: "cells getting hot and catching fire" → thermal runaway |
| Structured | SQL over metadata | "what do *regulators* say", "since 2023", authority-weighted |
| Graph | triples, 1–2 hops | "what detects thermal runaway" via `detected_by` |

RFF is used rather than score normalisation because BM25 and cosine scores are not
comparable and any attempt to make them comparable is a tuning liability.

Retrieval is then **authority-reranked**: a source's `authority` (1–5) and recency
adjust final ordering, so a regulator's text outranks a blog on a tie. Weights live
in the manifest, not in code, so a knowledge area can be tuned without a deploy —
and any change to them is caught by the retrieval metrics in the nightly suite.

---

## 7. Specialist agent (§9, §10)

```
question → scope check → retrieve(hybrid) → sufficiency gate → answer → validate → respond
```

1. **Scope check** — is this question inside the knowledge area's declared scope?
   Out-of-scope questions are refused *before* retrieval, which is the cheapest
   possible defence against §13's "questions outside the knowledge base".
2. **Sufficiency gate** — if the best evidence is below threshold, the agent
   abstains and says what it would need. This is the mechanism behind §9's critical
   rule. Abstention is a *success* state in the evaluation suite, not a failure.
3. **Generation** — evidence is presented to the model as numbered, provenance-
   tagged passages; the system prompt (from the KA, not from code) requires
   `[n]` citations, explicit separation of **evidence** from **inference**, and
   surfacing of contradictions between sources.
4. **Validation — the part that matters.** Every answer is machine-checked *after*
   generation, independently of the model:
   * every citation marker resolves to a chunk that was actually retrieved;
   * every claim-bearing sentence is grounded — checked by content-term overlap and
     numeric agreement against its cited chunks;
   * numbers and quoted strings in the answer appear in the cited evidence
     (a cheap, effective hallucination trap for exactly the failure mode this
     domain suffers from);
   * sentences that fail grounding are reported as **unsupported claims**, which
     drives the `unsupported_claim_rate` metric and can force a downgrade to
     abstention.

   This is why the plan does not depend on the model being good. A weak model that
   abstains honestly is a better product than a strong model that is confidently
   wrong (§25).
5. **Response shape** per §10: `answer`, `reasoning`, `sources[]`, `evidence
   confidence`, `limitations`, plus `knowledge_version` and `evaluation_status` for
   §19's API contract.

---

## 8. Evaluation framework (§11, §12)

Question sets are YAML in the knowledge area. Nine types, each with its own grader:

`factual` · `multi_document` · `reasoning` · `ambiguous` · `unanswerable` ·
`citation` · `temporal` · `adversarial` · `out_of_scope`

Graders are composable and deterministic first: `must_include`, `must_not_include`,
`numeric_within`, `must_cite_source`, `must_abstain`, `must_flag_ambiguity`,
`retrieval_hit@k`. An optional `judge` grader uses an LLM for open-ended answers —
but the suite is designed so that a full regression verdict can be reached with
**no model API at all**, because a CI gate that needs a paid API is a CI gate that
gets disabled.

Metrics reported per run (§12 — and deliberately *not* collapsed into one number):

```
retrieval_precision@k   retrieval_recall@k     answer_correctness
citation_validity       citation_relevance     unsupported_claim_rate
hallucination_rate      abstention_correctness source_coverage
corpus_freshness_days   contradiction_rate     regression_rate
```

**Regression gate.** A new build is compared against the last published version's
report. It publishes only if no metric regresses beyond its declared tolerance and
no previously-passing question now fails. Otherwise the build stays `draft` and the
last published version keeps serving. Newly-failing questions are listed by id, so
"investigate/rollback" in §12's diagram is a concrete list, not a feeling.

---

## 9. Red-team framework (§13)

A generator produces adversarial questions from the corpus and ontology across
eight strategies — out-of-scope, false premise, misleading framing, unanswerable,
over-specific, multi-source, contradictory-source, temporal trap — using the
knowledge area's own entities so the attacks are domain-real rather than generic.
Both a template-based generator (free, deterministic, always available) and an
LLM-backed generator (richer, optional) implement one interface.

A judge then scores each response on five failure modes drawn directly from §13:
`hallucinated`, `irrelevant_citation`, `misread_source`, `unsupported_deduction`,
`missed_uncertainty`.

**The loop that matters:** every red-team question the system fails is written into
`evaluation/regression_from_redteam.yaml` with its expected behaviour. Today's
break becomes tomorrow's permanent test. That is success criterion #9, and the same
path is used for expert challenges (§14) — `foundry challenge promote <id>` turns a
human's complaint into a graded test case.

---

## 10. Versioning and publishing (§12, §15)

A version is `v0.MINOR.PATCH` over a **content hash** of (manifest + source
register + corpus hashes + index config + prompts). Rebuilding identical inputs
yields an identical version, so "what changed between versions" is answerable
mechanically. Each published version records its full evaluation report, and the
version log renders as the *Updates* section of the public evidence page. Rollback
is repointing `current` at the previous version — the database keeps both.

## 11. Deployment model (§17, §18)

One process, one container, all knowledge areas:

```
  nginx/Caddy (TLS) → foundry web (single process)
       ├── var/kb-001-battery-failure/kb.sqlite3
       ├── var/kb-002-.../kb.sqlite3          ← a KA is a directory, not a server
       └── shared inference client → local model | any hosted API
```

* Knowledge areas are directories and SQLite files in one process. Adding the
  tenth knowledge area adds disk, not a VPS (§17).
* Nightly evaluation runs as a scheduled workflow, writes a report, and publishes
  or holds the version.
* Cost at MVP: VPS ~$6–12/month, object storage for raw cache ~$1, domain ~$1.
  Inference is the only usage-scaled cost, and the default local provider makes it
  optional. Comfortably inside the ~$100/month target with room for the models.
* **Multi-tenancy** (§18) is designed in now at its cheapest useful form: each KA
  carries `visibility: public | private | enterprise` and a `tenant`; the web layer
  filters by an API token → tenant mapping. Real isolation (separate process or
  separate deployment per enterprise tenant) reuses the same layout unchanged.

---

## 12. Build sequence (§22)

Implemented as small, independently tested increments. Every step ends with
passing tests — a step that cannot be tested is a step that is wrongly scoped.

| Step | Deliverable | Maps to |
|---|---|---|
| 1 | Manifest schema + validation; storage schema + provenance | Sat AM |
| 2 | Ingestion: fetch, cache, extract, clean, chunk, index | Sat AM |
| 3 | Embeddings + hybrid retrieval + fusion + authority rerank | Sat AM |
| 4 | LLM provider registry incl. keyless local provider | Sat AM |
| 5 | Specialist agent: scope gate, sufficiency gate, citations, grounding validation | Sat PM |
| 6 | KA-001 corpus: source register, ingest, ontology | Sat PM |
| 7 | Evaluation: question sets, graders, runner, metrics | Sat PM/Eve |
| 8 | Red-team generator + judge; failures → regression tests | Sun AM |
| 9 | Versioning, regression gate, publish/rollback | Sun AM |
| 10 | Web: ask API, evidence page, source viewer, "try to break it" | Sun PM |
| 11 | Nightly workflow, Makefile, deployment notes | Sun PM |
| 12 | KA-002 skeleton — proves criterion #10 costs a directory, not a rewrite | Sun PM |

## 13. How we will know it worked

The MVP is judged against §24, not against feature count. Concretely:

* `make build` ingests KA-001 from its source register with no manual steps.
* `foundry ask kb-001-battery-failure "..."` returns an evidence-first answer whose
  every citation resolves to a stored chunk with a URI, publisher and retrieval date.
* `make eval` produces a multi-metric report; `make redteam` produces failures that
  become tests; `make publish` refuses to publish a regression.
* A second knowledge area is created by adding a directory — no changes under
  `src/foundry/`.

If the system cannot yet answer a question, the correct MVP behaviour is to say so
and show what it would need. That is the product (§25).
