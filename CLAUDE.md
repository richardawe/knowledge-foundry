# Knowledge Foundry — working memory

Read this before touching anything. It carries the context a fresh session needs
and the things that have already gone wrong.

## What this is

A one-person factory for specialist-knowledge systems, built from
`PROJECT_INITIATION.md`. Each knowledge area is a directory of configuration; the
code in `src/foundry/` knows nothing about batteries or fires. That separation is
the product. **Do not build a generic chatbot.**

Two knowledge areas exist:

* `kb-001-battery-failure` — lithium-ion battery failure, 34 sources, 103 questions
* `kb-002-industrial-fire-explosion` — 13 sources, 30 questions

Branch: `claude/knowledge-foundry-initiation-3vvxdh`. Do not push elsewhere.
Do not open a PR unless asked.

## Where things run

```
GitHub Actions  (all the Python)            the Mac  (only the model)
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

The queue travels through the repository itself. No tunnel, no open port. If the
Mac is asleep the work waits. Actions answers with the keyless extractive
provider so it always produces a publishable site alone; the Mac's answers are an
upgrade applied when they arrive.

There is a third path, and it uses the same seam. `.github/workflows/ask.yml`
answers a question submitted as a GitHub issue: build, retrieve, gates, model,
validate, score, post the reply, commit the exchange as a challenge, republish.
No host, no key in a browser. The model there comes from `OPENROUTER_API_KEY`
and the `FOUNDRY_LLM_*` repository variables, not from the manifest — the
manifest names where *production* inference runs, and a runner is not that.

**A challenge is repository state, not `var/` state.** `var/` is disposable and
is rebuilt from the source register; a question somebody asked is not derivable
from any source, so it is written to `knowledge_areas/<id>/challenges/` and
re-imported by `build`. The repository wins on conflict, so a review decision
recorded there survives a rebuild. Same reasoning as the inference queue:
anything arriving from outside travels in the repository.

**The published ask box answers in the browser.** `browser_index.py` exports the
corpus (336 KB gzipped for kb-001) and `web/static/engine.js` reimplements
retrieval, the gates, extraction and grounding in JavaScript. No model, no
server, no key, milliseconds per answer.

That is a second implementation of graded behaviour, so parity is tested, not
hoped for: `tests/test_browser_parity.py` compares verdict, evidence ordering
*and answer text* against Python over every suite question, on the real corpus
when one is built. Comparing statuses alone is not enough — a threshold can
move a long way without flipping one, and the first question it flips would be
a reader's. Mutation-checked: 8 of 11 deliberate breaks are caught; the misses
are paths the suite does not exercise (responsiveness threshold, prose filter,
numeric-cue boost).

Parity required one production change: **keyword retrieval now uses BM25 over
foundry's own tokenizer**, not SQLite FTS5's. `text.py` opens by saying
tokenisation must not drift between components; keyword retrieval was the one
that had, invisibly, until a second implementation had to agree with it. Worth
78/103 against FTS5's 79 and the hybrid's 77. `retrieval.keyword.engine: fts5`
restores the old one.

`Specialist.ask` is `prepare()` + `judge()`. `prepare` runs retrieval, the gates
and prompt assembly, and either settles the turn or hands back a prompt. `judge`
runs validation and scoring over whatever a model returned. The split is what
makes the two-machine pipeline possible.

## Invariants — do not weaken these

* **The worker never answers without a model.** `provider_for(require_model=True)`
  refuses `local_extractive` and `scripted`. A keyless reply in the queue would be
  graded and published as the model's work.
* **A reply is rejected unless its `prompt_sha` still matches** the prompt the
  corpus produces now. Otherwise it is graded against evidence it never saw.
* **Questions a gate settles are never queued.**
* **No domain vocabulary in executable factory code.** An AST test enforces this
  and has already caught a real leak (`"nfpa"`, `"iec"` hard-coded as retrieval
  cues — they belong in manifests).
* **Every queued answer names the machine that produced it.** `cmd_worker`
  defaults `--name` to the hostname. Anonymous answers are how fabricated ones got
  in (see below).
* **A reply posted to a person names what produced it.** `intake` prints the
  provider and model on every answer and flags a fallback explicitly. The ask
  workflow passes `--require-model` whenever a key is configured, for the same
  reason the worker does: a corpus extract must never stand where a model's
  answer would.

## Commands

```
python -m foundry.cli list --ids
python -m foundry.cli build <ka>
python -m foundry.cli ask <ka> "question"
python -m foundry.cli eval <ka>
python -m foundry.cli redteam <ka> --count 64 --promote
python -m foundry.cli version <ka> publish
python -m foundry.cli plan <ka>          # queue prompts
python -m foundry.cli worker             # Mac only: call Ollama
python -m foundry.cli apply <ka>         # validate + score queued replies
python -m foundry.cli deploy --base-url /knowledge-foundry
python -m foundry.cli doctor
python -m foundry.cli intake --body-file b.md --challenge-id ch_issue_1   # public ask path
python -m pytest -q                      # 358 tests (parity needs node)
```

## Current state

* **Nightly runs green in Actions** (run 4, `35393758504`). Full pipeline: tests,
  build, eval, red team, publish, plan, export, gh-pages deploy, push.
* Live at `https://richardawe.github.io/knowledge-foundry/`
* kb-001 76/103 (74%), kb-002 20/30 (67%) — the suite grew when red-team failures
  were promoted; the pass count did not. Hallucination rate 0.000 on both.
  Red team survival 92% / 94%.
* The published page's ask box shows "no instance reachable" and falls back to
  searching recorded answers. **That is correct.** A static page cannot run
  retrieval and a model. The live ask needs `foundry serve` on the Mac with
  Ollama behind it.

## Open decisions — waiting on the user, do not build these unasked

1. **Completeness: metric or gate?** Found by auditing a real answer: the FAA
   watt-hour answer scores 100% citation-valid and 100% grounded while dropping
   "with airline approval" from a chunk it cited. **No metric in the framework can
   detect an omission** — every one asks "is what was said supported?", none asks
   "does what was left out change the meaning?" A gate would withhold such answers
   and will visibly lower the pass rate. A metric only reports. Claude leans gate;
   the user has not decided.
2. **Should `bf-fact-010` have passed?** It asserts the right number (100 Wh) and
   misrepresents the rule. If the answer is no, the suite needs questions that
   assert the *absence* of a condition — a tenth question type.

3. **Nobody has challenged it yet.** The path is built and open (issue form ->
   workflow -> recorded challenge), but the challenge count is still 0. §14 is
   only half built until real people use it, and that is an invitation to send,
   not code to write.

Also noted, not acted on: `WebFetch` gets 403 from faa.gov; plain `curl` needs a
browser user-agent for 200. If ingestion re-fetches that source with default
headers it will fail. The ingest user-agent has not been checked.

## Things that have already gone wrong

**Fabricated answers reached the published evidence chain.** Six responses
generated against a *mock* Ollama were committed. The mock returned one canned
reply — "limited to a rating of 100 watt hours" — to every prompt, stamped
`provider: ollama, model: llama3.1:8b`. `bf-fact-010` expects 100 Wh, so the
mock's sentence became a published "pass", and the first honest run afterwards was
recorded as a regression against it. Deleted; `var/` cache key bumped to abandon
the poisoned baseline; worker now records the hostname. **Nothing can distinguish
a mock Ollama from a real one over HTTP — provenance is the only defence.**

**The runner's `FOUNDRY_LLM_PROVIDER=local_extractive` leaked into pytest.** Four
tests that exist to prove "an unreachable model degrades visibly" ran through a
provider that cannot be unreachable. `tests/conftest.py` now clears ambient
provider/model/host/root vars. The same leak was a real hole in the worker, which
is why `require_model` exists.

**The nightly's 403** was a missing `permissions: contents: write`. Fixed and
verified.

**Don't over-build.** The user has said "Don't build yet, tell me what's possible"
once already. When a decision is theirs, ask and stop.

## Style

Commits end with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CyaxCLmf5HJ1WgSf3uE5pm
```

No model identifier anywhere in pushed artefacts — chat replies only.
Report failures plainly, with the output. A held build is the system working.
