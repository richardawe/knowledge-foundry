# Deployment

The constraint that shapes everything here: the whole lab must run for under
~$100/month, and infrastructure cost must scale with usage and corpus size,
never with the number of knowledge areas (PROJECT_INITIATION.md §17).

## Shape

```
         Caddy or nginx (TLS)
                 │
         foundry serve  ── one process, every knowledge area
                 │
   ┌─────────────┼─────────────┐
   ▼             ▼             ▼
var/kb-001/   var/kb-002/   var/kb-00N/
 kb.sqlite3    kb.sqlite3    kb.sqlite3
 raw/          raw/          raw/
   └─────────────┼─────────────┘
                 ▼
        shared inference client
     local model │ any hosted API
```

A knowledge area is a directory and a SQLite file. Adding the tenth one adds
disk, not a VPS.

## Minimum viable host

A 1 vCPU / 2 GB VPS is enough for the MVP. Retrieval is a flat cosine scan plus
FTS5 over ~10⁴ passages, which is milliseconds, and the default embedder has no
model to load.

```bash
git clone <repo> && cd knowledge-foundry
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[web,pdf,fast]"

foundry build kb-001-battery-failure
foundry eval  kb-001-battery-failure
foundry version kb-001-battery-failure publish

foundry serve --host 127.0.0.1 --port 8000
```

### systemd

```ini
# /etc/systemd/system/foundry.service
[Unit]
Description=Knowledge Foundry
After=network.target

[Service]
User=foundry
WorkingDirectory=/srv/knowledge-foundry
Environment=FOUNDRY_VAR=/srv/knowledge-foundry/var
ExecStart=/srv/knowledge-foundry/.venv/bin/foundry serve --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Put Caddy or nginx in front for TLS. The app sets `X-Content-Type-Options` and
`X-Frame-Options`; everything else (HSTS, rate limiting) belongs in the proxy.

### Nightly health check

`.github/workflows/nightly.yml` runs the §12 loop on a schedule and is the
reference implementation. On a VPS, the same thing is one cron line:

```cron
17 3 * * * cd /srv/knowledge-foundry && make nightly KA=kb-001-battery-failure >> var/nightly.log 2>&1
```

It builds, evaluates, red-teams, promotes new failures into the regression
suite, and publishes **only if nothing regressed**. A held build leaves the
previous published version serving.

## GitHub Pages

The evidence pages are static, so they host for nothing:

```bash
make site      # renders ./site with BASE_URL=/knowledge-foundry
make deploy    # worktree checkout of gh-pages, copy, commit, push
```

Then once, in the repository: **Settings → Pages → Source: Deploy from a
branch → `gh-pages` / `(root)`**. The site appears at
`https://<owner>.github.io/<repo>/` within a minute of the push.

### Making the published ask box work

The exported page carries an ask client that calls a running instance over the
JSON API. It resolves the endpoint in this order: `?api=` in the URL, then what
the reader last chose, then the address baked in at export (`API_BASE`), then
`http://localhost:8000`.

* **For yourself**: run `make serve` on the machine with Ollama. The published
  page finds it and answers live.
* **For everyone**: expose that instance over HTTPS and bake the address in:

  ```bash
  make deploy API_BASE=https://kf.example.dev
  ```

  A Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:8000`) is the
  cheapest way; the Mac has to be awake and serving.

Cross-origin calls need CORS, which the server sets on the read/ask API only —
never on the routes that write a challenge, so a page you did not open cannot
record anything against your knowledge base. The default allows any origin,
because the server binds to 127.0.0.1 and serves public knowledge-base content.
Narrow it when the instance is exposed beyond localhost:

```bash
export FOUNDRY_CORS_ORIGINS=https://richardawe.github.io
```

`BASE_URL` must match how Pages serves the repo. A *project* site is served
from `/<repo>/`, which is the default here. For a user site or a custom domain,
use `make site BASE_URL=` and `make deploy BASE_URL=`.

The deploy uses a **git worktree**, the same mechanism as the sibling
`localtest` project: the branch you are working on is never touched, and the
`gh-pages` contents are replaced wholesale each time so a page from an earlier
build can never linger looking current. Authentication prefers a configured
credential helper (the macOS keychain already has one) and only embeds
`GITHUB_TOKEN` when there is none, which is the case on a bare CI runner.

### Running it from launchd

`com.knowledgefoundry.agent.plist` runs the whole loop nightly on the Mac where
Ollama lives — build, evaluate, red team, publish or hold, then deploy the
evidence pages:

```bash
cp com.knowledgefoundry.agent.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.knowledgefoundry.agent.plist
launchctl start com.knowledgefoundry.agent      # run it now, to test
```

It expects the repository at `~/knowledge-foundry` and Ollama already running.
Logs land in `/tmp/knowledge-foundry.*.log`. To remove it:

```bash
launchctl unload ~/Library/LaunchAgents/com.knowledgefoundry.agent.plist
```

## Choosing models

Defaults are keyless and local, so nothing below is required.

Both knowledge areas ship pointing at a local Ollama:

```yaml
# knowledge_areas/<id>/manifest.yaml
specialist:
  llm:
    provider: openai_compatible    # Ollama's OpenAI-compatible endpoint
    model: llama3.1:8b
    fallback: true                 # degrade to the keyless local provider, and say so
retrieval:
  semantic:
    provider: lsa                  # or openai_compatible (nomic-embed-text), sentence_transformers
```

To use Ollama for embeddings too:

```bash
ollama pull nomic-embed-text
```

then set `retrieval.semantic.provider: openai_compatible` and rebuild the
index. The LSA default needs no model and works offline, so this is an
experiment to run, not an upgrade to assume.

Secrets come from the environment, never the manifest, so a knowledge area stays
publishable as plain text:

```bash
# Ollama on the same machine needs no configuration at all -- it is the default.
# Point elsewhere (another host, vLLM, a hosted API) with:
export FOUNDRY_LLM_BASE_URL=http://localhost:11434/v1
export FOUNDRY_LLM_API_KEY=not-needed
export FOUNDRY_EMBED_BASE_URL=http://localhost:11434/v1
# Or switch provider entirely:
export ANTHROPIC_API_KEY=...
```

Change one line, rebuild, and **run the evaluation suite**. That comparison is
the whole point of keeping the knowledge layer separate from the model.

## Multi-tenancy

Each knowledge area declares its exposure:

```yaml
knowledge_area:
  visibility: public | private | enterprise
  tenant: acme
```

Public areas are served to everyone. Non-public areas are served only to
requests carrying a token mapped to their tenant:

```bash
export FOUNDRY_API_TOKENS='tok_abc:acme,tok_xyz:globex'
curl -H 'Authorization: Bearer tok_abc' https://…/k/kb-private-area
```

This is process-level filtering, not isolation. An enterprise tenant that needs
real isolation gets its own process and its own `var/` directory using the same
layout — no code changes.

## Backup and recovery

`var/` is derived and rebuildable from the repository plus the network, so the
backup that matters is git.

* **`knowledge_areas/`** — the asset. Sources, ontology, prompts, evaluation
  suites, version history. Back this up by pushing.
* **`var/<ka>/raw/`** — content-addressed source cache. Worth keeping on object
  storage (~$1/month): it makes rebuilds fast and preserves what a source said
  on the day it was retrieved, which matters when a URL changes under you.
* **`var/<ka>/kb.sqlite3`** — rebuild with `foundry build`.
* **`var/<ka>/baseline.json`** — the published report the regression gate
  compares against. Losing it does not break anything; the next build simply has
  nothing to regress against.

```bash
cp var/kb-001-battery-failure/kb.sqlite3 /backup/   # safe while serving: WAL mode
```

## Rollback

```bash
foundry version kb-001-battery-failure list
foundry version kb-001-battery-failure rollback            # previous published
foundry version kb-001-battery-failure rollback --version v0.3.1
```

Rollback is cheap because nothing was destroyed: every version's record stays in
the database and the index is shared.

## Cost at MVP

| Item | Monthly |
|---|---|
| VPS (1 vCPU, 2 GB) | $6–12 |
| Object storage for the raw cache | ~$1 |
| Domain | ~$1 |
| Inference | $0 with the local default; usage-scaled otherwise |
| CI (GitHub Actions on a public repo) | $0 |

Comfortably inside the target, with the headroom reserved for models rather than
for machines.
