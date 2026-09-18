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

## Choosing models

Defaults are keyless and local, so nothing below is required.

```yaml
# knowledge_areas/<id>/manifest.yaml
specialist:
  llm:
    provider: anthropic            # or openai_compatible, or local_extractive
    model: claude-sonnet-5
retrieval:
  semantic:
    provider: sentence_transformers  # or lsa (default), tfidf, openai_compatible
```

Secrets come from the environment, never the manifest, so a knowledge area stays
publishable as plain text:

```bash
export ANTHROPIC_API_KEY=...
# or point at anything OpenAI-compatible, including a local Ollama:
export FOUNDRY_LLM_BASE_URL=http://localhost:11434/v1
export FOUNDRY_LLM_API_KEY=not-needed
export FOUNDRY_EMBED_BASE_URL=http://localhost:11434/v1
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
