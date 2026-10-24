# Template: one knowledge area per company

A hand-built knowledge area is a directory somebody wrote. This is a directory
that writes knowledge areas: `foundry instantiate company --param domain=... --param name=...`
renders it into `companies/co-<domain>/`, and `foundry collect` fills in the
source register by crawling the company's site and looking it up elsewhere.

Instances are not committed. They hold third-party findings about a named
company, which are shown only to that company once it has claimed its page.

| File               | What it is                                                    |
|--------------------|---------------------------------------------------------------|
| `manifest.yaml`    | Jinja: the knowledge area, scoped to one company              |
| `collectors.yaml`  | Where sources are found, by tier, with licence and authority  |
| `evaluation/core.yaml` | Jinja: questions every company is asked, with graders     |
| `prompts/system.md` | Instructions to a model, when one is used                    |

Four tiers of voice, deliberately kept apart:

* `official_record` -- public registers. Authority 5 when the company's own site
  corroborates the match, 3 when only the name matched.
* `self_published` -- the company's own site. What it claims.
* `press` -- reporting about it.
* `community` / `review` -- what its users say.
