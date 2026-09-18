# Knowledge Foundry — the whole loop, one target at a time.
#
# The default knowledge area is KA-001; override with KA=<id> on any target.

KA ?= kb-001-battery-failure
PY ?= python3
PORT ?= 8000

.PHONY: help install test build ingest index eval redteam publish rollback \
        serve stats site deploy nightly nightly-site doctor ollama-check up clean clean-all

# One command from a fresh clone to a working ask box.
up:
	./start.sh

help:
	@echo "Knowledge Foundry"
	@echo "  make up          install, build if needed, and serve — start here"
	@echo "  make install     install the package with dev extras"
	@echo "  make test        run the test suite"
	@echo "  make build       ingest + index + cut a draft version  (KA=$(KA))"
	@echo "  make eval        run the evaluation suites"
	@echo "  make redteam     run adversarial evaluation, promoting failures"
	@echo "  make publish     publish the build if the gate allows it"
	@echo "  make rollback    repoint at the previous published version"
	@echo "  make nightly     build + eval + redteam + publish (the §12 health check)"
	@echo "  make site        render the public site into ./site"
	@echo "  make deploy      render and publish ./site to the gh-pages branch"
	@echo "                   (API_BASE=<url> sets the instance its ask box calls)"
	@echo "  make nightly-site  nightly, then deploy the evidence pages"
	@echo "  make doctor      diagnose the model, corpus and server on this machine"
	@echo "  make serve       serve the public evidence pages on :$(PORT)"
	@echo "  make stats       print the knowledge area report as JSON"
	@echo ""
	@echo "  Override the knowledge area with KA=<id>, e.g. make eval KA=kb-002-industrial-fire-explosion"

install:
	$(PY) -m pip install -e ".[dev,web]"

test:
	$(PY) -m pytest -q

ingest:
	$(PY) -m foundry.cli ingest $(KA)

index:
	$(PY) -m foundry.cli index $(KA)

build:
	$(PY) -m foundry.cli build $(KA)

eval:
	$(PY) -m foundry.cli eval $(KA)

redteam:
	$(PY) -m foundry.cli redteam $(KA) --count 48 --promote

publish:
	$(PY) -m foundry.cli version $(KA) publish

rollback:
	$(PY) -m foundry.cli version $(KA) rollback

stats:
	$(PY) -m foundry.cli stats $(KA)

serve:
	$(PY) -m foundry.cli serve --host 0.0.0.0 --port $(PORT)

# The nightly health check from §12. `eval` is allowed to fail so that the
# red team and the publish gate still run and record why the build was held.
nightly:
	$(PY) -m foundry.cli build $(KA)
	-$(PY) -m foundry.cli eval $(KA)
	$(PY) -m foundry.cli redteam $(KA) --count 48 --promote
	$(PY) -m foundry.cli version $(KA) publish

# --- public site -------------------------------------------------------
# BASE_URL must match how GitHub Pages serves the repo. A project site lives
# at /<repo>/, so that is the default; set BASE_URL= for a user/apex domain.
BASE_URL ?= /knowledge-foundry
SITE ?= site
# The instance the published ask box calls. localhost works for anyone running
# the project themselves; point it at a tunnelled HTTPS endpoint to make the
# box work for readers who are not.
API_BASE ?= http://localhost:8000

site:
	$(PY) -m foundry.cli export $(SITE) --base-url $(BASE_URL) --api-base $(API_BASE)

deploy:
	$(PY) -m foundry.cli deploy --site $(SITE) --base-url $(BASE_URL) --api-base $(API_BASE)

# --- local model -------------------------------------------------------
# Every model call goes through Ollama at localhost:11434. This is a check,
# not a requirement: without it the system answers with the keyless local
# provider and records that it fell back.
# One command that checks every link in the chain on this machine: the Ollama
# daemon, the model, the corpus, the index and the server.
doctor:
	$(PY) -m foundry.cli doctor

ollama-check: doctor

# The full Mac-side loop: rebuild the knowledge, test it, publish the version,
# then publish the evidence pages. This is what the launchd agent runs.
nightly-site: nightly deploy

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

# Deletes the derived index. Everything is rebuildable from the repository
# plus the source cache, which is the point of keeping var/ out of git.
clean-all: clean
	rm -rf var
