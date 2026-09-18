# Knowledge Foundry — the whole loop, one target at a time.
#
# The default knowledge area is KA-001; override with KA=<id> on any target.

KA ?= kb-001-battery-failure
PY ?= python3
PORT ?= 8000

.PHONY: help install test build ingest index eval redteam publish rollback \
        serve stats nightly clean clean-all

help:
	@echo "Knowledge Foundry"
	@echo "  make install     install the package with dev extras"
	@echo "  make test        run the test suite"
	@echo "  make build       ingest + index + cut a draft version  (KA=$(KA))"
	@echo "  make eval        run the evaluation suites"
	@echo "  make redteam     run adversarial evaluation, promoting failures"
	@echo "  make publish     publish the build if the gate allows it"
	@echo "  make rollback    repoint at the previous published version"
	@echo "  make nightly     build + eval + redteam + publish (the §12 health check)"
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

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

# Deletes the derived index. Everything is rebuildable from the repository
# plus the source cache, which is the point of keeping var/ out of git.
clean-all: clean
	rm -rf var
