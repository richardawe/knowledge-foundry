#!/usr/bin/env bash
# Knowledge Foundry: fresh clone to a working ask box, in one command.
#
#   ./start.sh
#
# Idempotent. Safe to re-run: it installs only what is missing, builds only if
# the corpus is absent, and always ends by serving.
#
# Note the two processes involved, because they are easy to conflate:
#   * ollama   -- the model, on :11434. Shared with anything else you run.
#   * foundry  -- this app, on :8000. Uses Ollama; is not Ollama.
# A running Ollama does not mean this is running.

set -euo pipefail

PORT="${PORT:-8000}"
KA="${KA:-kb-001-battery-failure}"
PY="${PY:-python3}"
cd "$(dirname "$0")"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }

say "1/4  Checking Python"
if ! command -v "$PY" >/dev/null 2>&1; then
  note "python3 not found. Install it (brew install python@3.12) and re-run."
  exit 1
fi
version=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
note "python $version at $(command -v "$PY")"
note "(override with: PY=/opt/homebrew/bin/python3.12 ./start.sh)"
"$PY" - <<'EOF' || { echo "  Python 3.10+ is required."; exit 1; }
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
EOF

say "2/4  Installing"
# Always into a virtualenv. Homebrew and system Pythons are marked
# externally-managed (PEP 668), so a plain `pip install -e .` against them
# fails outright -- which is the first wall anyone hits on a Mac. A venv
# sidesteps it and leaves the system Python untouched.
VENV=".venv"
if [ ! -x "$VENV/bin/python" ]; then
  note "creating $VENV"
  "$PY" -m venv "$VENV"
fi
PY="$PWD/$VENV/bin/python"
if "$PY" -c 'import foundry, jinja2' >/dev/null 2>&1; then
  note "already installed in $VENV"
else
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -e ".[dev,web]"
  note "installed knowledge-foundry into $VENV"
fi

say "3/4  Checking the model and the corpus"
"$PY" -m foundry.cli doctor --port "$PORT" || true

if ! "$PY" - "$KA" <<'EOF'
import sys
from foundry.storage import Store
with Store(sys.argv[1]) as store:
    raise SystemExit(0 if store.counts()["chunks"] else 1)
EOF
then
  note "corpus is empty — building it now (fetches ~30 sources, about a minute)"
  "$PY" -m foundry.cli build "$KA"
  "$PY" -m foundry.cli eval "$KA" || true
fi

say "4/4  Serving"
note "Ask here:        http://localhost:$PORT/k/$KA/ask"
note "Published page:  https://richardawe.github.io/knowledge-foundry/k/$KA/ask"
note "                 (reload it — it will find this instance)"
note ""
note "Ctrl-C to stop."
exec "$PY" -m foundry.cli serve --host 127.0.0.1 --port "$PORT"
