#!/usr/bin/env bash
# The model half of the pipeline, for the machine that has Ollama.
#
#   ./worker.sh
#
# Pull the prompts GitHub Actions queued, answer them with the local model,
# push the replies back. Actions validates and scores them and republishes the
# site; nothing here decides whether an answer is any good.
#
# This is the ONLY part of the system that calls a model.

set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${LIMIT:-0}"          # 0 = answer everything pending
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "Syncing"
git pull --rebase --autostash origin "$BRANCH"

say "Checking the model"
"$PY" -m foundry.cli doctor --json >/dev/null 2>&1 || true
"$PY" -m foundry.cli queue

pending=$("$PY" -m foundry.cli queue --json | "$PY" -c \
  'import json,sys; print(sum(q["pending"] for q in json.load(sys.stdin)["queues"]))')
if [ "$pending" -eq 0 ]; then
  say "Nothing queued. Done."
  exit 0
fi

say "Answering $pending prompt(s) with the local model"
if [ "$LIMIT" -gt 0 ]; then
  "$PY" -m foundry.cli worker --limit "$LIMIT" --name "$(hostname -s)"
else
  "$PY" -m foundry.cli worker --name "$(hostname -s)"
fi

say "Pushing the answers back"
git add inference
if git diff --cached --quiet; then
  echo "  no new answers to push"
else
  git -c user.name="knowledge-foundry-worker" \
      -c user.email="noreply@anthropic.com" \
      commit -m "Worker: answers from $(hostname -s) via the local model"
  git pull --rebase --autostash origin "$BRANCH"
  git push origin "HEAD:$BRANCH"
  echo "  pushed; GitHub Actions will validate, score and republish"
fi
