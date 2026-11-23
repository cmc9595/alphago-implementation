#!/bin/zsh
set -eu
cd "${0:A:h}/.."
mkdir -p runs
# Same command resumes completed generation and persisted training checkpoints.
exec caffeinate -i .venv/bin/python -u scripts/replay_loop.py "$@"
