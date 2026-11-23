#!/bin/zsh
set -eu
cd "${0:A:h}/.."
mkdir -p runs
exec caffeinate -i .venv/bin/python -u scripts/continuous_replay.py "$@"
