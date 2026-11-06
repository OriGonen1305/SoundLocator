#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt' >&2
    exit 1
fi
if [[ "${1:-}" == "--test" ]]; then
    exec .venv/bin/python -m unittest discover -s tests -v
fi
if [[ $# -eq 0 ]]; then
    exec .venv/bin/python -m sound_locator demo
fi
exec .venv/bin/python -m sound_locator "$@"
