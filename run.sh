#!/bin/sh
set -eu
release_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$release_root/.venv/bin/python" ]; then
    python3 -m venv "$release_root/.venv"
fi
"$release_root/.venv/bin/python" -m pip install -r "$release_root/requirements.txt" -e "$release_root"
exec "$release_root/.venv/bin/python" "$release_root/reproduce.py" "$@"
