#!/bin/sh
# Avvia QVex (motore e interfaccia web) su http://127.0.0.1:8080
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Creo l'ambiente Python (solo la prima volta)..."
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
fi
exec .venv/bin/python -m engine.server "$@"
