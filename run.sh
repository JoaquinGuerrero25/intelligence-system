#!/usr/bin/env bash
# Levanta todo el sistema (API + Redis Stack + Phoenix) con una sola instrucción:  ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> Se creó .env desde .env.example (LLM_PROVIDER=mock: no hacen falta API keys)."
fi

echo ">> API:     http://localhost:8000/docs"
echo ">> Phoenix: http://localhost:6006"
exec docker compose up --build "$@"
