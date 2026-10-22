#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$LAB_ROOT/../deltric_minimal/.venv/bin/python" "$LAB_ROOT/server.py" "$@"
