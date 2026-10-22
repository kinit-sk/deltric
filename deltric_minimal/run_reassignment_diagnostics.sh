#!/usr/bin/env bash
set -euo pipefail
DIAGNOSTICS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIAGNOSTICS_ROOT/.venv/bin/python" "$DIAGNOSTICS_ROOT/studies/reassignment_diagnostics_app.py" "$@"
