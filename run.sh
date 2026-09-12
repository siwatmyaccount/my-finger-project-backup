#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR/src_code"
exec "$PROJECT_DIR/.venv/bin/python" -u -X faulthandler main.py
