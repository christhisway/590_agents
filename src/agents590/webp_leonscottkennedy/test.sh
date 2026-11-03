#!/usr/bin/env bash
# Run the shared tests plus Leon's own, which the shared pytest testpaths leave out.
# Paths given on the command line override testpaths, so pyproject.toml stays untouched.
# Usage: src/agents590/webp_leonscottkennedy/test.sh [extra pytest arguments]
set -euo pipefail
leon="$(cd "$(dirname "$0")" && pwd)"
cd "$leon/../../.."
uv run --locked python -m pytest tests "$leon/tests" "$@"
