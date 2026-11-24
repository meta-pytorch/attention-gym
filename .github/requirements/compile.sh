#!/usr/bin/env bash
# Pin the CI tool requirements with hashes for the GitHub-hosted ubuntu-latest runners.
# Each run re-resolves to the newest releases outside a three-day cooldown, giving PyPI time
# to pull malware.
set -euo pipefail
cd "$(dirname "$0")"
for name in release modal; do
  uv pip compile "$name.in" -o "$name.txt" \
    --upgrade --generate-hashes --only-binary :all: --exclude-newer "3 days" \
    --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
    --index-url https://pypi.org/simple --no-config --quiet
done
