#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m scholar_mind.db.init_db
PYTHONPATH=src python3 - <<'PY'
from scholar_mind.app import get_container
container = get_container()
print("Seeded papers:", len(container.paper_repository.all_papers()))
PY
