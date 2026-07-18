"""One-shot bootstrap of the gold standard tables from gs_data/*_gs.json.

Runs at FastAPI startup. If ``web_resources`` is already populated (either
from a previous boot's volume or because the test suite has seeded it),
nothing happens: this keeps the loader idempotent.
"""

import json
from pathlib import Path

from . import queries


def _gold_standard_data_directory() -> Path:
    """Locate the gs_data/ directory in both Docker and local-dev layouts."""
    docker_path = Path("/gs_data")
    if docker_path.exists():
        return docker_path
    # db/ → lib/ → src/ → backend/ → project root
    return Path(__file__).resolve().parents[4] / "gs_data"


def _load_all_json_entries() -> list[dict]:
    """Load every entry from all gs_data/*_gs.json files."""
    entries: list[dict] = []
    for json_file in sorted(_gold_standard_data_directory().glob("*_gs.json")):
        content = json_file.read_text(encoding="utf-8").strip()
        if content:
            entries.extend(json.loads(content))
    return entries


def populate_if_empty() -> int:
    """Seed the tables from gs_data/ if empty. Returns the number of new entries inserted."""
    if queries.count_resources() > 0:
        return 0
    entries = _load_all_json_entries()
    queries.add_entries(entries)
    return len(entries)
