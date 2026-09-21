"""One-shot bootstrap of the evaluation runs from runs_data/*.json.

Runs at FastAPI startup, next to the gold standard loader. Without it a fresh
clone has the forty pages but none of the results measured on them: the
comparison page is empty, and the LLM parser finds no stored extraction to
show. The runs cost real money to produce and cannot be recomputed by
whoever is only reading the project.

If ``llm_runs`` already holds anything, nothing happens. The files are a
starting point, not a source of truth: a run imported later must never be
overwritten by the seed on the next boot.
"""

import json
from pathlib import Path

from . import llm_queries
from .connection import fetch_one


def _runs_data_directory() -> Path:
    """Locate the runs_data/ directory in both Docker and local-dev layouts."""
    docker_path = Path("/runs_data")
    if docker_path.exists():
        return docker_path
    # db/ -> lib/ -> src/ -> backend/ -> project root
    return Path(__file__).resolve().parents[4] / "runs_data"


def populate_if_empty() -> int:
    """Seed the run tables from runs_data/ if empty. Returns how many runs were added."""
    row = fetch_one("SELECT COUNT(*) FROM llm_runs", ())
    if row and row[0]:
        return 0

    added = 0
    directory = _runs_data_directory()
    if not directory.exists():
        return 0
    # Sorted so the runs land in a stable order, which is also the order the
    # comparison page falls back to when nobody has picked two to compare.
    for json_file in sorted(directory.glob("*.json")):
        content = json_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        llm_queries.save_seed_run(json.loads(content))
        added += 1
    return added
