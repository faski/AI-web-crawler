"""Write the runs in the database out to runs_data/, so a clone gets them too.

The gold standard already travels with the repository and is loaded on first
boot; the results measured on it did not, so anyone cloning the project found
an empty comparison page. This writes the same runs to files small enough to
track: the raw replies and the HTML are left out, only what the pages show is
kept.

Run it after importing a run that should ship with the project.

Example:
    python scripts/export_run_seed.py 4 9 11
    python scripts/export_run_seed.py --all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib.db import apply_schema, close_pool, init_pool, llm_queries
from src.lib.db.connection import fetch_all


def slug(label: str) -> str:
    """Turn a run label into a file name."""
    kept = [c.lower() if c.isalnum() else "_" for c in label]
    return "_".join("".join(kept).split("_")) or "run"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", type=int, help="id delle run")
    parser.add_argument("--all", action="store_true", help="tutte le run")
    parser.add_argument("--out", default="/runs_data", help="cartella di uscita")
    args = parser.parse_args()

    init_pool()
    apply_schema()
    try:
        ids = args.run_ids
        if args.all:
            ids = [r[0] for r in fetch_all("SELECT id FROM llm_runs ORDER BY id", ())]
        if not ids:
            sys.exit("indica almeno un id, oppure --all")

        os.makedirs(args.out, exist_ok=True)
        for run_id in ids:
            data = llm_queries.export_run(run_id)
            if data is None:
                print(f"run {run_id}: non esiste, saltata")
                continue
            path = os.path.join(args.out, f"{slug(data['label'])}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=1)
            size = os.path.getsize(path) / 1024
            checked = sum(1 for p in data["pages"] if p.get("check_coherent") is not None)
            print(f"run {run_id} '{data['label']}': {len(data['pages'])} pagine"
                  f"{f', {checked} con autovalutazione' if checked else ''}"
                  f" -> {path} ({size:.0f} KB)")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
