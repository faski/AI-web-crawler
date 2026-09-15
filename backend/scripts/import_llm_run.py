"""Load the JSON produced by run_llm_eval.py into the database.

The comparison pages read from the database only, so a run has to be imported
once before it can be looked at. Importing the same file again replaces the
run instead of duplicating it.

Example:
    python scripts/import_llm_run.py output/v2_qwen9b_raw.json --label "9B condizione A"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib.db import apply_schema, close_pool, init_pool
from src.lib.db import llm_queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="uno o piu' file JSON di run")
    parser.add_argument(
        "--label",
        help="nome della run (default: il nome del file). Con piu' file viene ignorato.",
    )
    args = parser.parse_args()

    init_pool()
    apply_schema()
    try:
        for path in args.files:
            data = json.load(open(path, encoding="utf-8"))
            label = (
                args.label
                if args.label and len(args.files) == 1
                else os.path.splitext(os.path.basename(path))[0]
            )
            run_id = llm_queries.save_run(
                label=label,
                model_name=data["model"],
                provider=data["provider"],
                condition_name=data["condition"],
                budget_tokens=data["budget_tokens"],
                records=data["records"],
                eur_per_usd=data.get("eur_per_usd"),
            )
            ok = sum(1 for r in data["records"] if r["status"] == "ok")
            print(
                f"importata '{label}' come run {run_id}: "
                f"{len(data['records'])} pagine, {ok} riuscite"
            )
    finally:
        close_pool()


if __name__ == "__main__":
    main()
