"""Load the verdicts produced by run_self_check.py onto a run already imported.

The self-check is a second pass over an extraction that is already in the
database, so its results are written onto that run's pages instead of becoming
a run of their own. Put another way: the verdict is a column of the page, not
a page of its own.

Example:
    python scripts/import_self_check.py output/selfcheck_v4.json \
        --run-label "Qwen 3.5 9B"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib.db import apply_schema, close_pool, init_pool
from src.lib.db import llm_queries
from src.lib.db.connection import fetch_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="JSON prodotto da run_self_check.py")
    parser.add_argument("--run-label", required=True,
                        help="etichetta della run di parsing a cui attaccare i verdetti")
    args = parser.parse_args()

    data = json.load(open(args.file, encoding="utf-8"))

    init_pool()
    apply_schema()
    try:
        row = fetch_one("SELECT id FROM llm_runs WHERE label = ?", (args.run_label,))
        if row is None:
            sys.exit(f"nessuna run con etichetta '{args.run_label}'")
        run_id = row[0]

        results = data["results"]
        updated = llm_queries.attach_self_check(run_id, results)
        checked = [r for r in results if r["status"] == "ok"]
        # A page in the file that matched no row means the verdicts are being
        # attached to the wrong run, which would silently mislabel every
        # number on the comparison page.
        if updated != len(checked):
            print(f"ATTENZIONE: {len(checked) - updated} verdetti non hanno "
                  f"trovato la pagina corrispondente nella run '{args.run_label}'")
        # Coherence alone is the verdict; see get_self_check_summary.
        passed = sum(1 for r in checked if r["coherent"])
        print(f"run {run_id} '{args.run_label}': {updated} pagine aggiornate, "
              f"{passed} promosse, {len(checked) - passed} bocciate")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
