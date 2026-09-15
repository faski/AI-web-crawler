"""Report how often the reply cleaning fires, and on what.

``_clean_markdown`` removes a reasoning block and a code fence wrapping the
whole reply. The results files store the Markdown *after* that, so the
cleaning leaves no trace and there was no way to tell whether it ever ran.
Runs made with ``--keep-text`` now also store the reply as it arrived, and
this script replays the cleaning over those replies and says what it changed.

The point is to decide with numbers whether the cleaning should exist at all:
the extraction pipeline is meant to be raw HTML in, Markdown out, and every
step that touches the model's answer has to earn its place.

    python scripts/audit_cleaning.py output/v5_run.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib.parsers.llm_parser import _clean_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="uno o piu' file JSON di run")
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="quante risposte modificate mostrare per file (default: 5)",
    )
    args = parser.parse_args()

    for path in args.files:
        data = json.load(open(path, encoding="utf-8"))
        replies = [
            (record["url"], reply)
            for record in data["records"]
            for reply in record.get("raw_responses") or []
        ]
        if not replies:
            print(f"{path}: nessuna risposta grezza salvata "
                  f"(la run e' stata fatta senza --keep-text, o prima che "
                  f"venissero conservate)")
            continue

        changed = [
            (url, reply)
            for url, reply in replies
            if _clean_markdown(reply).strip() != reply.strip()
        ]
        fenced = [r for _, r in changed if r.strip().startswith("```")]
        thought = [r for _, r in changed if "<think>" in r.lower()]

        print(f"{path}: {len(replies)} risposte, {len(changed)} modificate dalla pulizia")
        print(f"  blocco di codice avvolgente: {len(fenced)}")
        print(f"  blocco <think>:              {len(thought)}")
        for url, reply in changed[: args.show]:
            removed = len(reply.strip()) - len(_clean_markdown(reply).strip())
            print(f"    -{removed:>6} caratteri  {url[:58]}")
            print(f"            inizio: {reply.strip()[:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
