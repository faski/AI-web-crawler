"""Check that splitting a page loses nothing and respects the budget.

Run this after changing ``html_chunker`` or upgrading BeautifulSoup. The
splitter relies on the serialiser reproducing a tag's opening markup exactly;
if an upgrade changes that, the fragments stop adding up to the page and this
script says so instead of the damage showing up as a mysterious drop in
recall three runs later.

    python scripts/verify_chunker.py
    python scripts/verify_chunker.py --budgets 60000 100000 180000
"""

import argparse
import os
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib.parsers.html_chunker import (
    PageContext,
    read_page_context,
    split_html,
    stitch,
)
# Private on purpose - it is an implementation detail of the parser - but
# the rule it encodes was found by a real run and has to stay pinned.
from src.lib.parsers.llm_parser import _without_repeated_title, estimate_tokens

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from try_llm_parser import load_pages


def check_page(page: dict, budget: int) -> tuple[int, list[str]]:
    """Return how many fragments the page took, and the problems found."""
    html = page["html_text"]
    fragments = split_html(html, budget, estimate_tokens)
    problems = []

    # A page that fits is handed back untouched, so it must equal the HTML as
    # it arrived. A page that was split went through the parser, so the
    # fragments have to add up to what the parser produced instead.
    expected = html if len(fragments) == 1 else str(BeautifulSoup(html, "lxml"))
    if "".join(fragments) != expected:
        problems.append("the fragments do not add up to the page")

    oversized = [n for n, f in enumerate(fragments) if estimate_tokens(f) > budget]
    if oversized:
        problems.append(f"fragments over budget: {oversized}")

    if len(fragments) > 1 and not read_page_context(html).title:
        problems.append("split page with no <title> to carry into the fragments")

    return len(fragments), problems


# The seams are where chunking can quietly corrupt a page: a block that
# straddles a cut comes back from both sides and would be written twice. These
# cases pin the repair down, including the one it must NOT make - a heading
# that genuinely appears twice, far apart, is not a seam.
STITCH_CASES = (
    ("risposte vuote scartate", ["# Titolo", "", "   ", "Coda"], "# Titolo\n\nCoda"),
    ("nessuna sovrapposizione", ["Uno", "Due"], "Uno\n\nDue"),
    (
        "sovrapposizione di una riga",
        ["Paragrafo uno.\n\n## Sezione", "## Sezione\n\nParagrafo due."],
        "Paragrafo uno.\n\n## Sezione\n\nParagrafo due.",
    ),
    (
        "sovrapposizione di tre righe",
        ["a\n\n## S\n\nb", "## S\n\nb\n\nc"],
        "a\n\n## S\n\nb\n\nc",
    ),
    (
        "spaziatura e maiuscole diverse",
        ["testo\n\n##   Sezione", "## SEZIONE\n\nresto"],
        "testo\n\n##   Sezione\n\nresto",
    ),
    ("tutte vuote", ["", "  "], ""),
    (
        "ripetizione lontana, non una giuntura",
        ["## S\n\nx\n\ny", "z\n\n## S"],
        "## S\n\nx\n\ny\n\nz\n\n## S",
    ),
)


# A page whose article spans two fragments came back with its "# Titolo"
# written again in the middle: the model reopened the document on the second
# fragment. The seam repair cannot see it, because it does not repeat the
# previous fragment's last lines. These cases pin the removal - and pin what
# must NOT be removed, which is any heading that is not the page's own title.
TITLE_CONTEXT = PageContext(title="Prova del chunking", heading="Prova   del CHUNKING")
TITLE_CASES = (
    ("titolo ripetuto identico", "# Prova del chunking\n\nParagrafo 12.", "Paragrafo 12."),
    ("titolo ripetuto, altra spaziatura", "#   prova   del  chunking  \n\nX.", "X."),
    ("corrisponde all'h1 e non al title", "# Prova del CHUNKING\n\nY.", "Y."),
    ("intestazione di sezione vera", "# Altra sezione\n\nZ.", "# Altra sezione\n\nZ."),
    ("h2 non toccato", "## Prova del chunking\n\nW.", "## Prova del chunking\n\nW."),
    ("non in prima riga", "Testo.\n\n# Prova del chunking", "Testo.\n\n# Prova del chunking"),
    ("risposta vuota", "", ""),
)


def check_repeated_title() -> int:
    """Return how many of the repeated-title cases came out wrong."""
    failures = 0
    for name, given, expected in TITLE_CASES:
        produced = _without_repeated_title(given, TITLE_CONTEXT)
        if produced != expected:
            failures += 1
            print(f"FALLITO  titolo ripetuto: {name}")
            print(f"         atteso   {expected!r}")
            print(f"         ottenuto {produced!r}")
    print(f"titolo ripetuto: {len(TITLE_CASES) - failures}/{len(TITLE_CASES)} casi superati")
    return failures


def check_stitch() -> int:
    """Return how many of the seam cases came out wrong."""
    failures = 0
    for name, parts, expected in STITCH_CASES:
        produced = stitch(parts)
        if produced != expected:
            failures += 1
            print(f"FALLITO  ricucitura: {name}")
            print(f"         atteso   {expected!r}")
            print(f"         ottenuto {produced!r}")
    print(f"ricucitura: {len(STITCH_CASES) - failures}/{len(STITCH_CASES)} casi superati")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[60_000, 100_000, 180_000, 262_144],
        help="input budgets, in tokens, to try (default: 60k 100k 180k 262k)",
    )
    args = parser.parse_args()

    failures = check_stitch() + check_repeated_title()
    pages = load_pages()

    for budget in args.budgets:
        split_pages, fragments_used = 0, 0
        for page in pages:
            count, problems = check_page(page, budget)
            if count > 1:
                split_pages += 1
                fragments_used += count
            for problem in problems:
                failures += 1
                print(f"FALLITO  budget {budget}  {page['url']}\n         {problem}")
        extra = fragments_used - split_pages
        print(
            f"budget {budget:>7}: {split_pages}/{len(pages)} pagine divise, "
            f"{fragments_used} frammenti, {extra} chiamate in piu"
        )

    if failures:
        print(f"\n{failures} problemi")
        return 1
    print("\nnessuna perdita: i frammenti ricostruiscono la pagina")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
