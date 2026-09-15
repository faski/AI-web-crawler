"""Run the LLM parser on one gold-standard page and show what it extracted.

Meant to be run by hand, to see the extraction and how it scores against the
gold text before launching anything expensive.

Examples:
    # what is in the corpus, and what fits the current budget
    python scripts/try_llm_parser.py --list

    # parse one page with the local model
    python scripts/try_llm_parser.py --domain www.xe.com --index 0

    # same page without <script>/<style>, which is what makes it fit a
    # small context window
    python scripts/try_llm_parser.py --domain www.cnbc.com --index 0 --strip-code
"""

import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib.evaluation import calculate_content_metrics, calculate_token_level_metrics
from src.lib.parsers.llm_parser import (
    HtmlTooLongError,
    LlmParser,
    estimate_tokens,
)
from src.lib.llm import client

GS_DIR = os.environ.get("GS_DIR", "/gs_data")

CODE_PATTERN = re.compile(r"(?is)<(script|style|noscript)\b.*?</\1>")
COMMENT_PATTERN = re.compile(r"(?s)<!--.*?-->")


def load_pages() -> list[dict]:
    """Return every gold-standard entry, sorted by domain then URL."""
    pages = []
    for path in sorted(glob.glob(os.path.join(GS_DIR, "*.json"))):
        pages.extend(json.load(open(path, encoding="utf-8")))
    return sorted(pages, key=lambda e: (e["domain"], e["url"]))


def strip_code(html: str) -> str:
    """Remove <script>, <style>, <noscript> and comments.

    This is the only reduction that cannot touch visible text, so it is the
    one honest way to make a page fit a smaller context window. Anything more
    aggressive would be doing the parser's job by hand.
    """
    return COMMENT_PATTERN.sub(" ", CODE_PATTERN.sub(" ", html))



def add_provider_flag(parser: argparse.ArgumentParser) -> None:
    """Add the flag that opts in to the paid remote provider."""
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="usa OpenRouter (A PAGAMENTO). Senza questo flag si usa Ollama locale.",
    )


def apply_provider(args) -> None:
    """Pin the provider for this process, ignoring whatever the env says.

    The environment is overwritten rather than read, so a LLM_PROVIDER left
    over in .env can never send a run to the paid provider by accident. Using
    the remote models has to be asked for, every time, on the command line.
    """
    os.environ["LLM_PROVIDER"] = "openrouter" if args.openrouter else "ollama"
    if args.openrouter:
        print("provider  OpenRouter - questa run CONSUMA CREDITO")
    else:
        print("provider  Ollama locale (gratis). Usa --openrouter per i modelli remoti.")


def print_listing(pages: list[dict], budget: int) -> None:
    """Print every page with its size and whether it fits the budget."""
    print(f"budget di contesto: {budget:,} token\n")
    header = f"{'#':>3}  {'dominio':18} {'KB':>6} {'token':>8} {'entra':>6}  {'senza code':>10} {'entra':>6}"
    print(header)
    print("-" * len(header))
    by_domain: dict[str, int] = {}
    for page in pages:
        index = by_domain.get(page["domain"], 0)
        by_domain[page["domain"]] = index + 1
        raw = estimate_tokens(page["html_text"])
        lean = estimate_tokens(strip_code(page["html_text"]))
        print(
            f"{index:>3}  {page['domain']:18} {len(page['html_text'])/1024:6.0f} "
            f"{raw:8,} {'si' if raw <= budget else 'NO':>6}  "
            f"{lean:10,} {'si' if lean <= budget else 'NO':>6}"
        )


def select_page(pages: list[dict], args) -> dict:
    """Return the page named by --url, or by --domain plus --index."""
    if args.url:
        for page in pages:
            if page["url"] == args.url:
                return page
        sys.exit(f"URL non trovato nel gold standard: {args.url}")
    same_domain = [p for p in pages if p["domain"] == args.domain]
    if not same_domain:
        sys.exit(f"Dominio sconosciuto: {args.domain}")
    if args.index >= len(same_domain):
        sys.exit(f"{args.domain} ha {len(same_domain)} pagine (indici 0-{len(same_domain)-1})")
    return same_domain[args.index]


def print_scores(parsed_text: str, gold_text: str) -> None:
    """Print the base-project metrics of the extraction against the gold text."""
    token_metrics = calculate_token_level_metrics(parsed_text, gold_text)
    content_metrics = calculate_content_metrics(parsed_text, gold_text)
    print("--- punteggi contro il gold standard ---")
    print(f"  precision     {token_metrics.precision:.3f}")
    print(f"  recall        {token_metrics.recall:.3f}")
    print(f"  f1            {token_metrics.f1:.3f}")
    print(f"  cosine        {content_metrics.cosine:.3f}")
    print(f"  jaccard       {content_metrics.jaccard:.3f}")
    print(f"  excess_ratio  {content_metrics.excess_ratio:.3f}")
    print(
        f"  token estratti {token_metrics.extracted_count}, "
        f"nel gold {token_metrics.sample_count}, "
        f"in comune {token_metrics.intersection_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="elenca le pagine e esci")
    add_provider_flag(parser)
    parser.add_argument("--domain", default="www.xe.com")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--url", help="URL esatto, al posto di --domain/--index")
    parser.add_argument(
        "--strip-code",
        action="store_true",
        help="togli <script>/<style>/commenti prima di inviare",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="chiedi al modello se la sua estrazione e' completa e coerente",
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="dividi in pezzi la pagina se non entra, invece di rinunciare",
    )
    parser.add_argument(
        "--chars", type=int, default=2000, help="caratteri di output da stampare"
    )
    args = parser.parse_args()
    apply_provider(args)
    os.environ["LLM_PARSER_CHUNKING"] = "1" if args.chunk else "0"

    pages = load_pages()
    budget = int(os.environ.get("LLM_PARSER_CONTEXT_TOKENS", 128_000))
    if args.list:
        print_listing(pages, budget)
        return

    page = select_page(pages, args)
    html = strip_code(page["html_text"]) if args.strip_code else page["html_text"]

    print(f"url       {page['url']}")
    print(f"modello   {client.get_model_name()}")
    print(f"num_ctx   {os.environ.get('OLLAMA_NUM_CTX', '(default di Ollama)')}")
    print(f"HTML      {len(page['html_text'])/1024:.0f} KB grezzo", end="")
    if args.strip_code:
        print(f" -> {len(html)/1024:.0f} KB senza script/style", end="")
    print(f"  (~{estimate_tokens(html):,} token, budget {budget:,})")
    print(f"chunking  {'attivo' if args.chunk else 'spento'}")
    print()

    llm_parser = LlmParser()
    started = time.monotonic()
    try:
        outcome = llm_parser.parse_html_detailed(page["url"], html)
        parsed_text = outcome.text
    except HtmlTooLongError as error:
        print(f"NON ENTRA: {error}")
        print("Prova --chunk per leggerla a pezzi, oppure --strip-code, "
              "oppure alza LLM_PARSER_CONTEXT_TOKENS se il modello ha una "
              "finestra piu' grande.")
        return
    elapsed = time.monotonic() - started

    if outcome.fragments > 1:
        print(f"--- letta in {outcome.fragments} pezzi, "
              f"{outcome.empty_fragments} senza contenuto ---")
    print(f"--- estratto in {elapsed:.0f}s, {len(parsed_text)} caratteri ---")
    print(parsed_text[: args.chars])
    if len(parsed_text) > args.chars:
        print(f"\n[... altri {len(parsed_text) - args.chars} caratteri, alza --chars]")
    print()
    print_scores(parsed_text, page["gold_text"])

    if args.self_check:
        print()
        started = time.monotonic()
        verdict = llm_parser.self_check(page["url"], html, parsed_text)
        print(f"--- autovalutazione senza gold standard ({time.monotonic()-started:.0f}s) ---")
        print(f"  completa  {verdict.complete}")
        print(f"  coerente  {verdict.coherent}")
        print(f"  note      {verdict.notes}")


if __name__ == "__main__":
    main()
