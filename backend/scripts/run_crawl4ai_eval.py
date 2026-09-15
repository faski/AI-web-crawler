"""Run the Crawl4AI pipeline over the gold standard and score every page.

This is the baseline the project brief asks the LLM parser to be compared
against. It writes the same JSON shape as ``run_llm_eval.py``, so the two are
imported into the same tables and shown side by side by the comparison page.

The pipeline is exactly the one the application uses: the stored HTML goes
through Crawl4AI's HTML-to-markdown step and then through the domain parser,
which is what ``/parse`` does for a cached page. Both approaches therefore
start from the same bytes.

Example:
    python scripts/run_crawl4ai_eval.py --out output/crawl4ai.json
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.lib.crawling.crawler import close_crawler, fetch_page_from_html
from src.lib.evaluation import calculate_content_metrics, calculate_token_level_metrics
from src.lib.parsers import get_parser_for_url
from try_llm_parser import load_pages

CONDITION = "Crawl4AI + parser di dominio"


def score(parsed_text: str, gold_text: str) -> dict:
    """Return the base-project metrics of one extraction."""
    token_metrics = calculate_token_level_metrics(parsed_text, gold_text)
    content_metrics = calculate_content_metrics(parsed_text, gold_text)
    return {
        "precision": token_metrics.precision,
        "recall": token_metrics.recall,
        "f1": token_metrics.f1,
        "cosine": content_metrics.cosine,
        "jaccard": content_metrics.jaccard,
        "excess_ratio": content_metrics.excess_ratio,
        "extracted_count": token_metrics.extracted_count,
        "sample_count": token_metrics.sample_count,
    }


async def run_page(page: dict, keep_text: bool) -> dict:
    """Convert one stored page with Crawl4AI, parse it, and score it.

    Two clocks are taken. ``seconds`` is wall time, comparable with the LLM
    runs. ``cpu_seconds`` is processor time actually burnt in this process,
    which is the resource this pipeline spends and a remote LLM does not: it
    is what makes the two costs impossible to compare on one axis.
    """
    record = {
        "url": page["url"],
        "domain": page["domain"],
        "html_kb": round(len(page["html_text"]) / 1024),
        # Crawl4AI consumes no model tokens at all: the field is left empty
        # rather than zero, so the page shows "—" and not a misleading 0.
        "input_tokens": None,
    }
    wall_start = time.monotonic()
    cpu_start = time.process_time()
    try:
        converted = await fetch_page_from_html(page["url"], page["html_text"])
        parsed_text = get_parser_for_url(page["url"]).parse(
            page["url"], converted.markdown_text
        )
    except Exception as error:  # noqa: BLE001 - one page must not stop the run
        record.update(status="error", error=f"{type(error).__name__}: {error}")
        print(f"  [errore] {page['url']}: {type(error).__name__}: {error}", flush=True)
        return record

    record.update(
        status="ok",
        seconds=round(time.monotonic() - wall_start, 2),
        cpu_seconds=round(time.process_time() - cpu_start, 2),
        chars=len(parsed_text),
        **score(parsed_text, page["gold_text"]),
    )
    if keep_text:
        record["parsed_text"] = parsed_text
    print(
        f"  [ok] {page['url'][:62]:62} f1={record['f1']:.3f} "
        f"rec={record['recall']:.3f} {record['seconds']:.1f}s",
        flush=True,
    )
    return record


def summarise(records: list[dict]) -> None:
    """Print a per-domain table and the totals."""
    domains = sorted({r["domain"] for r in records})
    header = (
        f"{'dominio':18} {'ok':>4} {'err':>4} {'prec':>6} {'rec':>6} "
        f"{'f1':>6} {'cos':>6} {'exc':>6} {'sec':>7} {'cpu':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for domain in [*domains, None]:
        rows = [r for r in records if domain is None or r["domain"] == domain]
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            continue
        mean = lambda key: sum(r[key] for r in ok) / len(ok)  # noqa: E731
        print(
            f"{(domain or 'TOTALE'):18} {len(ok):>4} "
            f"{sum(1 for r in rows if r['status']=='error'):>4} "
            f"{mean('precision'):>6.3f} {mean('recall'):>6.3f} {mean('f1'):>6.3f} "
            f"{mean('cosine'):>6.3f} {mean('excess_ratio'):>6.3f} "
            f"{mean('seconds'):>7.2f} {mean('cpu_seconds'):>7.2f}"
        )
        if domain is domains[-1]:
            print("-" * len(header))


async def main_async(args) -> None:
    pages = load_pages()
    if args.domain:
        pages = [p for p in pages if p["domain"] == args.domain]
    print(f"pipeline  {CONDITION}")
    print(f"da fare   {len(pages)} pagine, in sequenza\n")

    started = time.monotonic()
    records = []
    # Sequential on purpose: the numbers being measured are time and CPU, and
    # running pages in parallel would make both meaningless.
    for page in pages:
        records.append(await run_page(page, args.keep_text))
    elapsed = time.monotonic() - started

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "crawl4ai",
                "provider": "locale",
                "condition": CONDITION,
                "budget_tokens": 0,
                "records": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    summarise(records)
    print(f"\ntempo totale {elapsed/60:.1f} min -> {args.out}")
    await close_crawler()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="file JSON dei risultati")
    parser.add_argument("--domain", help="limita a un dominio")
    parser.add_argument("--keep-text", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
