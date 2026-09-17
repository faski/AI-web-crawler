"""Run the LLM parser over the whole gold standard and score every page.

Writes one JSON file per run, so results from different models and conditions
can be compared later. A run that is interrupted can be started again with the
same output file: pages already recorded are skipped, and nothing is paid for
twice.

Examples:
    # condition A: raw HTML, exactly as stored
    python scripts/run_llm_eval.py --out output/qwen9b_raw.json

    # condition B: same pages without <script>/<style>
    python scripts/run_llm_eval.py --strip-code --out output/qwen9b_lean.json

    # condition A again, cutting the pages that do not fit instead of
    # skipping them
    python scripts/run_llm_eval.py --chunk --out output/qwen9b_raw_chunked.json
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib.evaluation import calculate_content_metrics, calculate_token_level_metrics
from src.lib.llm import client
from src.lib.parsers.llm_parser import HtmlTooLongError, LlmParser, estimate_tokens

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from try_llm_parser import add_provider_flag, apply_provider, load_pages, strip_code

_print_lock = threading.Lock()


def log(message: str) -> None:
    """Print one line, without interleaving lines from other threads."""
    with _print_lock:
        print(message, flush=True)


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


def run_page(page: dict, strip: bool, keep_text: bool) -> dict:
    """Parse and score one page, recording a failure instead of raising."""
    html = strip_code(page["html_text"]) if strip else page["html_text"]
    record = {
        "url": page["url"],
        "domain": page["domain"],
        "input_tokens": estimate_tokens(html),
        "html_kb": round(len(page["html_text"]) / 1024),
    }
    started = time.monotonic()
    try:
        outcome = LlmParser().parse_html_detailed(page["url"], html)
        parsed_text = outcome.text
    except HtmlTooLongError as error:
        record.update(status="too_long", budget_tokens=error.budget_tokens)
        log(f"  [troppo lunga] {page['url']}  {record['input_tokens']:,} token")
        return record
    except Exception as error:  # noqa: BLE001 - one page must not stop the run
        record.update(status="error", error=f"{type(error).__name__}: {error}")
        log(f"  [errore] {page['url']}: {type(error).__name__}: {error}")
        return record

    record.update(
        status="ok",
        seconds=round(time.monotonic() - started, 1),
        chars=len(parsed_text),
        fragments=outcome.fragments,
        empty_fragments=outcome.empty_fragments,
        # The dollar figure is what the provider charged; the euro one is a
        # conversion at the rate recorded on the run, so a stale rate can be
        # recomputed instead of being baked into the data.
        call_costs_usd=list(outcome.call_costs_usd),
        cost_usd=outcome.cost_usd,
        cost_eur=outcome.cost_eur,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        # Share of the answer found in the page's visible text: a page that
        # describes something else entirely shows up here and nowhere else.
        grounded=outcome.grounded,
        **score(parsed_text, page["gold_text"]),
    )
    if keep_text:
        record["parsed_text"] = parsed_text
        # The reply as it arrived, before any cleaning. Only written with
        # --keep-text, since it roughly doubles the size of the results file.
        record["raw_responses"] = list(outcome.raw_responses)
    pieces = (
        f" {outcome.fragments} pezzi ({outcome.empty_fragments} vuoti)"
        if outcome.fragments > 1
        else ""
    )
    price = f" EUR {outcome.cost_eur:.4f}" if outcome.cost_usd else ""
    anchor = (
        f" ancoraggio {outcome.grounded:.0%}"
        if outcome.grounded is not None and outcome.grounded < 0.9
        else ""
    )
    log(
        f"  [ok] {page['url'][:62]:62} f1={record['f1']:.3f} "
        f"rec={record['recall']:.3f} {record['seconds']:.0f}s{pieces}{price}{anchor}"
    )
    return record


def summarise(records: list[dict]) -> None:
    """Print a per-domain table and the totals."""
    domains = sorted({r["domain"] for r in records})
    header = (
        f"{'dominio':18} {'ok':>5} {'lunghe':>7} {'err':>4} "
        f"{'prec':>6} {'rec':>6} {'f1':>6} {'cos':>6} {'exc':>6} {'sec':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for domain in domains:
        rows = [r for r in records if r["domain"] == domain]
        ok = [r for r in rows if r["status"] == "ok"]
        too_long = sum(1 for r in rows if r["status"] == "too_long")
        errors = sum(1 for r in rows if r["status"] == "error")
        if ok:
            mean = lambda key: sum(r[key] for r in ok) / len(ok)  # noqa: E731
            print(
                f"{domain:18} {len(ok):>5} {too_long:>7} {errors:>4} "
                f"{mean('precision'):>6.3f} {mean('recall'):>6.3f} "
                f"{mean('f1'):>6.3f} {mean('cosine'):>6.3f} "
                f"{mean('excess_ratio'):>6.3f} {mean('seconds'):>6.0f}"
            )
        else:
            print(f"{domain:18} {len(ok):>5} {too_long:>7} {errors:>4}"
                  f"{'  nessuna pagina riuscita':>40}")

    ok = [r for r in records if r["status"] == "ok"]
    spent = sum(r.get("cost_eur") or 0 for r in records)
    if spent:
        calls = sum(r.get("fragments") or 1 for r in ok)
        print(f"\ncosto totale EUR {spent:.4f} in {calls} chiamate "
              f"(EUR {spent / calls:.5f} a chiamata)")
    print("-" * len(header))
    if ok:
        mean = lambda key: sum(r[key] for r in ok) / len(ok)  # noqa: E731
        print(
            f"{'TOTALE':18} {len(ok):>5} "
            f"{sum(1 for r in records if r['status']=='too_long'):>7} "
            f"{sum(1 for r in records if r['status']=='error'):>4} "
            f"{mean('precision'):>6.3f} {mean('recall'):>6.3f} "
            f"{mean('f1'):>6.3f} {mean('cosine'):>6.3f} "
            f"{mean('excess_ratio'):>6.3f} {mean('seconds'):>6.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="file JSON dei risultati")
    add_provider_flag(parser)
    parser.add_argument("--strip-code", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--domain", help="limita a un dominio")
    parser.add_argument(
        "--urls",
        help="file con una URL per riga: esegue solo quelle pagine. Serve a "
             "provare una modifica su un sottoinsieme scelto invece di "
             "ripagare l'intero gold standard.",
    )
    parser.add_argument("--keep-text", action="store_true",
                        help="salva anche il markdown estratto (file piu' grande)")
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="divide in pezzi le pagine oltre il budget invece di saltarle",
    )
    args = parser.parse_args()
    apply_provider(args)
    # Set for this process only, like the provider, so a run says what it did
    # rather than inheriting it from whatever the environment happened to hold.
    os.environ["LLM_PARSER_CHUNKING"] = "1" if args.chunk else "0"

    pages = load_pages()
    if args.domain:
        pages = [p for p in pages if p["domain"] == args.domain]
    if args.urls:
        wanted = {
            line.strip()
            for line in open(args.urls, encoding="utf-8")
            if line.strip() and not line.startswith("#")
        }
        pages = [p for p in pages if p["url"] in wanted]
        missing = wanted - {p["url"] for p in pages}
        if missing:
            raise SystemExit(
                "URL non presenti nel gold standard:\n  "
                + "\n  ".join(sorted(missing))
            )

    done: dict[str, dict] = {}
    if os.path.exists(args.out):
        previous = json.load(open(args.out, encoding="utf-8"))
        # Only settled outcomes are kept. A page that failed on a network
        # error is left out so the resume retries it instead of freezing
        # the failure into the dataset.
        done = {
            r["url"]: r
            for r in previous["records"]
            if r["status"] in ("ok", "too_long")
        }
        failed = len(previous["records"]) - len(done)
        print(f"ripresa: {len(done)} pagine gia' fatte in {args.out}"
              + (f", {failed} da ritentare" if failed else ""))

    todo = [p for p in pages if p["url"] not in done]
    condition = "B (senza script/style)" if args.strip_code else "A (HTML grezzo)"
    if args.chunk:
        condition += " + chunking"
    print(f"modello   {client.get_model_name()}")
    print(f"condizione {condition}")
    print(f"budget    {os.environ.get('LLM_PARSER_CONTEXT_TOKENS', 128000)} token")
    print(f"chunking  {'attivo' if args.chunk else 'spento'}")
    print(f"da fare   {len(todo)} pagine su {len(pages)}, {args.workers} in parallelo\n")

    def save() -> list[dict]:
        """Write what has been paid for so far, and return it.

        Called after every page rather than once at the end. A run of forty
        pages costs real money and takes half an hour; writing only on the
        last line means a crash, a dropped connection or a Ctrl-C throws away
        everything already bought, and the resume logic above then has nothing
        to resume from.
        """
        records = [done[p["url"]] for p in pages if p["url"] in done]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": client.get_model_name(),
                    "provider": os.environ.get("LLM_PROVIDER", "ollama"),
                    "condition": condition,
                    "budget_tokens": int(
                        os.environ.get("LLM_PARSER_CONTEXT_TOKENS", 128000)
                    ),
                    "chunking": args.chunk,
                    "eur_per_usd": client.eur_per_usd(),
                    "records": records,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return records

    started = time.monotonic()
    records = save()
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for record in pool.map(
                lambda p: run_page(p, args.strip_code, args.keep_text), todo
            ):
                done[record["url"]] = record
                records = save()
    elapsed = time.monotonic() - started

    summarise(records)
    print(f"\ntempo totale {elapsed/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
