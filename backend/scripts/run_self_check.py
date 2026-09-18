"""Ask the model to judge its own extractions, and see whether it is right.

This is the brief's second requirement: give the raw HTML back to the LLM in a
structured prompt and have it say whether the text it extracted is complete and
coherent. No gold standard is involved in the question - that is the point of a
reference-free check - but the gold standard is used afterwards, here, to find
out whether the verdicts are worth anything.

The extractions are not recomputed. They are read from the results file of a
finished parsing run, so this costs one pass of checking and not a second pass
of parsing.

Examples:
    # two pages with the local model, to see the shape of the answers
    python scripts/run_self_check.py --run output/v4_qwen9b_raw_chunked.json \
        --out output/selfcheck_pilot.json --limit 2

    # the whole corpus on the paid provider
    python scripts/run_self_check.py --run output/v4_qwen9b_raw_chunked.json \
        --out output/selfcheck_v4.json --openrouter
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lib.evaluation import DEFAULT_MIN_GROUNDED, grounded_fraction
from src.lib.llm import client
from src.lib.parsers.llm_parser import LlmParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from try_llm_parser import add_provider_flag, apply_provider, load_pages

_print_lock = threading.Lock()


def log(message: str) -> None:
    """Print one line, without interleaving lines from other threads."""
    with _print_lock:
        print(message, flush=True)


def check_page(page: dict, record: dict) -> dict:
    """Run the self-check on one page, recording a failure instead of raising."""
    result = {
        "url": record["url"],
        "domain": record["domain"],
        "f1": record["f1"],
        "recall": record["recall"],
        "precision": record["precision"],
        # Recomputed here rather than read: the parsing run that produced this
        # file predates the grounding measurement, and recomputing it costs
        # nothing and is deterministic.
        "grounded": grounded_fraction(record["parsed_text"], page["html_text"]),
        "parse_fragments": record.get("fragments", 1),
    }
    started = time.monotonic()
    try:
        outcome = LlmParser().self_check_detailed(
            record["url"], page["html_text"], record["parsed_text"]
        )
    except Exception as error:  # noqa: BLE001 - one page must not stop the run
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        log(f"  [errore] {record['url']}: {type(error).__name__}: {error}")
        return result

    result.update(
        status="ok",
        seconds=round(time.monotonic() - started, 1),
        complete=outcome.result.complete,
        coherent=outcome.result.coherent,
        notes=outcome.result.notes,
        check_fragments=outcome.fragments,
        call_costs_usd=list(outcome.call_costs_usd),
        cost_usd=outcome.cost_usd,
        cost_eur=outcome.cost_eur,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        providers=list(outcome.providers),
        generation_ids=list(outcome.generation_ids),
        per_fragment=[
            {"complete": v.complete, "coherent": v.coherent, "notes": v.notes}
            for v in outcome.per_fragment
        ],
    )
    verdict = "PASSA" if outcome.result.coherent else "BOCCIA"
    pieces = f" {outcome.fragments} pezzi" if outcome.fragments > 1 else ""
    price = f" EUR {outcome.cost_eur:.4f}" if outcome.cost_usd else ""
    log(
        f"  [{verdict:6}] {record['url'][:58]:58} f1={record['f1']:.3f} "
        f"anc={result['grounded'] if result['grounded'] is not None else float('nan'):.2f}"
        f" {result['seconds']:.0f}s{pieces}{price}"
    )
    return result


def correlation(xs: list[float], ys: list[float]) -> float | None:
    """Return Pearson's r, or None when one of the series does not vary."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = sum(d * d for d in dx) ** 0.5
    sy = sum(d * d for d in dy) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


def report_providers(results: list[dict]) -> None:
    """Say who served the run, and complain if that is more than one company.

    OpenRouter routes each request on its own, so a run left unpinned can be
    answered by several providers serving the same model id at different
    quantisations - fp4, fp8 and bf16 were all on offer for qwen3.5-9b. The
    verdicts of such a run cannot be attributed to one model, and comparing it
    with another run measures the routing as much as the change under test.

    This has to be checked here rather than trusted, because asking for a
    provider and getting it are different things, and a run that silently
    changed weights halfway looks exactly like a run that did not.
    """
    served: dict[str, int] = {}
    for record in results:
        for name in record.get("providers") or []:
            served[name or "(non dichiarato)"] = served.get(name or "(non dichiarato)", 0) + 1
    if not served:
        return

    asked = os.environ.get("OPENROUTER_PROVIDER", "").strip()
    print(f"\nFORNITORE  richiesto: {asked or 'nessuno (instradamento libero)'}")
    for name, calls in sorted(served.items(), key=lambda item: -item[1]):
        print(f"  {name:<24} {calls:>3} chiamate")
    if len(served) > 1:
        print("  ATTENZIONE: piu' di un fornitore ha servito questa run, quindi i")
        print("  suoi numeri non appartengono a un solo modello. Fissare")
        print("  OPENROUTER_PROVIDER e rifarla prima di usarli per un confronto.")


def summarise(results: list[dict], threshold: float) -> None:
    """Print what the check said, and whether it agrees with the gold standard.

    A verdict is only useful if it separates the good extractions from the bad
    ones. The way to see that is not the accuracy of the flags but the F1 of
    the pages on each side of them: a check that passes everything has a
    perfect agreement rate and tells you nothing.
    """
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        print("\nnessuna pagina controllata")
        return

    # The verdict is the coherence answer alone. The completeness answer was
    # measured on this corpus and dropped: it tracks the gold standard at
    # r = +0.06, because answering it means noticing an absence and then
    # deciding whether the extraction rules authorised it, and the model
    # reports the absence without applying the filter.
    report_providers(ok)

    passed = [r for r in ok if r["coherent"]]
    failed = [r for r in ok if not r["coherent"]]
    print(f"\n{'=' * 66}\nAUTOVALUTAZIONE su {len(ok)} pagine")
    for name, group in (("promosse", passed), ("bocciate", failed)):
        score = f"f1 medio {mean([r['f1'] for r in group]):.3f}" if group else ""
        print(f"  {name}  {len(group):>3}   {score}")
    print(f"  di cui non complete: {sum(1 for r in ok if not r['complete'])}, "
          f"non coerenti: {sum(1 for r in ok if not r['coherent'])}")

    # The verdict is a flag, the F1 is a number: correlating them means asking
    # how far apart the two groups sit, which is exactly the question.
    r_check = correlation([float(r["coherent"]) for r in ok],
                          [r["f1"] for r in ok])
    r_complete = correlation([float(r["complete"]) for r in ok],
                             [r["f1"] for r in ok])
    anchored = [r for r in ok if r["grounded"] is not None]
    r_anchor = correlation([r["grounded"] for r in anchored],
                           [r["f1"] for r in anchored])
    print(f"\n{'CORRELAZIONE con f1':<34}")
    print(f"  verdetto (coerenza)        "
          f"{'n/d' if r_check is None else f'{r_check:+.3f}'}")
    print(f"  completezza (scartata)     "
          f"{'n/d' if r_complete is None else f'{r_complete:+.3f}'}")
    print(f"  ancoraggio (misura locale) "
          f"{'n/d' if r_anchor is None else f'{r_anchor:+.3f}'}  su {len(anchored)} pagine")

    below = [r for r in anchored if r["grounded"] < threshold]
    above = [r for r in anchored if r["grounded"] >= threshold]
    if below and above:
        print(f"  ancoraggio sotto {threshold:.1f}: {len(below)} pagine, "
              f"f1 medio {mean([r['f1'] for r in below]):.3f}; "
              f"sopra: {len(above)}, f1 medio {mean([r['f1'] for r in above]):.3f}")

    worst = sorted(ok, key=lambda r: r["f1"])[:5]
    print(f"\n{'le 5 pagine peggiori':<50} {'f1':>6} {'anc':>5} {'esito':>7}")
    for r in worst:
        verdict = "passa" if r["coherent"] else "boccia"
        anchor = "n/d" if r["grounded"] is None else f"{r['grounded']:.2f}"
        print(f"{r['url'][:50]:50} {r['f1']:>6.3f} {anchor:>5} {verdict:>7}")

    spent = sum(r.get("cost_eur") or 0 for r in ok)
    calls = sum(r.get("check_fragments") or 1 for r in ok)
    if spent:
        print(f"\ncosto del controllo EUR {spent:.4f} in {calls} chiamate")
    print("=" * 66)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True,
                        help="file JSON di una run di parsing, con --keep-text")
    parser.add_argument("--out", required=True, help="file JSON dei verdetti")
    add_provider_flag(parser)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--domain", help="limita a un dominio")
    parser.add_argument("--limit", type=int, help="controlla solo le prime N pagine")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MIN_GROUNDED,
                        help="soglia di ancoraggio usata nel confronto")
    args = parser.parse_args()
    apply_provider(args)

    run = json.load(open(args.run, encoding="utf-8"))
    records = [r for r in run["records"] if r["status"] == "ok" and r.get("parsed_text")]
    if not records:
        sys.exit(f"{args.run} non contiene testo estratto: rifai la run con --keep-text")

    by_url = {p["url"]: p for p in load_pages()}
    missing = [r["url"] for r in records if r["url"] not in by_url]
    if missing:
        sys.exit(f"{len(missing)} pagine della run non sono nel gold standard: {missing[:3]}")

    if args.domain:
        records = [r for r in records if r["domain"] == args.domain]
    if args.limit:
        records = records[: args.limit]

    done: dict[str, dict] = {}
    if os.path.exists(args.out):
        previous = json.load(open(args.out, encoding="utf-8"))
        # Only settled verdicts are kept, so an interrupted run retries what
        # failed instead of freezing a network error into the dataset.
        done = {r["url"]: r for r in previous["results"] if r["status"] == "ok"}
        print(f"ripresa: {len(done)} pagine gia' controllate in {args.out}")

    todo = [r for r in records if r["url"] not in done]
    print(f"modello   {client.get_model_name()}")
    richiesto = os.environ.get("OPENROUTER_PROVIDER", "").strip()
    print(f"fornitore {richiesto or 'NESSUNO - instradamento libero, run non confrontabile'}")
    print(f"run       {args.run}  ({run.get('condition')})")
    print(f"budget    {os.environ.get('LLM_PARSER_CONTEXT_TOKENS', 128000)} token")
    print(f"da fare   {len(todo)} pagine su {len(records)}, {args.workers} in parallelo\n")

    started = time.monotonic()
    results = list(done.values())
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(lambda r: check_page(by_url[r["url"]], r), todo):
                results.append(result)
                json.dump(
                    {
                        "model": client.get_model_name(),
                        "provider": os.environ.get("LLM_PROVIDER"),
                        "provider_pinned": os.environ.get(
                            "OPENROUTER_PROVIDER", ""
                        ),
                        "source_run": os.path.basename(args.run),
                        "condition": run.get("condition"),
                        "budget_tokens": int(
                            os.environ.get("LLM_PARSER_CONTEXT_TOKENS", 128000)
                        ),
                        "eur_per_usd": client.eur_per_usd(),
                        "results": results,
                    },
                    open(args.out, "w", encoding="utf-8"),
                    ensure_ascii=False,
                    indent=1,
                )

    print(f"\nfinito in {time.monotonic() - started:.0f}s -> {args.out}")
    summarise(results, args.threshold)


if __name__ == "__main__":
    main()
