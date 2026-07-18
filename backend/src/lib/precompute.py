"""Precompute and cache the quantitative metrics for every GS entry.

Run at backend startup so ``/db_stats`` can serve ``avg_eval`` immediately.
The judge score is not precomputed: it is populated on demand by
``/full_gs_eval`` to keep the boot time reasonable (~1-2 min vs ~10+ min).
"""

from .crawling.crawler import fetch_page_from_html
from .db import queries
from .evaluation.similarity import calculate_content_metrics
from .evaluation.token_level import calculate_token_level_metrics
from .parsers import get_parser_for_url


async def precompute_quantitative_evaluations() -> int:
    """Compute and cache token-level + similarity for every GS URL missing them."""
    missing = queries.get_urls_missing_quantitative_eval()
    if not missing:
        return 0

    for url in missing:
        entry = queries.get_gold_standard(url)
        if entry is None:
            continue
        try:
            page = await fetch_page_from_html(entry.url, entry.html_text)
        except RuntimeError:
            continue
        parsed_text = get_parser_for_url(entry.url).parse(entry.url, page.markdown_text)
        token_level = calculate_token_level_metrics(parsed_text, entry.gold_text)
        similarity = calculate_content_metrics(parsed_text, entry.gold_text)
        queries.save_quantitative_eval(
            entry.url,
            token_level.precision, token_level.recall, token_level.f1,
            similarity.cosine, similarity.jaccard, similarity.excess_ratio,
        )

    return len(missing)
