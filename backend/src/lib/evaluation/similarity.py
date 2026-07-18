"""Similarity evaluation: cosine, Jaccard, excess ratio.

These metrics work on how many times each word appears (a word-frequency
count), so they react both to repeated words and to how much content the
parser extracted compared to the gold text.
"""

import math
from collections import Counter
from dataclasses import dataclass

from .tokens import normalize_for_compare, strip_markdown


@dataclass(frozen=True)
class ContentMetrics:
    """Result of the frequency-based content evaluation (read-only).

    cosine       - how aligned the two word-frequency counts are (higher = more similar).
    jaccard      - shared words divided by all the distinct words in either text.
    excess_ratio - share of parsed words the gold text does not cover (lower is better).
    """

    cosine: float
    jaccard: float
    excess_ratio: float


def calculate_content_metrics(parsed_text: str, gold_text: str) -> ContentMetrics:
    """Compute cosine similarity, Jaccard similarity, and excess ratio.

    Markdown is stripped from both texts first, then each text is turned into a
    word-frequency count (how many times each word appears). Returns zeros for
    all metrics if either text is empty.
    """
    parsed_frequencies = Counter(normalize_for_compare(strip_markdown(parsed_text)).split())
    gold_frequencies = Counter(normalize_for_compare(strip_markdown(gold_text)).split())

    if not parsed_frequencies or not gold_frequencies:
        return ContentMetrics(cosine=0.0, jaccard=0.0, excess_ratio=0.0)

    return ContentMetrics(
        cosine=_cosine_similarity(parsed_frequencies, gold_frequencies),
        jaccard=_jaccard_similarity(parsed_frequencies, gold_frequencies),
        excess_ratio=_excess_ratio(parsed_frequencies, gold_frequencies),
    )


def _cosine_similarity(parsed_frequencies: Counter, gold_frequencies: Counter) -> float:
    """How aligned the two word-frequency counts are (1.0 = identical mix)."""
    dot_product = sum(
        count * gold_frequencies[word]
        for word, count in parsed_frequencies.items()
        if word in gold_frequencies
    )
    parsed_magnitude = math.sqrt(sum(count ** 2 for count in parsed_frequencies.values()))
    gold_magnitude = math.sqrt(sum(count ** 2 for count in gold_frequencies.values()))
    if not parsed_magnitude or not gold_magnitude:
        return 0.0
    return dot_product / (parsed_magnitude * gold_magnitude)


def _jaccard_similarity(parsed_frequencies: Counter, gold_frequencies: Counter) -> float:
    """Shared distinct words divided by all distinct words in either text."""
    parsed_words = set(parsed_frequencies)
    gold_words = set(gold_frequencies)
    union_size = len(parsed_words | gold_words)
    if not union_size:
        return 0.0
    return len(parsed_words & gold_words) / union_size


def _excess_ratio(parsed_frequencies: Counter, gold_frequencies: Counter) -> float:
    """Share of parsed word occurrences the gold text does not cover (lower = better)."""
    overlap = sum(
        min(count, gold_frequencies[word])
        for word, count in parsed_frequencies.items()
        if word in gold_frequencies
    )
    total_parsed = sum(parsed_frequencies.values())
    if not total_parsed:
        return 0.0
    return 1.0 - (overlap / total_parsed)
