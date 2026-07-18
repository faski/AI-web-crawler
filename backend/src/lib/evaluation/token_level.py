"""Token-level evaluation (set-based): precision, recall, F1.

Operates on token *sets* (presence/absence), treating each unique token
equally regardless of how many times it appears.
"""

from dataclasses import dataclass

from .tokens import extract_unique_tokens, strip_markdown


@dataclass(frozen=True)
class TokenLevelMetrics:
    """Immutable result of a token-level evaluation (set-based)."""

    extracted_count: int
    sample_count: int
    intersection_count: int
    precision: float
    recall: float
    f1: float


def calculate_token_level_metrics(parsed_text: str, gold_text: str) -> TokenLevelMetrics:
    """Compute precision, recall, and F1 between parsed and gold text.

    Both texts are passed through strip_markdown before tokenization so that
    markdown formatting is removed and HTML entities are decoded symmetrically.
    Tokenization also normalises typographic quotes/dashes, so quote style does
    not change the score. Returns zero for all scores if either text is empty.
    """
    extracted_tokens = extract_unique_tokens(strip_markdown(parsed_text))
    sample_tokens = extract_unique_tokens(strip_markdown(gold_text))

    intersection_count = len(extracted_tokens & sample_tokens)
    extracted_count = len(extracted_tokens)
    sample_count = len(sample_tokens)

    precision = intersection_count / extracted_count if extracted_count else 0.0
    recall = intersection_count / sample_count if sample_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return TokenLevelMetrics(
        extracted_count=extracted_count,
        sample_count=sample_count,
        intersection_count=intersection_count,
        precision=precision,
        recall=recall,
        f1=f1,
    )
