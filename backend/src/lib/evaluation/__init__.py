"""Evaluation module: token-level and similarity metrics."""

from .energy import LocalCost, local_cost
from .grounding import (
    DEFAULT_MIN_GROUNDED,
    grounded_fraction,
    is_anchored,
    normalised_text,
    visible_text,
)
from .omissions import MIN_QUOTE_CHARS, QuoteCheck, check_quotes, omissions
from .similarity import ContentMetrics, calculate_content_metrics
from .token_level import TokenLevelMetrics, calculate_token_level_metrics
from .tokens import extract_unique_tokens, strip_markdown

__all__ = [
    "TokenLevelMetrics",
    "calculate_token_level_metrics",
    "ContentMetrics",
    "calculate_content_metrics",
    "extract_unique_tokens",
    "strip_markdown",
    "grounded_fraction",
    "local_cost",
    "LocalCost",
    "visible_text",
    "normalised_text",
    "is_anchored",
    "DEFAULT_MIN_GROUNDED",
    "QuoteCheck",
    "check_quotes",
    "omissions",
    "MIN_QUOTE_CHARS",
]
