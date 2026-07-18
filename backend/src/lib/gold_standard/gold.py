"""Gold standard data access (backed by MariaDB).

Thin wrappers over ``db.queries`` kept stable so the rest of the codebase
(route handlers, crawler fallback) doesn't depend on the DB layer directly.
The dict shape returned here matches what callers already expect.
"""

from ..db import queries


def get_entry_for_url(url: str) -> dict | None:
    """Return the gold standard entry for the URL, or None if not found.

    The returned dict has keys: url, domain, title, html_text, gold_text.
    """
    entry = queries.get_gold_standard(url)
    return entry.model_dump() if entry else None


def load_gold_text(url: str) -> str | None:
    """Return only the gold_text for the URL, or None if not found."""
    entry = queries.get_gold_standard(url)
    return entry.gold_text if entry else None
