"""Check the passages the model says are missing from an extraction.

Asking the model "is it complete?" did not work: it correlates with the gold
standard at about zero, and the same prompt gives different answers. So we ask
for quotes instead, and the code checks them. A quote can be searched for; an
opinion cannot.
"""

from dataclasses import dataclass

from .grounding import is_anchored, normalised_text, visible_text

# Shorter than this and the quote could match by chance, so we report it but
# never count it as an omission.
MIN_QUOTE_CHARS = 25


@dataclass(frozen=True)
class QuoteCheck:
    """One quote the model said was missing, and where it actually is."""

    quote: str
    in_page: bool
    in_extraction: bool
    too_short: bool

    @property
    def verdict(self) -> str:
        """Say what the quote turned out to be.

        Only "confermata" is a real omission. The other two are the model
        getting it wrong, and the old boolean could not tell them apart.
        """
        if self.too_short:
            return "troppo corta"
        if not self.in_page:
            return "inventata"
        if self.in_extraction:
            return "presente"
        return "confermata"

    @property
    def is_omission(self) -> bool:
        """Return whether this claim is a real omission from the extraction."""
        return self.verdict == "confermata"


def check_quotes(
    quotes: list[str], markdown: str, html_text: str
) -> tuple[QuoteCheck, ...]:
    """Look for each quote in the page and in the extraction.

    Uses the same matching as the grounding measure, so "present" means the
    same thing everywhere and punctuation cannot turn a quote into an omission.
    """
    page = visible_text(html_text)
    extraction = normalised_text(markdown)
    checks = []
    for quote in quotes:
        text = quote.strip()
        if not text:
            continue
        checks.append(
            QuoteCheck(
                quote=text,
                in_page=is_anchored(text, page),
                in_extraction=is_anchored(text, extraction),
                too_short=len(text) < MIN_QUOTE_CHARS,
            )
        )
    return tuple(checks)


def omissions(checks: tuple[QuoteCheck, ...]) -> tuple[QuoteCheck, ...]:
    """Return only the claims that turned out to be real omissions."""
    return tuple(c for c in checks if c.is_omission)
