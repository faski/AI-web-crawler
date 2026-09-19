"""Measures how much of an extraction really appears on the page.

This is the reference-free check the project brief asks for - whether the
extracted text is complete and coherent with respect to the HTML - answered
with code rather than by asking the model about its own work.

It exists because of one page. On ``xe.com/en-eu/business/payments/`` the
model returned a whole article, "Amex Global Pay Is Shutting Down", that is
nowhere on the page: it lives inside a ``<script>``, in the JSON payload the
site's framework embeds to prefetch a different page. The extraction scored
F1 0.297 against the gold standard, which looked like a mediocre page rather
than a total failure. It was a total failure: the recall came from words like
"payments" and "business" appearing in both texts by coincidence, and once
the invented text is set aside nothing correct remains.

What separates that page from the other 39 is not its score but where its
words come from. Measured here, it anchors **0%** of its output in the
visible page against a median of 100%, which makes it the one page a
threshold can catch without catching anything else.

A warning about what this is for. As a **detector** it is excellent. As a
**filter** - deleting the unanchored lines - it was measured over the whole
corpus and made things worse: mean F1 fell from 0.934 to 0.926, because
verbatim matching does not survive small differences in spacing and
punctuation, and four correct pages lost recall.

So this module only reports, and it sits on the output side: it reads the
Markdown the parser already produced and never touches what the parser is
given. The extraction pipeline is raw HTML in, Markdown out, with chunking
as its only concession; measuring the result afterwards is the same kind of
act as scoring it against the gold standard, which the base project already
does with BeautifulSoup in ``evaluation/tokens.py``.
"""

import re

from bs4 import BeautifulSoup

# Words compared at a time when looking for a line in the page. Long enough
# that a run of common words cannot match by chance, short enough that one
# altered word in the middle of a sentence does not hide the whole line.
ANCHOR_WINDOW_WORDS = 8

# Below this share of anchored lines an extraction is treated as off-page.
# The corpus separates cleanly: the broken page anchors 0%, the next worst
# 72%, and everything else 94% or more.
DEFAULT_MIN_GROUNDED = 0.6

# Lines this short carry too few words to place, and are usually headings or
# list fragments that repeat text counted elsewhere.
MIN_LINE_CHARS = 40

_MARKDOWN_NOISE = re.compile(r"\[\d+\]|\*\*|[#>*|`_~-]")
_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)


def visible_text(html_text: str) -> str:
    """Return the page's text as a reader sees it, normalised for comparison.

    ``<script>`` and ``<style>`` are dropped: their contents are code and
    data, never shown, and treating them as page text is exactly the mistake
    this module exists to catch.
    """
    soup = BeautifulSoup(html_text, "lxml")
    for element in soup(["script", "style"]):
        element.decompose()
    return _normalise(soup.get_text(" ", strip=True))


def grounded_fraction(markdown: str, html_text: str) -> float | None:
    """Return the share of the extraction's long lines found on the page.

    Returns None when the extraction has no line long enough to place, which
    is not the same as a score of zero: there is nothing to judge, and
    reporting 0.0 would flag an empty answer as an invention.
    """
    visible = visible_text(html_text)
    lines = [line for line in markdown.split("\n") if len(line.strip()) >= MIN_LINE_CHARS]
    if not lines:
        return None
    return sum(is_anchored(line, visible) for line in lines) / len(lines)


def is_anchored(line: str, visible: str) -> bool:
    """Return whether any run of words from ``line`` occurs in ``visible``."""
    words = _normalise(line).split()
    if not words:
        return True
    if len(words) < ANCHOR_WINDOW_WORDS:
        return " ".join(words) in visible
    return any(
        " ".join(words[start : start + ANCHOR_WINDOW_WORDS]) in visible
        for start in range(len(words) - ANCHOR_WINDOW_WORDS + 1)
    )


def normalised_text(text: str) -> str:
    """Normalise text the way anchoring compares it.

    Public because the omission check needs the same normalisation: two
    different ones would disagree about what "present" means.
    """
    return _normalise(text)


def _normalise(text: str) -> str:
    """Return ``text`` lowercased, stripped of markup noise and punctuation."""
    without_noise = _MARKDOWN_NOISE.sub(" ", text)
    return " ".join(_PUNCTUATION.sub(" ", without_noise).lower().split())
