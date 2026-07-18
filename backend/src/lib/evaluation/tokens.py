"""Text tokenization and markdown stripping utilities.

Tokens are defined as whitespace-separated words.
This module is used for both gold standard loading and parser output evaluation.
"""

import html as html_mod
import re

import mistune
from bs4 import BeautifulSoup


_NUMBER_RE = re.compile(r'[+\-]?\d[\d.,]*\d%?|[+\-]?\d%?')


# Curly quotes and long dashes mapped to their plain ASCII form. Applied only
# when comparing texts, so that a token like l'arte counts the same whether the
# source uses a straight (') or a curly (’) apostrophe. The parser output and
# the displayed text are never changed by this.
_QUOTE_TABLE = str.maketrans({
    '‘': "'",   # left single quotation mark
    '’': "'",   # right single quotation mark / apostrophe
    '“': '"',   # left double quotation mark
    '”': '"',   # right double quotation mark
    '–': '-',   # en dash
    '—': '-',   # em dash
})


def normalize_for_compare(text: str) -> str:
    """Map curly quotes and dashes to ASCII and lowercase, only for comparison.

    Used when comparing texts to compute the metrics. The parser output and the
    displayed text are never changed.
    """
    return text.translate(_QUOTE_TABLE).lower()


def _canonicalize_number(token: str) -> str:
    """Drop thousand/decimal separators so 1,400 and 1.400 are the same token."""
    core = token.strip("()€£")
    if _NUMBER_RE.fullmatch(core):
        return re.sub(r'[.,]', '', core.rstrip('%'))
    return token


def extract_unique_tokens(text: str) -> set[str]:
    """Return the set of unique whitespace-separated tokens in the text.

    Newlines are treated as spaces. Empty strings are discarded. Numbers are
    normalised so they compare equal regardless of locale formatting, and
    typographic quotes/dashes are normalised so quote style does not matter.
    """
    normalized = normalize_for_compare(text).replace("\n", " ")
    return {
        _canonicalize_number(token)
        for token in normalized.split(" ")
        if token
    }


def strip_markdown(text: str) -> str:
    """Remove markdown syntax and return plain text content.

    Converts markdown to HTML via mistune, then uses BeautifulSoup to
    extract only the text (unwrapping all tags in-place so punctuation
    and spacing are preserved). Used before token-level evaluation so
    that formatting characters do not affect scores.

    Typographic characters (curly quotes, en/em dashes) are kept exactly
    as they appear, so the text is never altered character by character.
    """
    html = mistune.html(text)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(True):
        tag.unwrap()
    text = re.sub(r'[ \t]+', ' ', str(soup))   # collapse horizontal whitespace
    text = re.sub(r'\n+', '\n', text)           # collapse multiple newlines
    text = html_mod.unescape(text)              # decode &amp; → & etc.
    return text.strip()
