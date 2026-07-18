"""XE-specific content parser (www.xe.com).

Receives Crawl4AI markdown (filtered by CrawlerRunConfig) and returns it
cleaned of trailing boilerplate sections, CTA links and pipe-table noise.
"""

import re

from ..base import ContentParser


class XeParser(ContentParser):
    """Parser for XE pages (blog articles + currency converter).

    On top of the CrawlerRunConfig stripping (header, nav, footer, banners,
    Trustpilot block, CTA buttons), this parser:

      * stops at known boilerplate section headings,
      * stops at known plain-text boilerplate lines (sections that appear on
        the converter page without a markdown heading),
      * drops residual CTA links the gold standard never includes,
      * drops markdown horizontal rules,
      * turns markdown tables into one cell per line (separator rows dropped).
    """

    # Headings that terminate the content (everything after is boilerplate).
    EXCLUDED_SECTIONS = frozenset({
        "related articles", "related posts", "you may also like",
        "sign up", "get started", "popular currencies",
        "xe is trusted by millions around the globe",
        "send money destinations",
        "international money transfers done right",
    })

    # Headings to drop in place, without terminating the document.
    _DROP_SECTIONS = frozenset({
        "compare and save",
    })

    # Plain-text (non-heading) lines that terminate the content.
    _TERMINAL_LINES = frozenset({
        "send money destinations",
        "xe is trusted by millions around the globe",
        "trusted by",
        "loading popular currency pairs...",
    })

    # Plain-text lines that should be dropped individually.
    _DROP_LINES = frozenset({
        "did you know you can send money abroad with xe?",
        "add currency",
    })

    # Standalone boilerplate paragraphs (matched on a line prefix).
    _DROP_PREFIXES = (
        "we use the mid-market rate",
        "when you compare xe to leading banks",
        "the comparison savings are based on",
    )

    # CTA links the gold standard strips out.
    _SKIP_LINK_PATTERNS = (
        re.compile(r'\[\s*Speak to an FX specialist', re.IGNORECASE),
        re.compile(r'\[\s*Download the Global Currency Outlook', re.IGNORECASE),
    )

    # Inline image: ``![alt](url)``.
    _IMAGE_RE = re.compile(r'!\[([^\]]*)\]\([^)]*\)')
    # Site logo image, dropped entirely.
    _LOGO_RE = re.compile(r'!\[[^\]]*\]\([^)]*logo-xe[^)]*\)', re.IGNORECASE)
    # Inline link glued to the preceding text: ``rates[Send money](url)``.
    _GLUED_LINK_RE = re.compile(r'(\S)(\[[^\]]+\]\([^)]*\))')
    # Adjacent inline elements rendered without whitespace produce glued
    # camelCase tokens (``EuroEUR``, ``nationwideNationwide``); split them
    # back at the lowercase/uppercase boundary.
    _CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])')

    # Markdown table separator row, e.g. "| --- | --- |".
    _TABLE_SEP_RE = re.compile(r'^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$')

    # Markdown horizontal rule, e.g. "* * *" or "---".
    _HR_RE = re.compile(r'^(\*\s*){3,}$|^-{3,}$|^_{3,}$')

    # Space wrongly left inside bold markers before punctuation, e.g. "** ,".
    _BOLD_PUNCT_RE = re.compile(r'\*\* +([,.;:!?\)])')

    @staticmethod
    def _table_row_cells(line: str) -> list[str]:
        """Split a pipe-formatted table row into its non-empty cell values."""
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        return [cell for cell in cells if cell]

    def _preprocess(self, markdown: str) -> str:
        """Normalise images and glued links before the line-by-line filtering."""
        markdown = self._BOLD_PUNCT_RE.sub(r'**\1', markdown)
        markdown = self._LOGO_RE.sub('', markdown)
        # Keep only the image alt text.
        markdown = self._IMAGE_RE.sub(r'\1', markdown)
        # Move links glued onto preceding text onto their own line.
        markdown = self._GLUED_LINK_RE.sub(r'\1\n\2', markdown)
        # Separate glued camelCase tokens from adjacent inline elements.
        markdown = self._CAMEL_RE.sub(' ', markdown)
        return markdown

    def parse(self, url: str, markdown: str) -> str:
        """Return cleaned Crawl4AI markdown for XE pages."""
        markdown = self._preprocess(markdown)
        collected: list[str] = []

        for raw_line in markdown.split("\n"):
            line = raw_line.rstrip()
            stripped = line.strip()
            lowered = stripped.lower()
            heading = self._heading_text(line)

            if self._ends_content(heading, lowered):
                break

            if self._is_dropped_line(heading, lowered, stripped):
                continue

            # Pipe-formatted table data row: one cell per line.
            if stripped.startswith("|") and "|" in stripped[1:]:
                collected.extend(self._table_row_cells(stripped))
                continue

            collected.append(line)

        return "\n".join(self._without_trailing_blank_lines(collected))

    def _ends_content(self, heading: str | None, lowered: str) -> bool:
        """True if this line marks the end of the useful content."""
        if heading is not None and heading in self.EXCLUDED_SECTIONS:
            return True
        if lowered in self._TERMINAL_LINES:
            return True
        return False

    def _is_dropped_line(self, heading: str | None, lowered: str, stripped: str) -> bool:
        """True if this single line is boilerplate to drop (without stopping)."""
        if heading is not None and heading in self._DROP_SECTIONS:
            return True
        if lowered in self._DROP_LINES:
            return True
        if any(lowered.startswith(prefix) for prefix in self._DROP_PREFIXES):
            return True
        if self._HR_RE.match(stripped):
            return True
        if self._TABLE_SEP_RE.match(stripped):
            return True
        if self._matches_any(stripped, self._SKIP_LINK_PATTERNS):
            return True
        return False
