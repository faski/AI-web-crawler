"""CNBC-specific content parser (www.cnbc.com).

Receives Crawl4AI markdown (filtered by CrawlerRunConfig) and removes
boilerplate sections that may still appear after the article body.
"""

import re

from ..base import ContentParser


class CnbcParser(ContentParser):
    """Parser for CNBC news articles.

    CrawlerRunConfig already removes nav/footer/ads/newsletter blocks. This
    parser stops at known boilerplate section headings and drops residual
    boilerplate lines that can still leak through the markdown conversion.
    """

    # Heading-level section names that signal the end of editorial content.
    EXCLUDED_SECTIONS = frozenset({
        "related content",
        "more from cnbc",
        "trending now",
        "you may also like",
        "watch now",
        "sign up",
        "watch live",
        "more in",
        "sponsored content",
        "advertisement",
    })

    # If the LAST paragraph starts with one of these, that paragraph and
    # everything after it is removed. Bold markdown (**) is allowed because
    # CNBC renders these call-to-action lines in bold.
    _TRAILING_CUTOFF_PATTERNS: tuple[re.Pattern, ...] = (
        # "Want to ...?" course/newsletter CTAs (bold, italic, or plain)
        # e.g. "**Want to get ahead at work?**", "_Want to improve..."
        re.compile(r"^[*_]*\s*Want to\b", re.IGNORECASE),
    )

    # Non-editorial lines to drop outright.
    _SKIP_PATTERNS: tuple[re.Pattern, ...] = (
        # "Choose CNBC as your preferred source on Google News"
        re.compile(r"choose cnbc", re.IGNORECASE),
        # "Follow us on" social media lines
        re.compile(r"follow us on", re.IGNORECASE),
        # "Subscribe to CNBC" newsletter / app CTAs
        re.compile(r"subscribe to cnbc", re.IGNORECASE),
        # Standalone bracketed image labels, e.g. "[Photo: ...]" or "[VIDEO]"
        re.compile(r"^\[(?:photo|video|image|watch)[^\]]*\]\s*$", re.IGNORECASE),
        # Lines that are only a markdown link with no surrounding text
        re.compile(r"^\s*\[[^\]]+\]\([^)]+\)\s*$"),
        # "Read more:" or "Read also:" intra-article promos
        re.compile(r"^read (more|also)\s*:", re.IGNORECASE),
        # "WATCH: [link]" inline video promos (bold or plain)
        re.compile(r"^\**\s*WATCH\s*:\**", re.IGNORECASE),
        # "[Sign up](url) for our weekly newsletter..." inline CTA
        re.compile(r"^\[Sign up\]\(", re.IGNORECASE),
        # "CNBC's new online course" promotional lines (multi-line CTA fallback)
        re.compile(r"cnbc[''']s new online course", re.IGNORECASE),
    )

    def parse(self, url: str, markdown: str) -> str:
        """Return cleaned Crawl4AI markdown for CNBC pages."""
        collected: list[str] = []

        for line in markdown.split("\n"):
            heading = self._heading_text(line)
            if heading is not None and self._is_excluded_heading(heading):
                break

            line = self._clean_inline_markers(line)

            if self._matches_any(line, self._SKIP_PATTERNS):
                continue

            collected.append(line)

        collected = self._without_trailing_blank_lines(collected)
        collected = self._cut_trailing_cta(collected)
        return "\n".join(collected)

    def _is_excluded_heading(self, heading: str) -> bool:
        """True if the heading starts a known boilerplate section."""
        return any(
            heading == section or heading.startswith(section)
            for section in self.EXCLUDED_SECTIONS
        )

    def _clean_inline_markers(self, line: str) -> str:
        """Remove empty bold/italic markers and fix spacing before punctuation.

        e.g. "**Airbnb supplies** :" -> "Airbnb supplies:"
        """
        line = re.sub(r'\*{1,3}\s*\*{1,3}', '', line)
        line = re.sub(r'\s+([,;:])', r'\1', line)
        return line

    def _cut_trailing_cta(self, lines: list[str]) -> list[str]:
        """Drop a trailing "Want to ...?" CTA paragraph and everything after it.

        We scan from the bottom up so adjacent non-blank lines don't hide the
        match inside a larger paragraph.
        """
        for index in range(len(lines) - 1, -1, -1):
            if any(pattern.match(lines[index]) for pattern in self._TRAILING_CUTOFF_PATTERNS):
                # Also drop the blank lines that came right before the CTA.
                while index > 0 and not lines[index - 1].strip():
                    index -= 1
                return lines[:index]
        return lines
