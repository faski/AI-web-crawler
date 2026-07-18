"""Wikipedia-specific content parser.

Receives Crawl4AI markdown (already filtered by CrawlerRunConfig) and
removes non-informative trailing sections plus residual boilerplate lines.
"""

import re

from ..base import ContentParser


class WikipediaParser(ContentParser):
    """Parser for Wikipedia articles (any language subdomain).

    How it works:
      1. Walk the Crawl4AI markdown line by line.
      2. When a heading matches an excluded section, stop collecting.
      3. Drop residual boilerplate lines (edit links, image residue, etc.).
      4. Return the cleaned markdown.
    """

    EXCLUDED_SECTIONS = frozenset({
        "collegamenti esterni", "note", "riferimenti",
        "bibliografia", "fonti", "altri progetti", "voci correlate",
    })

    _SKIP_PATTERNS: tuple[re.Pattern, ...] = (
        # Skip-navigation links: "[Vai al contenuto](...)", "[Salta a](...)"
        re.compile(r"^\[vai al\b", re.IGNORECASE),
        re.compile(r"^\[salta al?\b", re.IGNORECASE),
        re.compile(r"^\[modifica?\b", re.IGNORECASE),
        # Lines that are only wiki-style image/file markup residues
        re.compile(r"^\s*!\["),
    )

    # Footnote reference markers to strip inline: [1], [4], [12], [N 1], [N 2], etc.
    # Limited to 1-3 digit numbers so we don't strip year links like [1912].
    _FOOTNOTE_RE = re.compile(r"\[(?:\d{1,3}|N\s*\d+)\]")

    # Any markdown link -> keep only the link text.
    # Supports one level of nested parentheses in URLs/titles,
    # e.g. [Foo](/wiki/Foo_(bar) "Foo (bar)") -> "Foo"
    _LINK_RE = re.compile(r'\[([^\]]*)\]\((?:[^()]*|\([^()]*\))*\)')

    # Image residue that link/footnote substitutions may leave behind.
    _IMAGE_RESIDUE_RE = re.compile(r"^\s*!\[")

    def parse(self, url: str, markdown: str) -> str:
        """Remove non-informative sections and boilerplate from Crawl4AI markdown."""
        collected: list[str] = []

        for line in markdown.split("\n"):
            heading = self._heading_text(line)
            if heading is not None and heading in self.EXCLUDED_SECTIONS:
                break

            if self._matches_any(line, self._SKIP_PATTERNS):
                continue

            line = self._clean_inline_markup(line)

            if self._IMAGE_RESIDUE_RE.match(line):
                continue

            collected.append(line)

        return "\n".join(self._without_trailing_blank_lines(collected))

    def _clean_inline_markup(self, line: str) -> str:
        """Drop footnote markers, turn links into plain text, fix spacing."""
        line = self._FOOTNOTE_RE.sub("", line)
        line = self._LINK_RE.sub(r"\1", line)
        # Remove the space left before punctuation after stripping a link.
        line = re.sub(r'\s+([,;:])', r'\1', line)
        return line
