"""ESPN-specific content parser (www.espn.com).

Receives Crawl4AI markdown (filtered by CrawlerRunConfig) and removes
boilerplate sections and residual lines that may still appear after
the article body.
"""

import re

from ..base import ContentParser


class EspnParser(ContentParser):
    """Parser for ESPN pages.

    CrawlerRunConfig already removes nav/footer/sidebar/ads. This parser stops
    at known boilerplate section headings and drops residual boilerplate lines
    that can still leak through the markdown conversion.

    On player stats pages (/player/stats/) the useful content starts at the
    "Stats" heading, so we begin collecting from there. On every other page we
    collect from the top.
    """

    # Heading-level section names that signal the end of editorial content.
    EXCLUDED_SECTIONS = frozenset({
        "advertisement",
        "trending",
        "top stories",
        "more from espn",
        "related content",
        "you may also like",
        "related stories",
        "editors' picks",
    })

    # Footer/legal lines that signal the end of useful content on any page.
    # These patterns avoid the ^ and $ anchors on purpose, so they still match
    # when the text is inside a markdown list item (- Terms of Use) or a link
    # ([Terms of Use](...)).
    _FOOTER_PATTERNS: tuple[re.Pattern, ...] = (
        re.compile(r"terms of use", re.IGNORECASE),
        re.compile(r"privacy policy", re.IGNORECASE),
        re.compile(r"cookie policy", re.IGNORECASE),
        re.compile(r"interest.based ads", re.IGNORECASE),
        re.compile(r"eu privacy rights", re.IGNORECASE),
        re.compile(r"manage privacy preferences", re.IGNORECASE),
        re.compile(r"espn enterprises", re.IGNORECASE),
        re.compile(r"all rights reserved", re.IGNORECASE),
    )

    # Non-editorial lines to drop outright.
    _SKIP_PATTERNS: tuple[re.Pattern, ...] = (
        # "WATCH:" or "READ:" inline promos
        re.compile(r"^\**\s*(WATCH|READ)\s*:\**", re.IGNORECASE),
        # Lines that are only a markdown link (related articles, topic tags)
        re.compile(r"^\s*\[[^\]]+\]\([^)]+\)\s*$"),
        # List-item lines whose only content is a link, optionally wrapped in
        # bold markers (**). ESPN injects related-article promo blocks like:
        #   **-[Title](url)  /  - [Title](url)**  /  **-[Title](url)**
        re.compile(r"^\s*\**-\s*(?:[^\[\n]*:\s*)?\[[^\]]+\]\([^)]+\)\**\s*$"),
        # Social / follow lines
        re.compile(r"follow us on", re.IGNORECASE),
        # Photo caption (appears as a standalone italic line)
        re.compile(r"^\s*\*[^*]+\*\s*$"),
        # Web-font observer test string injected by ESPN's font loader
        re.compile(r"^BESbswy", re.IGNORECASE),
    )

    # Detects the start of the stats section on player stats pages.
    _STATS_HEADING = re.compile(r"^#+\s*Stats\s*$|^Stats\s*$", re.IGNORECASE)

    def parse(self, url: str, markdown: str) -> str:
        """Return cleaned Crawl4AI markdown for ESPN pages."""
        is_stats_page = "/player/stats/" in url
        collected: list[str] = []
        # On stats pages we wait for the "Stats" heading before collecting.
        collecting = not is_stats_page

        for line in markdown.split("\n"):
            if not collecting:
                if self._STATS_HEADING.match(line):
                    collecting = True
                    collected.append("Stats")
                continue

            if self._is_end_of_content(line):
                break

            if self._matches_any(line, self._SKIP_PATTERNS):
                continue

            collected.append(line)

        return "\n".join(self._without_trailing_blank_lines(collected))

    def _is_end_of_content(self, line: str) -> bool:
        """True if the line marks the end of the editorial content."""
        if self._matches_any(line, self._FOOTER_PATTERNS):
            return True
        heading = self._heading_text(line)
        if heading is not None and self._is_excluded_heading(heading):
            return True
        return False

    def _is_excluded_heading(self, heading: str) -> bool:
        """True if the heading starts a known boilerplate section."""
        return any(
            heading == section or heading.startswith(section)
            for section in self.EXCLUDED_SECTIONS
        )
