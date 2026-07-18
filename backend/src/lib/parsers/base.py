"""Abstract base class for all content parsers."""

import re
from abc import ABC, abstractmethod


class ContentParser(ABC):
    """Interface that every domain-specific parser must implement."""

    @abstractmethod
    def parse(self, url: str, markdown: str) -> str:
        """Parse raw Crawl4AI markdown and return clean text.

        Args:
            url:      the source URL, used to apply page-specific rules.
            markdown: raw markdown produced by the crawler.

        Returns:
            Clean text ready for token-level evaluation.
        """
        raise NotImplementedError

    # --- Small helpers shared by the concrete parsers -------------------------
    # Every parser walks the markdown line by line, so these cover the steps
    # they all repeat: spotting a heading, matching boilerplate patterns, and
    # trimming the blank lines left at the end.

    @staticmethod
    def _heading_text(line: str) -> str | None:
        """Return the heading text (lowercased) if the line is a heading.

        A markdown heading starts with one or more '#'. For any other line we
        return None, so the caller can tell a real heading apart from a normal
        line of text.
        """
        if not line.startswith("#"):
            return None
        return line.lstrip("#").strip().lower()

    @staticmethod
    def _matches_any(text: str, patterns) -> bool:
        """Return True if any of the compiled regex patterns is found in text."""
        return any(pattern.search(text) for pattern in patterns)

    @staticmethod
    def _without_trailing_blank_lines(lines: list[str]) -> list[str]:
        """Return the same lines with the empty lines at the end removed."""
        cleaned = list(lines)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        return cleaned
