"""Default no-op parser used for unsupported domains."""

from .base import ContentParser

class PassThroughParser(ContentParser):
    """Returns the raw markdown unchanged.

    Used as a fallback when no domain-specific parser is registered.
    """

    def parse(self, url: str, markdown: str) -> str:
        return markdown
