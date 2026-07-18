"""Crawler config for it.wikipedia.org."""

from crawl4ai import CrawlerRunConfig

CONFIG = CrawlerRunConfig(
    magic=True,
    excluded_tags=["style", "script", "link", "meta", "noscript", "nav"],
    excluded_selector=(
        # Global header and navigation
        ".vector-header-container, #siteNotice, #p-search, #p-lang-btn, .vector-page-toolbar, #siteSub, "
        # Infobox / sinottico tables
        ".infobox, .sinottico, "
        # Captions
        "figcaption, .itwiki-template-citazione-footer, "
        # Actions button
        ".mw-highlight-copy-button, .mw-editsection, "
        # Others
        ".ambox, .metadata, .thumbcaption"
        ),
    remove_forms=True,
)
