"""Crawler config for www.cnbc.com."""

from crawl4ai import CrawlerRunConfig

CONFIG = CrawlerRunConfig(
    magic=True,
    excluded_tags=["style", "script", "link", "meta", "noscript"],
    excluded_selector=(
        # Global navigation & menus
        "#GlobalNavigation, #MakeItGlobalNavigation, "
        "[class*='CNBCGlobalNav'], "
        "[class*='nav-menu-'], "
        # Account / sign-in / sign-up
        "[class*='account-menu'], "
        "[class*='SignInMenu'], "
        "[class*='SignUpMenu'], "
        # Skip / jump links
        "[class*='JumpLink'], "
        # Watch-live button
        "[class*='WatchLive'], "
        # Cookie consent (OneTrust)
        "#onetrust-consent-sdk, #onetrust-banner-sdk, "
        "#onetrust-pc-sdk, .otFlat, .ot-iab-2, "
        # Article eyebrow (category breadcrumb) and publish date
        "[class*='ArticleHeader-eyebrow'], "
        "[class*='ArticleHeader-time'], "
        # Author info block
        "[class*='Author-author'], "
        "[class*='ArticleHeader-authorAndShareInline'], "
        "[class*='ArticleHeader-styles-makeit-time'], "
        "[class*='ArticleHeader-styles-makeit-authorAndShare'], "
        # Inline video embeds
        "[class*='InlineVideo'], "
        "[class*='PlaceHolder-wrapper'], "
        "[class*='InlineImage-styles-makeit-imageEmbed'], "
        # Inline image embeds
        "[class*='InlineImage-imageEmbed'], "
        # Stock ticker / related quotes widget and inline ticker buttons
        ".RelatedQuotes-relatedQuotes, "
        ".QuoteInBody-inlineButton, "
        # "Choose CNBC as your preferred source" banner
        "[class*='googlePreferredSourceContainer'], "
        # Special-report topic navigation
        "[class*='PageHeaderWithTuneInText'], "
        # Related content
        "[class*='RelatedStories'], "
        "[id*='RelatedStories'], "
        "[id*='RelatedVideo'], "
        "[class*='RelatedContent-styles-makeit-relatedContent'], "
        ".RelatedContent-relatedContent, "
        # Social
        ".social-buttons-transporter, "
        # Transporter / recommended-articles sections
        "[class*='TransporterSection'], "
        "[class*='SectionWrapper'], "
        # Stock data
        ".Collapsible-proliveCollapsableContainer, .InteractiveChart-container, .InteractiveChart-caption, "
        # Global footer
        "#GlobalFooter, [class*='CNBCFooter'], [class*='MakeItFooter-styles-makeit-container']"
    ),
    remove_forms=True,
)
